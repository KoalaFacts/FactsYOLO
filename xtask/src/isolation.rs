// SPDX-License-Identifier: Apache-2.0
//! Deployed-library isolation gate (TD2 spec section 2, roadmap E2a).
//!
//! The deployed library must contain no networking code and must not create,
//! write, rename or delete files. Three independent checks enforce that for
//! every deployable feature set: the crate source, the normal dependency graph
//! and the dynamic symbols the built Linux `cdylib` imports.
use std::path::{Path, PathBuf};
use std::process::Command;

/// Feature sets shipped to customers. Contract fixtures are TEST_ONLY.
const DEPLOYED_FEATURES: [&str; 2] = ["", "td1"];

/// Source tokens that would bring networking or file mutation into the crate.
const FORBIDDEN_SOURCE: [&str; 9] = [
    "std::net",
    "TcpStream",
    "TcpListener",
    "UdpSocket",
    "std::fs",
    "File::create",
    "OpenOptions",
    "std::os::unix::net",
    "std::process",
];

/// Crates whose presence in the deployed graph means networking capability.
const FORBIDDEN_CRATES: [&str; 18] = [
    "async-std",
    "curl",
    "h2",
    "http",
    "hyper",
    "isahc",
    "libc-net",
    "mio",
    "native-tls",
    "openssl",
    "quinn",
    "reqwest",
    "rustls",
    "socket2",
    "surf",
    "tokio",
    "tungstenite",
    "ureq",
];

/// Imported C symbols for sockets, name resolution and file mutation.
/// `open64` stays allowed: the Rust runtime reads its own image for panic
/// backtraces; file creation is caught by the source check instead.
const FORBIDDEN_SYMBOLS: [&str; 34] = [
    "accept",
    "accept4",
    "bind",
    "connect",
    "creat",
    "creat64",
    "fopen",
    "fopen64",
    "freeaddrinfo",
    "ftruncate",
    "ftruncate64",
    "getaddrinfo",
    "gethostbyaddr",
    "gethostbyname",
    "gethostbyname_r",
    "getnameinfo",
    "listen",
    "mkdir",
    "mkdirat",
    "recv",
    "recvfrom",
    "recvmsg",
    "rename",
    "renameat",
    "rmdir",
    "send",
    "sendmsg",
    "sendto",
    "setsockopt",
    "socket",
    "socketpair",
    "truncate",
    "unlink",
    "unlinkat",
];

/// Returns `path: token` for every forbidden token in a source file.
pub(crate) fn forbidden_source(path: &str, text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let code = line.split("//").next().unwrap_or_default();
        for token in FORBIDDEN_SOURCE {
            if code.contains(token) {
                found.push(format!("{path}:{}: {token}", number + 1));
            }
        }
    }
    found
}

/// Parses `cargo tree --prefix none` output and returns forbidden crate names.
pub(crate) fn forbidden_crates(tree: &str) -> Vec<String> {
    let mut found: Vec<String> = tree
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| FORBIDDEN_CRATES.contains(name))
        .map(str::to_owned)
        .collect();
    found.sort();
    found.dedup();
    found
}

/// Parses `nm -D --undefined-only` output and returns forbidden imports.
pub(crate) fn forbidden_imports(nm: &str) -> Vec<String> {
    let mut found: Vec<String> = nm
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .map(|symbol| symbol.split('@').next().unwrap_or(symbol))
        .filter(|symbol| FORBIDDEN_SYMBOLS.contains(symbol))
        .map(str::to_owned)
        .collect();
    found.sort();
    found.dedup();
    found
}

fn rust_files(directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(directory).map_err(|e| format!("{directory:?}: {e}"))?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            rust_files(&path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

fn captured(command: &mut Command) -> Result<String, String> {
    eprintln!("+ {command:?}");
    let output = command
        .output()
        .map_err(|e| format!("cannot execute {command:?}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{command:?} exited {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

fn cargo(root: &Path, subcommand: &str, features: &str) -> Command {
    let executable = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(executable);
    command.current_dir(root);
    command.args([
        subcommand,
        "-p",
        "facts-yolo",
        "--locked",
        "--offline",
        "--no-default-features",
    ]);
    if !features.is_empty() {
        command.args(["--features", features]);
    }
    command
}

pub(super) fn run() -> Result<(), String> {
    if !cfg!(target_os = "linux") {
        return Err("the isolation gate reads ELF imports and requires Linux".to_owned());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("workspace root is absent")?;
    let mut files = Vec::new();
    rust_files(&root.join("crates/facts-yolo/src"), &mut files)?;
    files.sort();
    let mut violations = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).map_err(|e| format!("{file:?}: {e}"))?;
        let relative = file
            .strip_prefix(root)
            .unwrap_or(file)
            .display()
            .to_string();
        violations.extend(forbidden_source(&relative, &text));
    }
    println!("source: {} files scanned", files.len());
    for features in DEPLOYED_FEATURES {
        let label = if features.is_empty() {
            "default"
        } else {
            features
        };
        let mut tree = cargo(root, "tree", features);
        tree.args(["-e", "normal", "--prefix", "none"]);
        let crates = forbidden_crates(&captured(&mut tree)?);
        violations.extend(crates.iter().map(|name| format!("{label}: crate {name}")));
        let target = root.join(format!("target/isolation-{label}"));
        let mut build = cargo(root, "build", features);
        build.arg("--release").arg("--target-dir").arg(&target);
        captured(&mut build)?;
        let library = target.join("release/libfacts_yolo.so");
        let nm = captured(
            Command::new("nm")
                .args(["-D", "--undefined-only"])
                .arg(&library),
        )?;
        let imports = nm.lines().filter(|line| !line.trim().is_empty()).count();
        if imports == 0 {
            return Err(format!(
                "{}: no imports read; nm output unusable",
                library.display()
            ));
        }
        let symbols = forbidden_imports(&nm);
        violations.extend(
            symbols
                .iter()
                .map(|name| format!("{label}: imports {name}")),
        );
        println!(
            "{label}: {imports} imports checked, {} forbidden",
            symbols.len()
        );
    }
    if violations.is_empty() {
        println!("isolation gate passed: no networking or file-mutation code in deployed builds");
        Ok(())
    } else {
        Err(format!("isolation gate failed:\n{}", violations.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_scan_flags_networking_and_file_writes() {
        let text = "use std::net::TcpStream;\nlet f = std::fs::File::create(p);\n";
        let found = forbidden_source("lib.rs", text);
        assert!(found.contains(&"lib.rs:1: std::net".to_owned()));
        assert!(found.contains(&"lib.rs:1: TcpStream".to_owned()));
        assert!(found.contains(&"lib.rs:2: std::fs".to_owned()));
        assert!(found.contains(&"lib.rs:2: File::create".to_owned()));
    }

    #[test]
    fn source_scan_ignores_comments_and_clean_code() {
        let text = "// never use std::net here\nlet x = 1; // std::fs is banned\n";
        assert!(forbidden_source("lib.rs", text).is_empty());
    }

    #[test]
    fn crate_scan_reads_cargo_tree_names() {
        let tree =
            "facts-yolo v0.1.0 (/x)\nsha2 v0.10.9\ntokio v1.0.0\nhyper v1.0.0\ntokio v1.0.0 (*)\n";
        assert_eq!(forbidden_crates(tree), ["hyper", "tokio"]);
        assert!(forbidden_crates("sha2 v0.10.9\nhttparse v1.0.0\n").is_empty());
    }

    #[test]
    fn import_scan_strips_versions_and_flags_sockets() {
        let nm = "                 U socket@GLIBC_2.2.5\n                 U connect\n                 w __gmon_start__\n                 U open64@GLIBC_2.2.5\n                 U getaddrinfo@GLIBC_2.2.5\n";
        assert_eq!(forbidden_imports(nm), ["connect", "getaddrinfo", "socket"]);
    }

    #[test]
    fn import_scan_allows_the_current_runtime_imports() {
        let nm = "U malloc\nU open64@GLIBC_2.2.5\nU read\nU write\nU writev\nU readlink\nU close\n";
        assert!(forbidden_imports(nm).is_empty());
    }
}
