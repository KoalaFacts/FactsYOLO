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

/// Source tokens (after `compact`) for networking types, raw system
/// calls and foreign imports. Exported `extern "C" fn` items are allowed;
/// `extern` blocks that import foreign functions are not.
const FORBIDDEN_TOKENS: [&str; 8] = [
    "TcpStream",
    "TcpListener",
    "UdpSocket",
    "libc::",
    "syscall",
    "asm!",
    "use std as",
    "extern crate std as",
];

/// First path segments under `std` that give networking, file or process access.
const FORBIDDEN_STD_MODULES: [&str; 5] = ["fs", "net", "os", "process", "self"];

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
/// `open64` stays allowed because the Rust runtime reads its own image for
/// panic backtraces; std file APIs are caught by the source check instead.
const FORBIDDEN_SYMBOLS: [&str; 37] = [
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
    "open",
    "openat",
    "openat64",
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

/// Removes `//` and `/* */` comments and collapses whitespace (a single space
/// survives only between two identifier characters), keeping string literals (a forbidden token inside a string is reported conservatively).
fn compact(text: &str) -> String {
    let mut out = String::new();
    let mut pending_space = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '/' && chars.peek() == Some(&'/') {
            for c in chars.by_ref() {
                if c == '\n' {
                    break;
                }
            }
        } else if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            let mut previous = ' ';
            for c in chars.by_ref() {
                if previous == '*' && c == '/' {
                    break;
                }
                previous = c;
            }
        } else if c.is_whitespace() {
            pending_space = true;
        } else {
            let word = |c: char| c.is_alphanumeric() || c == '_';
            if pending_space && out.chars().last().is_some_and(word) && word(c) {
                out.push(' ');
            }
            pending_space = false;
            out.push(c);
        }
    }
    out
}

/// Collects the first path segment of every item in a `std::` path or
/// `std::{...}` group, recursing into nested groups.
fn std_segments(rest: &str, segments: &mut Vec<String>) {
    if let Some(group) = rest.strip_prefix('{') {
        let mut depth = 0usize;
        let mut item = String::new();
        for c in group.chars() {
            match c {
                '{' => depth += 1,
                '}' if depth == 0 => break,
                '}' => depth -= 1,
                ',' if depth == 0 => {
                    std_segments(&item, segments);
                    item.clear();
                    continue;
                }
                _ => {}
            }
            item.push(c);
        }
        std_segments(&item, segments);
    } else {
        let segment: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !segment.is_empty() {
            segments.push(segment);
        }
    }
}

/// True when the compacted code imports foreign functions through an
/// `extern "ABI" { ... }` or `extern { ... }` block.
fn has_extern_block(code: &str) -> bool {
    code.match_indices("extern").any(|(index, _)| {
        let after = code[index + "extern".len()..].trim_start();
        let after = match after.strip_prefix('"') {
            Some(abi) => abi.find('"').map_or("", |end| &abi[end + 1..]),
            None => after,
        };
        after.starts_with('{')
    })
}

/// Returns `path: finding` for networking, file, process or foreign-call
/// access in one source file.
pub(crate) fn forbidden_source(path: &str, text: &str) -> Vec<String> {
    let code = compact(text);
    let mut found = Vec::new();
    for token in FORBIDDEN_TOKENS {
        if code.contains(token) {
            found.push(format!("{path}: {token}"));
        }
    }
    let mut segments = Vec::new();
    for (index, _) in code.match_indices("std::") {
        let boundary = code[..index]
            .chars()
            .last()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        if boundary {
            std_segments(&code[index + "std::".len()..], &mut segments);
        }
    }
    segments.sort();
    segments.dedup();
    for segment in segments {
        if FORBIDDEN_STD_MODULES.contains(&segment.as_str()) {
            found.push(format!("{path}: std::{segment}"));
        }
    }
    if has_extern_block(&code) {
        found.push(format!("{path}: extern block"));
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

/// Directories of path (local) crates in `cargo tree --prefix none` output,
/// whose lines read `name vX.Y.Z (/absolute/path)`.
pub(crate) fn local_crates(tree: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = tree
        .lines()
        .filter_map(|line| line.split_once(" (/").map(|(_, rest)| rest))
        .filter_map(|rest| {
            rest.split_once(')')
                .map(|(path, _)| PathBuf::from(format!("/{path}")))
        })
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
    command.args([subcommand, "-p", "facts-yolo", "--locked", "--offline"]);
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
    let mut violations = Vec::new();
    let mut scanned = std::collections::BTreeSet::new();
    for features in DEPLOYED_FEATURES {
        let label = if features.is_empty() {
            "default"
        } else {
            features
        };
        let mut tree = cargo(root, "tree", features);
        tree.args(["-e", "normal", "--prefix", "none"]);
        let tree = captured(&mut tree)?;
        let crates = forbidden_crates(&tree);
        violations.extend(crates.iter().map(|name| format!("{label}: crate {name}")));
        let locals = local_crates(&tree);
        if locals.is_empty() {
            return Err(format!("{label}: cargo tree listed no local crate"));
        }
        for directory in locals {
            let mut files = Vec::new();
            rust_files(&directory.join("src"), &mut files)?;
            for file in files {
                if !scanned.insert(file.clone()) {
                    continue;
                }
                let text = std::fs::read_to_string(&file).map_err(|e| format!("{file:?}: {e}"))?;
                let relative = file
                    .strip_prefix(root)
                    .unwrap_or(&file)
                    .display()
                    .to_string();
                violations.extend(forbidden_source(&relative, &text));
            }
        }
        println!(
            "{label}: {} local source files scanned so far",
            scanned.len()
        );
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
        assert!(found.contains(&"lib.rs: std::net".to_owned()));
        assert!(found.contains(&"lib.rs: TcpStream".to_owned()));
        assert!(found.contains(&"lib.rs: std::fs".to_owned()));
    }

    #[test]
    fn source_scan_sees_grouped_aliased_and_spaced_std_paths() {
        assert_eq!(
            forbidden_source("a.rs", "use std::{ptr, fs as f};"),
            ["a.rs: std::fs"]
        );
        assert_eq!(
            forbidden_source("b.rs", "use std::{io::{self, Read}, os::unix::net};"),
            ["b.rs: std::os"]
        );
        assert_eq!(
            forbidden_source("c.rs", "use std :: {\n    process\n};"),
            ["c.rs: std::process"]
        );
        assert_eq!(
            forbidden_source("d.rs", "use std as s; s::fs::write(p, b);"),
            ["d.rs: use std as"]
        );
        assert_eq!(
            forbidden_source("e.rs", "use ::std::{self as s};"),
            ["e.rs: std::self"]
        );
    }

    #[test]
    fn source_scan_flags_foreign_imports_but_not_exports() {
        let block = "unsafe extern \"C\" { fn open(p: *const u8, f: i32) -> i32; }";
        assert_eq!(forbidden_source("a.rs", block), ["a.rs: extern block"]);
        assert_eq!(
            forbidden_source("b.rs", "extern { fn write(); }"),
            ["b.rs: extern block"]
        );
        let export = "pub unsafe extern \"C\" fn fy_abi_query() -> i32 { 0 }";
        assert!(forbidden_source("c.rs", export).is_empty());
        assert_eq!(
            forbidden_source("d.rs", "libc::syscall(41, 2, 1, 0);"),
            ["d.rs: libc::", "d.rs: syscall"]
        );
    }

    #[test]
    fn source_scan_ignores_comments_and_clean_code() {
        let text = "// never use std::net here\nlet x = 1; /* std::fs\n is banned */\nuse std::{ptr, slice};\nmystd::fs::x();\n";
        assert!(forbidden_source("lib.rs", text).is_empty());
    }

    #[test]
    fn local_crates_come_from_cargo_tree_paths() {
        let tree = "facts-yolo v0.1.0 (/r/crates/facts-yolo)\nsha2 v0.10.9\nfy-tiny-model v0.1.0 (/r/experiments/cpu-detector/model)\nfacts-yolo v0.1.0 (/r/crates/facts-yolo) (*)\n";
        assert_eq!(
            local_crates(tree),
            [
                PathBuf::from("/r/crates/facts-yolo"),
                PathBuf::from("/r/experiments/cpu-detector/model")
            ]
        );
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
        assert_eq!(
            forbidden_imports("U openat@GLIBC_2.4\nU open\n"),
            ["open", "openat"]
        );
    }
}
