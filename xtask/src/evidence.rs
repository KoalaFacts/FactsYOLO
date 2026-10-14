// SPDX-License-Identifier: Apache-2.0
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) fn native_compiler() -> std::ffi::OsString {
    std::env::var_os("CC").unwrap_or_else(|| if cfg!(windows) { "cl" } else { "cc" }.into())
}

pub(super) fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace parent")
}

pub(super) fn cargo() -> Command {
    let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command.current_dir(root());
    command
}

pub(super) fn start(label: &str) -> Result<PathBuf, String> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let directory = root()
        .join("target/m1-evidence")
        .join(format!("{label}-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    Ok(directory)
}

pub(super) fn run(directory: &Path, name: &str, command: &mut Command) -> Result<String, String> {
    eprintln!("+ {command:?}");
    let mut commands = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join("commands.tsv"))
        .map_err(|error| error.to_string())?;
    let output = command.output();
    let code = match &output {
        Ok(output) => output.status.to_string(),
        Err(error) => format!("spawn failed: {error}"),
    };
    writeln!(commands, "{name}\t{command:?}\t{code}").map_err(|error| error.to_string())?;
    let output = output.map_err(|error| format!("{command:?}: {error}"))?;
    std::fs::write(directory.join(format!("{name}.stdout.log")), &output.stdout)
        .map_err(|error| error.to_string())?;
    std::fs::write(directory.join(format!("{name}.stderr.log")), &output.stderr)
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "{command:?} exited {}; logs: {}\n{}\n{}",
            output.status,
            directory.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| error.to_string())
}

pub(super) fn manifest(directory: &Path) -> Result<(), String> {
    let revision = run(
        directory,
        "git-head",
        Command::new("git")
            .current_dir(root())
            .args(["rev-parse", "HEAD"]),
    )?;
    let tree = run(
        directory,
        "git-tree",
        Command::new("git")
            .current_dir(root())
            .args(["rev-parse", "HEAD^{tree}"]),
    )?;
    let dirty = run(
        directory,
        "git-status",
        Command::new("git")
            .current_dir(root())
            .args(["status", "--porcelain=v1"]),
    )?;
    let rust = run(
        directory,
        "rust-version",
        Command::new("rustc").args(["--version", "--verbose"]),
    )?;
    let cargo = run(directory, "cargo-version", cargo().arg("--version"))?;
    let mut text = format!(
        "inventory only; success claims require this run's commands.tsv and outcome.txt\nrevision={}\ncommitted_tree={}\nworktree_status={dirty:?}\nos={}\narch={}\nfeatures=normal:none;fixture:contract-fixtures\nfixture=basic-v1\nresource_workload=1000 warmup;10000 measured;100-cycle batches;16MiB median delta\n{rust}{cargo}",
        revision.trim(),
        tree.trim(),
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    text.push_str(&format!(
        "FACTSYOLO_BUILD_ID={:?}\n",
        std::env::var("FACTSYOLO_BUILD_ID").ok()
    ));
    let mut tools = vec![
        ("dotnet", "dotnet".into(), "--info"),
        (
            "native-compiler",
            native_compiler(),
            if cfg!(windows) { "/?" } else { "--version" },
        ),
        (
            "symbol-tool",
            if cfg!(windows) { "dumpbin" } else { "nm" }.into(),
            if cfg!(windows) {
                "/headers"
            } else {
                "--version"
            },
        ),
    ];
    if let Some(compiler) = std::env::var_os("CC_I686") {
        tools.push(("i686-compiler", compiler, "--version"));
    }
    for (name, tool, arg) in tools {
        // A standalone inventory records missing tools explicitly. The verify
        // preflight separately requires them; resource-only Rust jobs need not
        // have .NET installed merely to measure their process allocations.
        let mut command = Command::new(tool);
        command.current_dir(root().join("tests/dotnet")).arg(arg);
        if name == "symbol-tool" && cfg!(windows) {
            command.arg(std::env::current_exe().map_err(|error| error.to_string())?);
        }
        match run(directory, name, &mut command) {
            Ok(_) => text.push_str(&format!(
                "{name}: recorded in {name}.stdout.log / {name}.stderr.log\n"
            )),
            Err(error) => text.push_str(&format!("{name}: UNAVAILABLE ({error})\n")),
        }
    }
    std::fs::write(directory.join("manifest.txt"), text).map_err(|error| error.to_string())
}
