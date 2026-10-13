// SPDX-License-Identifier: Apache-2.0
//! Executes the unpublished .NET consumer against explicit native build paths.
use std::path::Path;
use std::process::Command;

fn checked(command: &mut Command) -> Result<(), String> {
    eprintln!("+ {command:?}");
    let status = command.status().map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{command:?} exited {status}"))
    }
}

fn capture(command: &mut Command) -> Result<String, String> {
    eprintln!("+ {command:?}");
    let output = command.output().map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "{command:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| error.to_string())
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    let host = if cfg!(target_os = "linux") {
        "linux-x64"
    } else if cfg!(windows) {
        "win-x64"
    } else {
        return Err(".NET conformance currently requires Linux/Windows x64".to_owned());
    };
    let (mode, rid) = match args {
        [flag, mode] if flag == "--mode" => (mode.as_str(), host),
        [flag, mode, rid_flag, rid] if flag == "--mode" && rid_flag == "--rid" => {
            (mode.as_str(), rid.as_str())
        }
        _ => return Err("expected --mode jit|aot [--rid linux-x64|win-x64]".to_owned()),
    };
    if !matches!(mode, "jit" | "aot") || rid != host || std::env::consts::ARCH != "x86_64" {
        return Err("unsupported execution mode or non-native RID".to_owned());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("workspace root absent")?;
    let dotnet_root = root.join("tests/dotnet");
    let sdk = capture(
        Command::new("dotnet")
            .current_dir(&dotnet_root)
            .arg("--version"),
    )?;
    if sdk.trim() != "10.0.401" {
        return Err(format!("expected pinned SDK 10.0.401, got {sdk}"));
    }
    super::native::run(true)?;
    let evidence = root.join("target/dotnet-conformance").join(rid);
    std::fs::create_dir_all(&evidence).map_err(|error| error.to_string())?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let layout = capture(Command::new(cargo).current_dir(root).args([
        "run",
        "-p",
        "facts-yolo",
        "--example",
        "abi_layout",
        "--locked",
        "--offline",
        "--quiet",
    ]))?;
    let layout_path = evidence.join("rust-layout.txt");
    std::fs::write(&layout_path, layout).map_err(|error| error.to_string())?;
    let destination = evidence.join(mode);
    let project = dotnet_root.join("FactsYOLO.Smoke/FactsYOLO.Smoke.csproj");
    let mut build = Command::new("dotnet");
    build
        .current_dir(&dotnet_root)
        .arg(if mode == "aot" { "publish" } else { "build" });
    build
        .arg(&project)
        .args(["-c", "Release", "-o"])
        .arg(&destination);
    if mode == "aot" {
        build.args(["-r", rid, "--self-contained", "true", "-p:PublishAot=true"]);
    }
    checked(&mut build)?;
    let executable = destination.join(if mode == "jit" {
        "FactsYOLO.Smoke.dll"
    } else if cfg!(windows) {
        "FactsYOLO.Smoke.exe"
    } else {
        "FactsYOLO.Smoke"
    });
    if !executable.is_file() {
        return Err("compiled consumer is absent".to_owned());
    }
    let mut failures = Vec::new();
    for (target, kind) in [("normal", "normal"), ("contract-fixtures", "fixture")] {
        let library = root
            .join("target")
            .join(target)
            .join("release")
            .join(if cfg!(windows) {
                "facts_yolo.dll"
            } else {
                "libfacts_yolo.so"
            });
        let mut command = if mode == "jit" {
            let mut command = Command::new("dotnet");
            command.arg(&executable);
            command
        } else {
            Command::new(&executable)
        };
        command
            .current_dir(&dotnet_root)
            .arg(library)
            .arg(&layout_path)
            .arg(kind)
            .arg(mode);
        command.env_remove("LD_PRELOAD");
        if let Err(error) = checked(&mut command) {
            failures.push(error);
        }
    }
    if failures.is_empty() {
        println!(".NET {mode} {rid}: normal and fixture executables passed");
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}
