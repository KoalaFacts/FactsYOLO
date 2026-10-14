// SPDX-License-Identifier: Apache-2.0
use super::evidence;
use std::process::Command;

pub(super) fn preflight() -> Result<(), String> {
    if !cfg!(any(target_os = "linux", windows)) || std::env::consts::ARCH != "x86_64" {
        return Err("mandatory local target is Linux/Windows x64".to_owned());
    }
    let tools = if cfg!(windows) {
        vec![
            ("rustc", "--version"),
            ("cargo", "--version"),
            ("git", "--version"),
            ("dotnet", "--version"),
            ("cl", "/?"),
            ("dumpbin", "/?"),
        ]
    } else {
        vec![
            ("rustc", "--version"),
            ("cargo", "--version"),
            ("git", "--version"),
            ("dotnet", "--version"),
            ("cc", "--version"),
            ("nm", "--version"),
        ]
    };
    for (tool, arg) in tools {
        let output = Command::new(tool)
            .current_dir(evidence::root().join("tests/dotnet"))
            .arg(arg)
            .output()
            .map_err(|error| format!("mandatory tool {tool} unavailable: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "mandatory tool {tool} exited {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        if tool == "dotnet" && String::from_utf8_lossy(&output.stdout).trim() != "10.0.401" {
            return Err("mandatory tool dotnet must use pinned SDK 10.0.401".to_owned());
        }
    }
    println!("mandatory local tools available (preflight only; no verification pass claimed)");
    Ok(())
}

pub(super) fn verify(worker: bool) -> Result<(), String> {
    preflight()?;
    let directory = evidence::start("verify")?;
    evidence::manifest(&directory)?;
    // A separate runner path permits Cargo to rebuild xtask during workspace
    // integration tests on Windows, where the running executable is locked.
    if !worker {
        let runner = directory.join(if cfg!(windows) {
            "runner.exe"
        } else {
            "runner"
        });
        std::fs::copy(
            std::env::current_exe().map_err(|error| error.to_string())?,
            &runner,
        )
        .map_err(|error| error.to_string())?;
        let text = evidence::run(
            &directory,
            "verification",
            Command::new(runner).arg("verify-worker"),
        )?;
        print!("{text}");
        return Ok(());
    }
    evidence::run(
        &directory,
        "dotnet-info",
        Command::new("dotnet")
            .current_dir(evidence::root().join("tests/dotnet"))
            .arg("--info"),
    )?;
    let compiler = if cfg!(windows) { "cl" } else { "cc" };
    evidence::run(
        &directory,
        "c-compiler",
        Command::new(compiler).arg(if cfg!(windows) { "/?" } else { "--version" }),
    )?;
    let result = (|| {
        for fixture in [false, true] {
            let mode = if fixture { "fixture" } else { "normal" };
            for release in [false, true] {
                let profile = if release { "release" } else { "debug" };
                let mut command = evidence::cargo();
                command.args([
                    "test",
                    "--workspace",
                    "--no-fail-fast",
                    "--locked",
                    "--offline",
                    "--no-default-features",
                ]);
                if fixture {
                    command.args([
                        "--features",
                        "contract-fixtures",
                        "--target-dir",
                        "target/contract-fixtures",
                    ]);
                }
                if release {
                    command.arg("--release");
                }
                evidence::run(&directory, &format!("tests-{mode}-{profile}"), &mut command)?;
            }
            let mut command = evidence::cargo();
            command.args([
                "clippy",
                "--workspace",
                "--all-targets",
                "--locked",
                "--offline",
                "--no-default-features",
            ]);
            if fixture {
                command.args([
                    "--features",
                    "contract-fixtures",
                    "--target-dir",
                    "target/contract-fixtures",
                ]);
            }
            command.args(["--", "-D", "warnings"]);
            evidence::run(&directory, &format!("clippy-{mode}"), &mut command)?;
        }
        evidence::run(
            &directory,
            "fmt",
            evidence::cargo().args(["fmt", "--all", "--", "--check"]),
        )?;
        let executable = std::env::current_exe().map_err(|error| error.to_string())?;
        for (name, args) in [
            ("c", vec!["c"]),
            ("resources", vec!["resources"]),
            ("jit", vec!["dotnet", "--mode", "jit"]),
            ("aot", vec!["dotnet", "--mode", "aot"]),
        ] {
            evidence::run(&directory, name, Command::new(&executable).args(args))?;
        }
        Ok::<(), String>(())
    })();
    std::fs::write(
        directory.join("outcome.txt"),
        if result.is_ok() { "PASS\n" } else { "FAIL\n" },
    )
    .map_err(|error| error.to_string())?;
    result?;
    println!(
        "local-target verification passed; dedicated Miri/i686 gates remain separately required; evidence: {}",
        directory.display()
    );
    Ok(())
}
