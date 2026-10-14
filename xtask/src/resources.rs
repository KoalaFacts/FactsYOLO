// SPDX-License-Identifier: Apache-2.0
use super::evidence;
use std::process::Command;

pub(super) fn run() -> Result<(), String> {
    let directory = evidence::start("resources")?;
    evidence::manifest(&directory)?;
    let target = evidence::root().join("target/resource-probe");
    evidence::run(
        &directory,
        "build",
        evidence::cargo()
            .args([
                "build",
                "-p",
                "facts-yolo",
                "--example",
                "resource_probe",
                "--release",
                "--no-default-features",
                "--features",
                "contract-fixtures",
                "--locked",
                "--offline",
                "--target-dir",
            ])
            .arg(&target),
    )?;
    let executable = target.join("release/examples").join(if cfg!(windows) {
        "resource_probe.exe"
    } else {
        "resource_probe"
    });
    let result = evidence::run(
        &directory,
        "samples",
        Command::new(executable).current_dir(evidence::root()),
    );
    std::fs::write(
        directory.join("outcome.txt"),
        if result.is_ok() { "PASS\n" } else { "FAIL\n" },
    )
    .map_err(|error| error.to_string())?;
    print!("{}", result?);
    println!("resource evidence: {}", directory.display());
    Ok(())
}
