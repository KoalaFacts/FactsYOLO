// SPDX-License-Identifier: Apache-2.0
use std::process::Command;

#[test]
fn normal_export_gate_rejects_the_real_fixture_library() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("exports")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("normal-export negative control rejected fixture library")
    );
}

#[test]
fn resource_command_executes_full_workload_and_records_samples() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("resources")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.contains("resources: 1000 warmup, 10000 measured, 100 samples"),
        "{text}"
    );
    assert!(
        text.contains("live objects and Rust allocation bytes returned to baseline"),
        "{text}"
    );
}

#[test]
fn verification_rejects_missing_mandatory_tools_before_success() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["verify", "--preflight"])
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(text.contains("mandatory tool"), "{text}");
    assert!(!text.contains("verification passed"));
}
