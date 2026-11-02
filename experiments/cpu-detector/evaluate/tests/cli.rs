// SPDX-License-Identifier: Apache-2.0
use std::process::Command;
#[test]
fn invalid_cli_reports_usage_and_does_not_claim_training() {
    let output = Command::new(env!("CARGO_BIN_EXE_fy-cpu-evaluate"))
        .args(["train", "invalid"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage:"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("PASS"));
}

#[test]
fn independent_process_uses_artifact_weights_and_rejects_bad_files() {
    use fy_tiny_model::{Model, PARAMETERS, W2};
    let directory = std::env::temp_dir().join(format!("fy-cpu-cli-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let artifact = directory.join("model.fytd");
    let rgb = directory.join("image.rgb");
    let mut model = Model {
        weights: vec![0.0; PARAMETERS],
    };
    model.weights[W2 + 1] = 10.0;
    std::fs::write(&artifact, model.to_bytes().unwrap()).unwrap();
    std::fs::write(&rgb, vec![0; 64 * 64 * 3]).unwrap();
    let infer = || {
        Command::new(env!("CARGO_BIN_EXE_fy-cpu-evaluate"))
            .arg("infer")
            .arg(&artifact)
            .arg(&rgb)
            .output()
            .unwrap()
    };
    let first = infer();
    assert!(first.status.success());
    let detection: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(detection["class"], 0);
    assert_eq!(
        detection["corners"],
        serde_json::json!([16.0, 16.0, 48.0, 48.0])
    );
    // Different serialized parameters must control a new process, without training state.
    model.weights[W2 + 1] = 0.0;
    model.weights[W2] = 10.0;
    std::fs::write(&artifact, model.to_bytes().unwrap()).unwrap();
    let background = infer();
    assert!(background.status.success());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&background.stdout).unwrap(),
        serde_json::Value::Null
    );
    for length in [0, 64 * 64 * 3 - 1, 64 * 64 * 3 + 1, 100000] {
        std::fs::write(&rgb, vec![0; length]).unwrap();
        assert!(!infer().status.success());
    }
    std::fs::write(&rgb, vec![0; 64 * 64 * 3]).unwrap();
    for bytes in [
        vec![],
        vec![0; 65537],
        model.to_bytes().unwrap()[..100].to_vec(),
    ] {
        std::fs::write(&artifact, bytes).unwrap();
        assert!(!infer().status.success());
    }
    let occupied = Command::new(env!("CARGO_BIN_EXE_fy-cpu-evaluate"))
        .args(["train", "burn"])
        .arg(&directory)
        .output()
        .unwrap();
    assert!(!occupied.status.success());
    assert!(String::from_utf8_lossy(&occupied.stderr).contains("create new output directory"));
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn background_artifact_export_is_opt_in_and_texture_remains_unsupported() {
    let command = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_fy-edge-training"))
            .args(args)
            .output()
            .unwrap()
    };
    // Invalid backend avoids spending 2,000 training steps, while proving
    // the --background export command reaches actual backend validation.
    let allowed = command(&["not-a-backend", "--background", "--evidence-artifact", "new.fytd"]);
    assert!(!allowed.status.success());
    assert!(String::from_utf8_lossy(&allowed.stderr).contains("backend must be burn or candle"));

    for invalid in [
        vec!["not-a-backend", "--texture", "--evidence-artifact", "new.fytd"],
        vec!["not-a-backend", "--background", "--evidence-artifact"],
        vec!["not-a-backend", "--background", "--evidence-artifact", ""],
    ] {
        let outcome = command(&invalid);
        assert!(!outcome.status.success());
        assert!(String::from_utf8_lossy(&outcome.stderr).contains("usage: fy-edge-training"));
    }
}
