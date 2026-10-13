// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "contract-fixtures")]
mod control_support;
use control_support::{Owner, SERIAL};
use facts_yolo::{ffi::testing::*, ffi::types::*, fixtures::*};
use std::ptr::null;

#[test]
fn unwinding_execution_fault_poisoning_is_sticky_and_releasable() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let baseline = resource_counts();
    let owner = Owner::new(PANIC);
    let first = owner.call(0.5);
    let second = owner.call(0.5);
    drop(owner);
    assert_eq!(first.0, FY_INTERNAL_ERROR);
    assert!(!first.2.contains("private-fixture"));
    assert_eq!(second.0, FY_POISONED);
    assert_eq!(resource_counts(), baseline);
}

#[test]
fn invalid_backend_output_is_not_silently_reused() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let owner = Owner::new(INVALID_OUTPUT);
    assert_eq!(owner.call(0.5).0, FY_MODEL_OUTPUT_INVALID);
    assert_eq!(owner.call(0.5).0, FY_POISONED);
}

#[test]
fn test_controls_reject_invalid_arguments_and_do_not_change_ready_state() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let basic = Owner::new(BASIC);
    let blocking = Owner::new(BLOCKING);
    unsafe {
        assert_eq!(fy_test_session_wait_running(null(), 1), FY_INVALID_ARGUMENT);
        assert_eq!(fy_test_session_resume(null()), FY_INVALID_ARGUMENT);
        assert_eq!(
            fy_test_session_wait_running(blocking.session, 0),
            FY_INVALID_ARGUMENT
        );
        assert_eq!(
            fy_test_session_wait_running(blocking.session, 1),
            FY_INTERNAL_ERROR
        );
    }
    assert_eq!(basic.resume(), FY_UNSUPPORTED_FEATURE);
    assert_eq!(basic.call(0.5).0, FY_OK);
}

#[test]
fn payload_drop_panic_is_contained_in_a_subprocess() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    const PROBE: &str = "FACTSYOLO_TEST_PAYLOAD_PROBE";
    if std::env::var_os(PROBE).is_some() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static HOOK_CALLS: AtomicUsize = AtomicUsize::new(0);
        // This test owns an isolated child process; the library must not replace its hook.
        std::panic::set_hook(Box::new(|_| {
            HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
        }));
        let baseline = resource_counts();
        let owner = Owner::new(PAYLOAD_DROP_PANIC);
        let first = owner.call(0.5);
        assert_eq!(first.0, FY_INTERNAL_ERROR);
        assert_eq!(HOOK_CALLS.load(Ordering::SeqCst), 2);
        assert!(!first.2.contains("private-fixture"));
        assert_eq!(owner.call(0.5).0, FY_POISONED);
        drop(owner);
        assert_eq!(resource_counts(), baseline);
        println!("PAYLOAD_PROBE_SURVIVED");
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "payload_drop_panic_is_contained_in_a_subprocess",
            "--nocapture",
        ])
        .env(PROBE, "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "child failed: {:?}", output);
    assert!(String::from_utf8_lossy(&output.stdout).contains("PAYLOAD_PROBE_SURVIVED"));
}
