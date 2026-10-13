// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "contract-fixtures")]
mod control_support;
use control_support::{Owner, SERIAL};
use facts_yolo::{ffi::types::*, fixtures::*};

#[test]
fn overlapping_detection_is_busy_without_execution_or_result_allocation() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let base = resource_counts();
    let owner = Owner::new(BLOCKING);
    let executions = execution_count();
    std::thread::scope(|scope| {
        let first = scope.spawn(|| owner.call(0.5));
        let waiting = owner.wait();
        let before = (execution_count(), resource_counts());
        let second = if waiting == FY_OK {
            owner.call(0.5)
        } else {
            (-1, 0, String::new())
        };
        let after = (execution_count(), resource_counts());
        // Always release the test latch before assertions or joining.
        let resumed = owner.resume();
        let completed = first.join().unwrap();
        assert_eq!(waiting, FY_OK);
        assert_eq!(resumed, FY_OK);
        assert_eq!(second.0, FY_BUSY);
        assert_eq!(second.1, 0);
        assert!(!second.2.is_empty());
        assert_eq!(before, after);
        assert_eq!((completed.0, completed.1), (FY_OK, 3));
        assert!(completed.2.is_empty());
    });
    assert_eq!(execution_count(), executions + 1);
    drop(owner);
    assert_eq!(resource_counts(), base);
}

#[test]
fn distinct_sessions_can_be_running_at_the_same_time() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let a = Owner::new(BLOCKING);
    let b = Owner::new(BLOCKING);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| a.call(0.5));
        let second = scope.spawn(|| b.call(0.5));
        let a_running = a.wait();
        let b_running = b.wait();
        let a_resume = a.resume();
        let b_resume = b.resume();
        let a_result = first.join().unwrap();
        let b_result = second.join().unwrap();
        assert_eq!((a_running, b_running), (FY_OK, FY_OK));
        assert_eq!((a_resume, b_resume), (FY_OK, FY_OK));
        assert_eq!((a_result.0, b_result.0), (FY_OK, FY_OK));
    });
}

#[test]
fn validation_errors_are_per_call_and_do_not_poison_a_ready_session() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let owner = Owner::new(BASIC);
    let before = execution_count();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8).map(|_| scope.spawn(|| owner.call(-1.0))).collect();
        for worker in workers {
            let result = worker.join().unwrap();
            assert_eq!(result.0, FY_INVALID_ARGUMENT);
            assert!(!result.2.is_empty());
        }
    });
    assert_eq!(execution_count(), before);
    assert_eq!(owner.call(0.5).0, FY_OK);
}
