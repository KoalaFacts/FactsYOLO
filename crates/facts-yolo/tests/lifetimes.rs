// SPDX-License-Identifier: Apache-2.0
use facts_yolo::ffi::{handles::*, types::*};
use std::ptr::{null, null_mut};

fn model_options() -> FyModelOptions {
    FyModelOptions {
        struct_size: size_of::<FyModelOptions>() as u32,
        struct_version: 1,
        required_flags: 0,
        max_artifact_bytes: 64,
        max_parser_steps: 16,
    }
}

fn diagnostic(buffer: &mut [u8]) -> FyDiagnostic {
    FyDiagnostic {
        struct_size: size_of::<FyDiagnostic>() as u32,
        struct_version: 1,
        required_flags: 0,
        data: buffer.as_mut_ptr().cast(),
        capacity: buffer.len(),
        required_bytes: usize::MAX,
    }
}

#[test]
fn model_open_rejects_artifacts_even_in_fixture_builds() {
    let options = model_options();
    let mut out = null_mut();
    for bytes in [&b"FY_TEST_BASIC"[..], &b"arbitrary model bytes"[..]] {
        assert_eq!(
            unsafe {
                fy_model_open(
                    1,
                    bytes.as_ptr(),
                    bytes.len(),
                    &options,
                    &mut out,
                    null_mut(),
                )
            },
            FY_UNSUPPORTED_MODEL
        );
        assert!(out.is_null());
    }
}

#[test]
fn loader_validates_major_positive_limits_and_declared_bytes() {
    let bytes = [1_u8; 4];
    let mut options = model_options();
    let mut out = null_mut();
    assert_eq!(
        unsafe { fy_model_open(2, bytes.as_ptr(), 4, &options, &mut out, null_mut()) },
        FY_UNSUPPORTED_ABI
    );
    options.max_parser_steps = 0;
    assert_eq!(
        unsafe { fy_model_open(1, bytes.as_ptr(), 4, &options, &mut out, null_mut()) },
        FY_INVALID_ARGUMENT
    );
    options.max_parser_steps = 16;
    options.max_artifact_bytes = 0;
    assert_eq!(
        unsafe { fy_model_open(1, bytes.as_ptr(), 4, &options, &mut out, null_mut()) },
        FY_INVALID_ARGUMENT
    );
    options.max_artifact_bytes = 3;
    assert_eq!(
        unsafe { fy_model_open(1, bytes.as_ptr(), 4, &options, &mut out, null_mut()) },
        FY_LIMIT_EXCEEDED
    );
    options.max_artifact_bytes = 64;
    assert_eq!(
        unsafe { fy_model_open(1, null(), 4, &options, &mut out, null_mut()) },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { fy_model_open(1, null(), 0, &options, &mut out, null_mut()) },
        FY_INVALID_ARGUMENT
    );
    assert!(out.is_null());
}

#[test]
fn nonempty_model_output_is_preserved_without_dereferencing_it() {
    let mut marker = 7_u64;
    let sentinel = (&mut marker as *mut u64).cast::<FyModel>();
    let mut out = sentinel;
    let options = model_options();
    let bytes = [1_u8];
    assert_eq!(
        unsafe { fy_model_open(1, bytes.as_ptr(), 1, &options, &mut out, null_mut()) },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(out, sentinel);
    assert_eq!(marker, 7);
    // This is a backed pointer sentinel, not an owned model. Never release it.
}

#[test]
fn null_value_release_is_noop_but_null_slot_is_invalid() {
    let mut model = null_mut();
    let mut session = null_mut();
    assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
    assert_eq!(
        unsafe { fy_session_release(&mut session, null_mut()) },
        FY_OK
    );
    assert_eq!(
        unsafe { fy_model_release(null_mut(), null_mut()) },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { fy_session_release(null_mut(), null_mut()) },
        FY_INVALID_ARGUMENT
    );
}

#[test]
fn short_options_are_rejected_before_full_record_access() {
    #[repr(C, align(8))]
    struct Prefix([u32; 2]);
    let prefix = Prefix([8, 1]);
    let mut out = null_mut();
    let bytes = [0_u8];
    assert_eq!(
        unsafe {
            fy_model_open(
                1,
                bytes.as_ptr(),
                1,
                prefix.0.as_ptr().cast(),
                &mut out,
                null_mut(),
            )
        },
        FY_INVALID_ARGUMENT
    );
    assert!(out.is_null());
}

#[test]
fn capabilities_report_only_the_compiled_build() {
    let mut required = usize::MAX;
    assert_eq!(
        unsafe { fy_capabilities_copy(null_mut(), 0, &mut required, null_mut()) },
        FY_OK
    );
    let mut expected = Vec::new();
    if cfg!(feature = "td1") {
        expected.push(FyCapability {
            backend_id: 1,
            device_kind: FY_DEVICE_CPU,
            device_index: 0,
            flags: FY_CAP_COMPILED | FY_CAP_AVAILABLE,
            precision_mask: 1,
        });
    }
    if cfg!(feature = "contract-fixtures") {
        expected.push(FyCapability {
            backend_id: 0xffff0001,
            device_kind: FY_DEVICE_CPU,
            device_index: 0,
            flags: FY_CAP_COMPILED | FY_CAP_AVAILABLE | FY_CAP_TEST_ONLY,
            precision_mask: 1,
        });
    }
    assert_eq!(required, expected.len());
    let sentinel = FyCapability {
        backend_id: 77,
        device_kind: 77,
        device_index: 77,
        flags: 77,
        precision_mask: 77,
    };
    let mut records = [sentinel; 2];
    assert_eq!(
        unsafe {
            fy_capabilities_copy(
                records.as_mut_ptr(),
                records.len(),
                &mut required,
                null_mut(),
            )
        },
        FY_OK
    );
    assert_eq!(&records[..expected.len()], expected.as_slice());
    assert!(records[expected.len()..].iter().all(|r| *r == sentinel));
}

#[test]
fn capability_failures_preserve_destinations_and_count_except_shortage() {
    let mut required = 77;
    assert_eq!(
        unsafe { fy_capabilities_copy(null_mut(), 1, &mut required, null_mut()) },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(required, 77);
    let mut record = FyCapability {
        backend_id: 77,
        device_kind: 77,
        device_index: 77,
        flags: 77,
        precision_mask: 77,
    };
    assert_eq!(
        unsafe { fy_capabilities_copy(&mut record, usize::MAX, &mut required, null_mut()) },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(record.backend_id, 77);
    assert_eq!(required, 77);
    assert_eq!(
        unsafe { fy_capabilities_copy(&mut record, 1, null_mut(), null_mut()) },
        FY_INVALID_ARGUMENT
    );
    let expected = if cfg!(any(feature = "contract-fixtures", feature = "td1")) {
        FY_BUFFER_TOO_SMALL
    } else {
        FY_OK
    };
    assert_eq!(
        unsafe { fy_capabilities_copy(&mut record, 0, &mut required, null_mut()) },
        expected
    );
    assert_eq!(
        required,
        usize::from(cfg!(feature = "contract-fixtures")) + usize::from(cfg!(feature = "td1"))
    );
    assert_eq!(record.backend_id, 77);
}

#[test]
fn malformed_diagnostic_has_precedence_and_leaves_outputs_unchanged() {
    let mut text = [0xA5_u8; 16];
    let mut diag = diagnostic(&mut text);
    diag.struct_version = 99;
    let mut count = 77;
    assert_eq!(
        unsafe { fy_capabilities_copy(null_mut(), 0, &mut count, &mut diag) },
        FY_UNSUPPORTED_ABI
    );
    assert_eq!(count, 77);
    assert_eq!(diag.required_bytes, usize::MAX);
    assert_eq!(text, [0xA5; 16]);
}

#[cfg(feature = "contract-fixtures")]
mod fixture_lifetimes {
    use super::*;
    use facts_yolo::fixtures::{BASIC, FY_BACKEND_TEST_FIXTURE, live_counts};
    use std::mem::MaybeUninit;
    use std::ptr::addr_of_mut;
    use std::sync::Mutex;

    // All allocation-count assertions in this test binary share one lock.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn options() -> FySessionOptions {
        FySessionOptions {
            struct_size: size_of::<FySessionOptions>() as u32,
            struct_version: 1,
            required_flags: 0,
            backend_id: FY_BACKEND_TEST_FIXTURE,
            device_kind: FY_DEVICE_CPU,
            device_index: 0,
            precision: FY_PRECISION_F32,
            fallback_policy: FY_FALLBACK_NONE,
            cpu_threads: 1,
            max_image_bytes: 1024,
            max_pixels: 64,
            max_detections: 16,
        }
    }

    fn info() -> FySessionInfo {
        FySessionInfo {
            struct_size: size_of::<FySessionInfo>() as u32,
            struct_version: 1,
            required_flags: 0,
            backend_id: 77,
            device_kind: 77,
            device_index: 77,
            precision: 77,
            execution_flags: 77,
        }
    }

    fn model() -> *mut FyModel {
        let mut out = null_mut();
        assert_eq!(
            unsafe { fy_test_model_create(1, BASIC, &mut out, null_mut()) },
            FY_OK
        );
        assert!(!out.is_null());
        out
    }

    fn session(model: *const FyModel) -> *mut FySession {
        let mut out = null_mut();
        assert_eq!(
            unsafe { fy_session_create(1, model, &options(), &mut out, null_mut()) },
            FY_OK
        );
        assert!(!out.is_null());
        out
    }

    #[test]
    fn release_model_with_live_session_keeps_model_state_until_last_owner() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let base = live_counts();
        let mut model = model();
        let mut session = session(model);
        assert_eq!(live_counts(), [base[0] + 1, base[1] + 1, base[2] + 1]);
        assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
        assert!(model.is_null());
        assert_eq!(live_counts(), [base[0], base[1] + 1, base[2] + 1]);
        let mut out = info();
        assert_eq!(
            unsafe { fy_session_describe(session, &mut out, null_mut()) },
            FY_OK
        );
        assert_eq!(out.backend_id, FY_BACKEND_TEST_FIXTURE);
        assert_eq!(
            (
                out.device_kind,
                out.device_index,
                out.precision,
                out.execution_flags
            ),
            (FY_DEVICE_CPU, 0, FY_PRECISION_F32, 0)
        );
        assert_eq!(
            unsafe { fy_session_release(&mut session, null_mut()) },
            FY_OK
        );
        assert!(session.is_null());
        assert_eq!(live_counts(), base);
        assert_eq!(
            unsafe { fy_session_release(&mut session, null_mut()) },
            FY_OK
        );
        assert_eq!(live_counts(), base);
    }

    #[test]
    fn multiple_sessions_retain_one_shared_model_state() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let base = live_counts();
        let mut model = model();
        let mut sessions = [session(model), session(model), session(model)];
        assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
        assert_eq!(live_counts(), [base[0], base[1] + 1, base[2] + 3]);
        for session in &mut sessions {
            assert_eq!(unsafe { fy_session_release(session, null_mut()) }, FY_OK);
        }
        assert_eq!(live_counts(), base);
    }

    #[test]
    fn failed_create_has_no_live_object_delta() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut model = model();
        let before = live_counts();
        let mut cases = Vec::new();
        for field in 0..4 {
            let mut bad = options();
            match field {
                0 => bad.cpu_threads = 0,
                1 => bad.max_image_bytes = 0,
                2 => bad.max_pixels = 0,
                _ => bad.max_detections = 0,
            }
            cases.push((bad, FY_INVALID_ARGUMENT));
        }
        let mut bad = options();
        bad.cpu_threads = 2;
        cases.push((bad, FY_UNSUPPORTED_FEATURE));
        bad = options();
        bad.precision = 9;
        cases.push((bad, FY_UNSUPPORTED_FEATURE));
        bad = options();
        bad.fallback_policy = 1;
        cases.push((bad, FY_UNSUPPORTED_FEATURE));
        bad = options();
        bad.backend_id = 0;
        cases.push((bad, FY_UNSUPPORTED_FEATURE));
        bad = options();
        bad.device_kind = FY_DEVICE_GPU;
        cases.push((bad, FY_DEVICE_UNAVAILABLE));
        bad = options();
        bad.device_index = 1;
        cases.push((bad, FY_DEVICE_UNAVAILABLE));
        bad = options();
        bad.device_kind = 999;
        cases.push((bad, FY_INVALID_ARGUMENT));
        for _ in 0..100 {
            for (options, status) in &cases {
                let mut out = null_mut();
                assert_eq!(
                    unsafe { fy_session_create(1, model, options, &mut out, null_mut()) },
                    *status
                );
                assert!(out.is_null());
                assert_eq!(live_counts(), before);
            }
        }
        assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
    }

    #[test]
    fn nonempty_session_output_is_preserved() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut model = model();
        let mut out = session(model);
        let existing = out;
        let before = live_counts();
        assert_eq!(
            unsafe { fy_session_create(1, model, &options(), &mut out, null_mut()) },
            FY_INVALID_ARGUMENT
        );
        assert_eq!(out, existing);
        assert_eq!(live_counts(), before);
        assert_eq!(unsafe { fy_session_release(&mut out, null_mut()) }, FY_OK);
        assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
    }

    #[test]
    fn fixture_factory_validates_abi_id_and_output() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let base = live_counts();
        let mut out = null_mut();
        assert_eq!(
            unsafe { fy_test_model_create(2, BASIC, &mut out, null_mut()) },
            FY_UNSUPPORTED_ABI
        );
        assert_eq!(
            unsafe { fy_test_model_create(1, 0, &mut out, null_mut()) },
            FY_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { fy_test_model_create(1, 7, &mut out, null_mut()) },
            FY_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { fy_test_model_create(1, BASIC, null_mut(), null_mut()) },
            FY_INVALID_ARGUMENT
        );
        assert!(out.is_null());
        assert_eq!(live_counts(), base);
    }

    #[test]
    fn creation_checks_abi_null_model_and_short_options() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut model = model();
        let mut out = null_mut();
        let base = live_counts();
        assert_eq!(
            unsafe { fy_session_create(2, model, &options(), &mut out, null_mut()) },
            FY_UNSUPPORTED_ABI
        );
        assert_eq!(
            unsafe { fy_session_create(1, null(), &options(), &mut out, null_mut()) },
            FY_INVALID_ARGUMENT
        );
        #[repr(C, align(8))]
        struct Prefix([u32; 2]);
        let prefix = Prefix([8, 1]);
        assert_eq!(
            unsafe { fy_session_create(1, model, prefix.0.as_ptr().cast(), &mut out, null_mut()) },
            FY_INVALID_ARGUMENT
        );
        assert!(out.is_null());
        assert_eq!(live_counts(), base);
        assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
    }

    #[test]
    fn session_copies_settings_and_only_writes_output_payload() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut model = model();
        let mut settings = options();
        let mut session = null_mut();
        assert_eq!(
            unsafe { fy_session_create(1, model, &settings, &mut session, null_mut()) },
            FY_OK
        );
        settings.device_index = 99;
        assert_eq!(settings.device_index, 99);
        let mut out = MaybeUninit::<FySessionInfo>::uninit();
        let p = out.as_mut_ptr();
        unsafe {
            addr_of_mut!((*p).struct_size).write(size_of::<FySessionInfo>() as u32);
            addr_of_mut!((*p).struct_version).write(1);
            addr_of_mut!((*p).required_flags).write(0);
            assert_eq!(fy_session_describe(session, p, null_mut()), FY_OK);
            let out = out.assume_init();
            assert_eq!(out.device_index, 0);
            assert_eq!(out.struct_size, size_of::<FySessionInfo>() as u32);
        }
        #[repr(C)]
        struct Extended {
            base: FySessionInfo,
            tail: [u8; 16],
        }
        let mut extended = Extended {
            base: info(),
            tail: [0xA5; 16],
        };
        extended.base.struct_size = size_of::<Extended>() as u32;
        assert_eq!(
            unsafe { fy_session_describe(session, &mut extended.base, null_mut()) },
            FY_OK
        );
        assert_eq!(extended.base.struct_size, size_of::<Extended>() as u32);
        assert_eq!(extended.tail, [0xA5; 16]);
        extended.base.struct_version = 2;
        extended.base.backend_id = 77;
        assert_eq!(
            unsafe { fy_session_describe(session, &mut extended.base, null_mut()) },
            FY_UNSUPPORTED_ABI
        );
        assert_eq!(extended.base.backend_id, 77);
        assert_eq!(
            unsafe { fy_session_release(&mut session, null_mut()) },
            FY_OK
        );
        assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
    }

    #[test]
    fn malformed_diagnostic_cannot_release_live_resources() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut model = model();
        let existing = model;
        let before = live_counts();
        let mut text = [0xA5; 8];
        let mut diag = diagnostic(&mut text);
        diag.required_flags = 1;
        assert_eq!(
            unsafe { fy_model_release(&mut model, &mut diag) },
            FY_UNSUPPORTED_FEATURE
        );
        assert_eq!(model, existing);
        assert_eq!(live_counts(), before);
        diag.required_flags = 0;
        assert_eq!(unsafe { fy_model_release(&mut model, &mut diag) }, FY_OK);
        assert!(model.is_null());
        assert_eq!(diag.required_bytes, 0);
        assert_eq!(text[0], 0);
    }

    #[test]
    fn immutable_session_queries_can_overlap_while_owner_is_live() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut model = model();
        let mut session = session(model);
        struct SharedSession(*const FySession);
        // SAFETY: this private test wrapper exposes only immutable description.
        // The scoped threads join before either owning handle is released. Do
        // not make a reference to the zero-sized opaque ABI marker itself.
        unsafe impl Sync for SharedSession {}
        impl SharedSession {
            fn describe(&self, out: &mut FySessionInfo) -> i32 {
                unsafe { fy_session_describe(self.0, out, null_mut()) }
            }
        }
        let borrowed = SharedSession(session);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..100 {
                        let mut out = info();
                        assert_eq!(borrowed.describe(&mut out), FY_OK);
                        assert_eq!(out.backend_id, FY_BACKEND_TEST_FIXTURE);
                    }
                });
            }
        });
        assert_eq!(
            unsafe { fy_session_release(&mut session, null_mut()) },
            FY_OK
        );
        assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
    }

    #[test]
    fn ten_thousand_create_release_cycles_return_counts_to_baseline() {
        let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let base = live_counts();
        for _ in 0..10_000 {
            let mut model = model();
            let mut session = session(model);
            assert_eq!(unsafe { fy_model_release(&mut model, null_mut()) }, FY_OK);
            assert_eq!(
                unsafe { fy_session_release(&mut session, null_mut()) },
                FY_OK
            );
            assert_eq!(live_counts(), base);
        }
    }
}
