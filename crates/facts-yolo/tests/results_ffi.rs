// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "contract-fixtures")]
use facts_yolo::{
    ffi::{handles::*, results::*, types::*},
    fixtures::{BASIC, EMPTY, FY_BACKEND_TEST_FIXTURE, INVALID_OUTPUT},
};
use std::mem::MaybeUninit;
use std::ptr::{addr_of_mut, null, null_mut};

struct Context {
    model: *mut FyModel,
    session: *mut FySession,
}

impl Context {
    fn new(fixture: u32) -> Self {
        let mut model = null_mut();
        let mut session = null_mut();
        let options = FySessionOptions {
            struct_size: size_of::<FySessionOptions>() as u32,
            struct_version: 1,
            required_flags: 0,
            backend_id: FY_BACKEND_TEST_FIXTURE,
            device_kind: FY_DEVICE_CPU,
            device_index: 0,
            precision: FY_PRECISION_F32,
            fallback_policy: FY_FALLBACK_NONE,
            cpu_threads: 1,
            max_image_bytes: 128,
            max_pixels: 16,
            max_detections: 16,
        };
        unsafe {
            assert_eq!(fy_test_model_create(1, fixture, &mut model, null_mut()), FY_OK);
            assert_eq!(fy_session_create(1, model, &options, &mut session, null_mut()), FY_OK);
        }
        Self { model, session }
    }

    fn run(&self, pixels: &[u8], cap: usize) -> *mut FyResult {
        let mut result = null_mut();
        assert_eq!(
            unsafe { fy_detect(self.session, &image(pixels), &options(cap), &mut result, null_mut()) },
            FY_OK
        );
        assert!(!result.is_null());
        result
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            assert_eq!(fy_session_release(&mut self.session, null_mut()), FY_OK);
            assert_eq!(fy_model_release(&mut self.model, null_mut()), FY_OK);
        }
    }
}

fn image(pixels: &[u8]) -> FyImage {
    FyImage {
        struct_size: size_of::<FyImage>() as u32,
        struct_version: 1,
        required_flags: 0,
        data: pixels.as_ptr(),
        byte_length: pixels.len(),
        width: 2,
        height: 2,
        row_stride: 6,
        pixel_format: FY_RGB8,
    }
}

fn options(cap: usize) -> FyDetectOptions {
    FyDetectOptions {
        struct_size: size_of::<FyDetectOptions>() as u32,
        struct_version: 1,
        required_flags: 0,
        score_threshold: 0.5,
        max_detections: cap,
    }
}

fn info() -> FyResultInfo {
    FyResultInfo {
        struct_size: size_of::<FyResultInfo>() as u32,
        struct_version: 1,
        required_flags: 0,
        source_width: 77,
        source_height: 77,
        count: 77,
        result_flags: 77,
        backend_id: 77,
        device_kind: 77,
        device_index: 77,
        precision: 77,
        execution_flags: 77,
        model_id: [77; 32],
    }
}

fn sentinel() -> FyDetection {
    FyDetection {
        x_min: 77.0,
        y_min: 77.0,
        x_max: 77.0,
        y_max: 77.0,
        score: 77.0,
        class_id: 77,
    }
}

#[test]
fn completed_result_outlives_pixels_model_and_session() {
    let mut result = {
        let context = Context::new(BASIC);
        let pixels = vec![0_u8; 12];
        context.run(&pixels, 16)
    };
    let mut metadata = info();
    assert_eq!(unsafe { fy_result_describe(result, &mut metadata, null_mut()) }, FY_OK);
    assert_eq!((metadata.source_width, metadata.source_height, metadata.count), (2, 2, 3));
    assert_eq!(metadata.backend_id, FY_BACKEND_TEST_FIXTURE);
    assert_eq!(metadata.device_kind, FY_DEVICE_CPU);
    assert_eq!(metadata.precision, FY_PRECISION_F32);
    assert_eq!(metadata.device_index, 0);
    assert_eq!(metadata.execution_flags, 0);
    assert_eq!(metadata.result_flags, 0);
    assert_eq!(metadata.model_id[0], BASIC as u8);
    assert_eq!(&metadata.model_id[1..], &[0; 31]);
    let mut copied = [sentinel(); 4];
    let mut required = 77;
    for _ in 0..100 {
        assert_eq!(unsafe { fy_result_copy(result, null_mut(), 0, &mut required, null_mut()) }, FY_OK);
        assert_eq!(required, 3);
        assert_eq!(unsafe { fy_result_copy(result, copied.as_mut_ptr(), 3, &mut required, null_mut()) }, FY_OK);
        assert_eq!(copied[0].class_id, 0);
        assert_eq!(copied[0].x_max, 2.0);
        assert_eq!(copied[1].class_id, 1);
        assert_eq!(copied[2].score, 0.5);
        assert_eq!(copied[3], sentinel());
    }
    assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
    assert!(result.is_null());
    assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
}

#[test]
fn insufficient_capacity_is_atomic_and_invalid_copy_preserves_count() {
    let context = Context::new(BASIC);
    let mut result = context.run(&[0; 12], 16);
    let mut copied = [sentinel(); 4];
    let mut required = 77;
    for cap in [0, 1, 2] {
        assert_eq!(unsafe { fy_result_copy(result, copied.as_mut_ptr(), cap, &mut required, null_mut()) }, FY_BUFFER_TOO_SMALL);
        assert_eq!(required, 3);
        assert_eq!(copied, [sentinel(); 4]);
    }
    required = 77;
    assert_eq!(unsafe { fy_result_copy(result, null_mut(), 1, &mut required, null_mut()) }, FY_INVALID_ARGUMENT);
    assert_eq!(required, 77);
    assert_eq!(unsafe { fy_result_copy(result, copied.as_mut_ptr(), usize::MAX, &mut required, null_mut()) }, FY_INVALID_ARGUMENT);
    assert_eq!(required, 77);
    assert_eq!(copied, [sentinel(); 4]);
    assert_eq!(unsafe { fy_result_copy(result, copied.as_mut_ptr(), 4, null_mut(), null_mut()) }, FY_INVALID_ARGUMENT);
    assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
}

#[test]
fn empty_and_capped_results_report_exact_metadata() {
    for (fixture, cap, count, flags) in [(EMPTY, 16, 0, 0), (BASIC, 2, 2, FY_RESULT_TRUNCATED)] {
        let context = Context::new(fixture);
        let mut result = context.run(&[0; 12], cap);
        let mut metadata = info();
        assert_eq!(unsafe { fy_result_describe(result, &mut metadata, null_mut()) }, FY_OK);
        assert_eq!(metadata.count, count);
        assert_eq!(metadata.result_flags, flags);
        let mut copied = sentinel();
        let mut required = 77;
        if count == 0 {
            assert_eq!(unsafe { fy_result_copy(result, &mut copied, 0, &mut required, null_mut()) }, FY_OK);
            assert_eq!(required, 0);
            assert_eq!(copied, sentinel());
        }
        assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
    }
}

#[test]
fn invalid_model_output_cannot_be_hidden_by_threshold() {
    let context = Context::new(INVALID_OUTPUT);
    let mut result = null_mut();
    let mut opts = options(1);
    opts.score_threshold = 1.0;
    assert_eq!(unsafe { fy_detect(context.session, &image(&[0; 12]), &opts, &mut result, null_mut()) }, FY_MODEL_OUTPUT_INVALID);
    assert!(result.is_null());
}

#[test]
fn failed_detect_preserves_nonempty_output_and_validates_options() {
    let context = Context::new(BASIC);
    let mut result = context.run(&[0; 12], 16);
    let existing = result;
    assert_eq!(unsafe { fy_detect(context.session, &image(&[0; 12]), &options(16), &mut result, null_mut()) }, FY_INVALID_ARGUMENT);
    assert_eq!(result, existing);
    assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
    for (cap, expected) in [(0, FY_INVALID_ARGUMENT), (17, FY_LIMIT_EXCEEDED)] {
        assert_eq!(unsafe { fy_detect(context.session, &image(&[0; 12]), &options(cap), &mut result, null_mut()) }, expected);
        assert!(result.is_null());
    }
    for threshold in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        let mut opts = options(16);
        opts.score_threshold = threshold;
        assert_eq!(unsafe { fy_detect(context.session, &image(&[0; 12]), &opts, &mut result, null_mut()) }, FY_INVALID_ARGUMENT);
        assert!(result.is_null());
    }
    assert_eq!(unsafe { fy_detect(context.session, &image(&[0; 11]), &options(16), &mut result, null_mut()) }, FY_INVALID_ARGUMENT);
    assert!(result.is_null());
    result = context.run(&[0; 12], 16);
    assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
}

#[test]
fn short_inputs_are_rejected_without_full_record_reads() {
    #[repr(C, align(8))]
    struct Prefix([u32; 2]);
    let context = Context::new(BASIC);
    let prefix = Prefix([8, 1]);
    let mut result = null_mut();
    assert_eq!(unsafe { fy_detect(context.session, prefix.0.as_ptr().cast(), &options(16), &mut result, null_mut()) }, FY_INVALID_ARGUMENT);
    assert_eq!(unsafe { fy_detect(context.session, &image(&[0; 12]), prefix.0.as_ptr().cast(), &mut result, null_mut()) }, FY_INVALID_ARGUMENT);
    assert!(result.is_null());
    assert_eq!(unsafe { fy_detect(null_mut(), null(), null(), &mut result, null_mut()) }, FY_INVALID_ARGUMENT);
    assert_eq!(unsafe { fy_result_release(null_mut(), null_mut()) }, FY_INVALID_ARGUMENT);
}

#[test]
fn output_payload_need_not_be_initialized_and_extensions_are_preserved() {
    let context = Context::new(BASIC);
    let mut result = context.run(&[0; 12], 16);
    let mut out = MaybeUninit::<FyResultInfo>::uninit();
    unsafe {
        let raw = out.as_mut_ptr();
        addr_of_mut!((*raw).struct_size).write(size_of::<FyResultInfo>() as u32);
        addr_of_mut!((*raw).struct_version).write(1);
        addr_of_mut!((*raw).required_flags).write(0);
        assert_eq!(fy_result_describe(result, raw, null_mut()), FY_OK);
        assert_eq!(out.assume_init().count, 3);
    }
    #[repr(C)]
    struct Extended {
        base: FyResultInfo,
        tail: [u8; 16],
    }
    let mut extended = Extended { base: info(), tail: [0xA5; 16] };
    extended.base.struct_size = size_of::<Extended>() as u32;
    assert_eq!(unsafe { fy_result_describe(result, &mut extended.base, null_mut()) }, FY_OK);
    assert_eq!(extended.tail, [0xA5; 16]);
    assert_eq!(extended.base.struct_size, size_of::<Extended>() as u32);
    extended.base.struct_version = 2;
    extended.base.count = 77;
    assert_eq!(unsafe { fy_result_describe(result, &mut extended.base, null_mut()) }, FY_UNSUPPORTED_ABI);
    assert_eq!(extended.base.count, 77);
    assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
}

#[test]
fn malformed_diagnostics_do_not_destroy_or_modify_a_result() {
    let context = Context::new(BASIC);
    let mut result = context.run(&[0; 12], 16);
    let existing = result;
    let mut diag = FyDiagnostic {
        struct_size: size_of::<FyDiagnostic>() as u32,
        struct_version: 2,
        required_flags: 0,
        data: null_mut(),
        capacity: 0,
        required_bytes: 77,
    };
    let mut required = 77;
    assert_eq!(unsafe { fy_result_copy(result, null_mut(), 0, &mut required, &mut diag) }, FY_UNSUPPORTED_ABI);
    assert_eq!(required, 77);
    assert_eq!(unsafe { fy_result_release(&mut result, &mut diag) }, FY_UNSUPPORTED_ABI);
    assert_eq!(result, existing);
    assert_eq!(diag.required_bytes, 77);
    assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
}
