// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "td1")]
#[path = "../../../tests/compat/td1_v1_vectors.rs"]
mod historical;
use facts_yolo::ffi::{handles::*, results::*, types::*};
use fy_tiny_model::{B1, Model, PARAMETERS};
use std::ptr::null_mut;

fn bytes(class: usize) -> Vec<u8> {
    let mut model = Model {
        weights: vec![0.0; PARAMETERS],
    };
    model.weights[0] = 1.0;
    model.weights[B1 + class + 1] = 10.0;
    model.to_bytes().unwrap()
}
fn options() -> FyModelOptions {
    FyModelOptions {
        struct_size: size_of::<FyModelOptions>() as u32,
        struct_version: 1,
        required_flags: 0,
        max_artifact_bytes: 65536,
        max_parser_steps: 65536,
    }
}
fn settings() -> FySessionOptions {
    FySessionOptions {
        struct_size: size_of::<FySessionOptions>() as u32,
        struct_version: 1,
        required_flags: 0,
        backend_id: 1,
        device_kind: 1,
        device_index: 0,
        precision: 1,
        fallback_policy: 0,
        cpu_threads: 1,
        max_image_bytes: 65536,
        max_pixels: 4096,
        max_detections: 4,
    }
}
fn image(pixels: &[u8], format: u32, stride: usize) -> FyImage {
    FyImage {
        struct_size: size_of::<FyImage>() as u32,
        struct_version: 1,
        required_flags: 0,
        data: pixels.as_ptr(),
        byte_length: pixels.len(),
        width: 64,
        height: 64,
        row_stride: stride,
        pixel_format: format,
    }
}
fn detect_options(threshold: f32) -> FyDetectOptions {
    FyDetectOptions {
        struct_size: size_of::<FyDetectOptions>() as u32,
        struct_version: 1,
        required_flags: 0,
        score_threshold: threshold,
        max_detections: 4,
    }
}
struct Context {
    model: *mut FyModel,
    session: *mut FySession,
}
impl Context {
    fn new(bytes: &[u8]) -> Self {
        let mut c = Self {
            model: null_mut(),
            session: null_mut(),
        };
        unsafe {
            assert_eq!(
                fy_model_open(
                    1,
                    bytes.as_ptr(),
                    bytes.len(),
                    &options(),
                    &mut c.model,
                    null_mut()
                ),
                FY_OK
            );
            assert_eq!(
                fy_session_create(1, c.model, &settings(), &mut c.session, null_mut()),
                FY_OK
            );
        }
        c
    }
    fn run(&self, image: &FyImage, threshold: f32) -> *mut FyResult {
        let mut result = null_mut();
        assert_eq!(
            unsafe {
                fy_detect(
                    self.session,
                    image,
                    &detect_options(threshold),
                    &mut result,
                    null_mut(),
                )
            },
            FY_OK
        );
        result
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            fy_session_release(&mut self.session, null_mut());
            fy_model_release(&mut self.model, null_mut());
        }
    }
}
fn copy(result: *mut FyResult) -> Vec<FyDetection> {
    let mut count = 0;
    assert_eq!(
        unsafe { fy_result_copy(result, null_mut(), 0, &mut count, null_mut()) },
        FY_OK
    );
    let mut data = vec![
        FyDetection {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.0,
            y_max: 0.0,
            score: 0.0,
            class_id: 0
        };
        count
    ];
    assert_eq!(
        unsafe {
            fy_result_copy(
                result,
                data.as_mut_ptr(),
                data.len(),
                &mut count,
                null_mut(),
            )
        },
        FY_OK
    );
    data
}
fn red() -> Vec<u8> {
    (0..4096).flat_map(|_| [255, 0, 0]).collect()
}
fn release(mut result: *mut FyResult) {
    assert_eq!(unsafe { fy_result_release(&mut result, null_mut()) }, FY_OK);
    assert!(result.is_null());
}

#[test]
fn real_artifact_parameters_and_pixels_control_detection() {
    for class in 0..2 {
        let c = Context::new(&bytes(class));
        let result = c.run(&image(&red(), FY_RGB8, 192), 0.0);
        let detections = copy(result);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_id, class as u32);
        assert!(detections[0].score > 0.99);
        assert_eq!(
            [
                detections[0].x_min,
                detections[0].y_min,
                detections[0].x_max,
                detections[0].y_max
            ],
            [16.0, 16.0, 48.0, 48.0]
        );
        release(result);
        let result = c.run(&image(&vec![0; 12288], FY_RGB8, 192), 0.0);
        assert!(copy(result).is_empty());
        release(result);
        let result = c.run(&image(&red(), FY_RGB8, 192), 1.0);
        assert!(copy(result).is_empty());
        release(result);
    }
}
#[test]
fn model_and_input_storage_are_not_retained_and_results_outlive_session() {
    let mut source = bytes(0);
    let mut identity = [0; 32];
    identity.copy_from_slice(&source[24..56]);
    let mut c = Context::new(&source);
    source.fill(0);
    drop(source);
    assert_eq!(unsafe { fy_model_release(&mut c.model, null_mut()) }, FY_OK);
    let mut pixels = red();
    let result = c.run(&image(&pixels, FY_RGB8, 192), 0.5);
    pixels.fill(0);
    drop(pixels);
    drop(c);
    let mut info: FyResultInfo = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<FyResultInfo>() as u32;
    info.struct_version = 1;
    assert_eq!(
        unsafe { fy_result_describe(result, &mut info, null_mut()) },
        FY_OK
    );
    assert_eq!(info.model_id, identity);
    assert_eq!(info.backend_id, 1);
    assert_eq!(info.source_width, 64);
    assert_eq!(info.count, 1);
    assert_eq!(copy(result)[0].class_id, 0);
    release(result);
}
#[test]
fn padded_rgb_bgr_and_rgba_share_pixels_and_shape_error_is_reusable() {
    let c = Context::new(&bytes(0));
    let plain = c.run(&image(&red(), FY_RGB8, 192), 0.5);
    let expected = copy(plain);
    release(plain);
    for (format, channels) in [(FY_RGB8, 3), (FY_BGR8, 3), (FY_RGBA8, 4)] {
        let stride = 64 * channels + 7;
        let mut pixels = vec![199; stride * 64];
        for y in 0..64 {
            for x in 0..64 {
                let i = y * stride + x * channels;
                pixels[i..i + 3].copy_from_slice(if format == FY_BGR8 {
                    &[0, 0, 255]
                } else {
                    &[255, 0, 0]
                });
            }
        }
        let result = c.run(&image(&pixels, format, stride), 0.5);
        assert_eq!(copy(result), expected);
        release(result);
    }
    let pixels = red();
    let mut bad = image(&pixels, FY_RGB8, 192);
    bad.width = 63;
    let mut result = null_mut();
    assert_eq!(
        unsafe {
            fy_detect(
                c.session,
                &bad,
                &detect_options(0.5),
                &mut result,
                null_mut(),
            )
        },
        FY_INVALID_ARGUMENT
    );
    assert!(result.is_null());
    let result = c.run(&image(&pixels, FY_RGB8, 192), 0.5);
    assert_eq!(copy(result), expected);
    release(result);
}
#[test]
fn model_limits_corruption_and_output_slot_fail_without_handles() {
    let good = bytes(0);
    let mut corrupt = good.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    let mut trailing = good.clone();
    trailing.push(0);
    for bad in [
        &corrupt[..],
        &good[..100],
        &trailing[..],
        &b"FY_TEST_BASIC"[..],
    ] {
        let mut out = null_mut();
        assert_eq!(
            unsafe { fy_model_open(1, bad.as_ptr(), bad.len(), &options(), &mut out, null_mut()) },
            FY_UNSUPPORTED_MODEL
        );
        assert!(out.is_null());
    }
    for byte_limit in [true, false] {
        let mut o = options();
        if byte_limit {
            o.max_artifact_bytes = good.len() - 1;
        } else {
            o.max_parser_steps = good.len() - 1;
        }
        let mut out = null_mut();
        assert_eq!(
            unsafe { fy_model_open(1, good.as_ptr(), good.len(), &o, &mut out, null_mut()) },
            FY_LIMIT_EXCEEDED
        );
        assert!(out.is_null());
    }
    let mut c = Context::new(&good);
    let before = c.model;
    assert_eq!(
        unsafe {
            fy_model_open(
                1,
                good.as_ptr(),
                good.len(),
                &options(),
                &mut c.model,
                null_mut(),
            )
        },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(before, c.model);
}
#[test]
fn actual_capability_is_cpu_and_unsupported_device_does_not_create_session() {
    let c = Context::new(&bytes(0));
    let mut count = 0;
    assert_eq!(
        unsafe { fy_capabilities_copy(null_mut(), 0, &mut count, null_mut()) },
        FY_OK
    );
    let mut caps = vec![
        FyCapability {
            backend_id: 0,
            device_kind: 0,
            device_index: 0,
            flags: 0,
            precision_mask: 0
        };
        count
    ];
    assert_eq!(
        unsafe { fy_capabilities_copy(caps.as_mut_ptr(), caps.len(), &mut count, null_mut()) },
        FY_OK
    );
    assert!(caps.iter().any(|c| c.backend_id == 1
        && c.flags == FY_CAP_COMPILED | FY_CAP_AVAILABLE
        && c.device_kind == FY_DEVICE_CPU
        && c.precision_mask == 1));
    for (device, threads, status) in [
        (2, 1, FY_DEVICE_UNAVAILABLE),
        (1, 2, FY_UNSUPPORTED_FEATURE),
    ] {
        let mut out = null_mut();
        let mut o = settings();
        o.device_kind = device;
        o.cpu_threads = threads;
        assert_eq!(
            unsafe { fy_session_create(1, c.model, &o, &mut out, null_mut()) },
            status
        );
        assert!(out.is_null());
    }
}

fn historical_open(bytes: &[u8], options: &FyModelOptions, major: u32, expected: i32) {
    let mut out = null_mut();
    let status = unsafe {
        fy_model_open(
            major,
            bytes.as_ptr(),
            bytes.len(),
            options,
            &mut out,
            null_mut(),
        )
    };
    assert_eq!(status, expected);
    if status == FY_OK {
        assert!(!out.is_null());
        assert_eq!(unsafe { fy_model_release(&mut out, null_mut()) }, FY_OK);
    }
    assert!(
        out.is_null(),
        "failed creation/release must preserve the empty slot"
    );
}

#[test]
fn historical_model_versions_and_numeric_statuses() {
    for vector in historical::VECTORS {
        let expected = if vector.accepted {
            FY_OK
        } else {
            FY_UNSUPPORTED_MODEL
        };
        historical_open(&vector.bytes(), &options(), 1, expected);
    }
    // A malformed artifact must not leave global/partial state that prevents
    // later valid creation. These are fixed synthetic bytes, not trained weights.
    historical_open(
        &historical::vector("red-gate").bytes(),
        &options(),
        1,
        FY_OK,
    );
}

#[test]
fn historical_model_budgets_and_precedence() {
    let good = historical::vector("red-gate").bytes();
    for parser_budget in [false, true] {
        for (budget, expected) in [
            (0, FY_INVALID_ARGUMENT),
            (51_283, FY_LIMIT_EXCEEDED),
            (51_284, FY_OK),
            (51_285, FY_OK),
        ] {
            let mut o = options();
            if parser_budget {
                o.max_parser_steps = budget;
            } else {
                o.max_artifact_bytes = budget;
            }
            historical_open(&good, &o, 1, expected);
        }
    }
    let mut o = options();
    o.max_parser_steps = 51_283;
    historical_open(
        &historical::vector("schema-two").bytes(),
        &o,
        1,
        FY_UNSUPPORTED_MODEL,
    );
    historical_open(
        &historical::vector("architecture-two").bytes(),
        &o,
        1,
        FY_LIMIT_EXCEEDED,
    );
    o.max_artifact_bytes = 51_283;
    historical_open(
        &historical::vector("schema-two").bytes(),
        &o,
        1,
        FY_LIMIT_EXCEEDED,
    );
    // ABI and descriptor errors keep their existing categories, separate from
    // schema/architecture rejection. All pointers refer to real live storage.
    historical_open(&good, &options(), 2, FY_UNSUPPORTED_ABI);
    let mut o = options();
    o.struct_version = 2;
    historical_open(&good, &o, 1, FY_UNSUPPORTED_ABI);
    o = options();
    o.required_flags = 1;
    historical_open(&good, &o, 1, FY_UNSUPPORTED_FEATURE);
    historical_open(&good, &options(), 1, FY_OK);
}

#[test]
fn historical_half_score_identity_and_recovery() {
    let bytes = historical::vector("half-score").bytes();
    let c = Context::new(&bytes);
    let pixels = vec![0; 12_288];
    let valid = image(&pixels, FY_RGB8, 192);
    let mut invalid = valid;
    invalid.width = 63;
    let mut out = c.run(&valid, 0.5);
    let before = out;
    assert_eq!(
        unsafe {
            fy_detect(
                c.session,
                &invalid,
                &detect_options(0.5),
                &mut out,
                null_mut(),
            )
        },
        FY_INVALID_ARGUMENT
    );
    assert_eq!(out, before);
    release(out);
    out = null_mut();
    assert_eq!(
        unsafe {
            fy_detect(
                c.session,
                &invalid,
                &detect_options(0.5),
                &mut out,
                null_mut(),
            )
        },
        FY_INVALID_ARGUMENT
    );
    assert!(out.is_null());
    for threshold in [f32::NAN, -f32::EPSILON, 1.0 + f32::EPSILON] {
        assert_eq!(
            unsafe {
                fy_detect(
                    c.session,
                    &valid,
                    &detect_options(threshold),
                    &mut out,
                    null_mut(),
                )
            },
            FY_INVALID_ARGUMENT
        );
        assert!(out.is_null());
    }
    let result = c.run(&valid, 0.5);
    let detections = copy(result);
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].class_id, 0);
    assert_eq!(detections[0].score, 0.5);
    assert_eq!(
        [
            detections[0].x_min,
            detections[0].y_min,
            detections[0].x_max,
            detections[0].y_max
        ],
        [16.0, 16.0, 48.0, 48.0]
    );
    let mut info: FyResultInfo = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<FyResultInfo>() as u32;
    info.struct_version = 1;
    assert_eq!(
        unsafe { fy_result_describe(result, &mut info, null_mut()) },
        FY_OK
    );
    assert_eq!(&info.model_id, &bytes[24..56]);
    release(result);
    let above = f32::from_bits(0.5_f32.to_bits() + 1);
    let result = c.run(&valid, above);
    assert!(copy(result).is_empty());
    release(result);
}
