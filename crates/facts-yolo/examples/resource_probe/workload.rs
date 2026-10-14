// SPDX-License-Identifier: Apache-2.0
use facts_yolo::{
    ffi::{handles::*, results::*, types::*},
    fixtures::*,
};
use std::ptr::null_mut;

pub fn cycle() {
    let settings = FySessionOptions {
        struct_size: size_of::<FySessionOptions>() as u32,
        struct_version: 1,
        required_flags: 0,
        backend_id: FY_BACKEND_TEST_FIXTURE,
        device_kind: FY_DEVICE_CPU,
        device_index: 0,
        precision: FY_PRECISION_F32,
        fallback_policy: FY_FALLBACK_NONE,
        cpu_threads: 1,
        max_image_bytes: 12,
        max_pixels: 4,
        max_detections: 3,
    };
    let pixels = [0_u8; 12];
    let image = FyImage {
        struct_size: size_of::<FyImage>() as u32,
        struct_version: 1,
        required_flags: 0,
        data: pixels.as_ptr(),
        byte_length: pixels.len(),
        width: 2,
        height: 2,
        row_stride: 6,
        pixel_format: FY_RGB8,
    };
    let options = FyDetectOptions {
        struct_size: size_of::<FyDetectOptions>() as u32,
        struct_version: 1,
        required_flags: 0,
        score_threshold: 0.5,
        max_detections: 3,
    };
    let mut model = null_mut();
    let mut session = null_mut();
    let mut result = null_mut();
    // SAFETY: all descriptors/buffers/slots are real disjoint live allocations;
    // each resource is released only after its last synchronous call.
    unsafe {
        assert_eq!(
            fy_test_model_create(1, BASIC, &mut model, null_mut()),
            FY_OK
        );
        let existing = model;
        assert_eq!(
            fy_test_model_create(1, BASIC, &mut model, null_mut()),
            FY_INVALID_ARGUMENT
        );
        assert_eq!(model, existing);
        let mut invalid = settings;
        invalid.cpu_threads = 0;
        assert_eq!(
            fy_session_create(1, model, &invalid, &mut session, null_mut()),
            FY_INVALID_ARGUMENT
        );
        assert!(session.is_null());
        assert_eq!(
            fy_session_create(1, model, &settings, &mut session, null_mut()),
            FY_OK
        );
        assert_eq!(fy_model_release(&mut model, null_mut()), FY_OK);
        let executions = execution_count();
        assert_eq!(
            fy_detect(session, &image, &options, &mut result, null_mut()),
            FY_OK
        );
        let existing = result;
        assert_eq!(
            fy_detect(session, &image, &options, &mut result, null_mut()),
            FY_INVALID_ARGUMENT
        );
        assert_eq!(result, existing);
        assert_eq!(fy_session_release(&mut session, null_mut()), FY_OK);
        assert_eq!(resource_counts(), [0, 0, 0, 1]);
        let sentinel = FyDetection {
            x_min: -7.0,
            y_min: -7.0,
            x_max: -7.0,
            y_max: -7.0,
            score: -7.0,
            class_id: 77,
        };
        for _ in 0..3 {
            let mut count = 99;
            assert_eq!(
                fy_result_copy(result, null_mut(), 0, &mut count, null_mut()),
                FY_OK
            );
            assert_eq!(count, 3);
            let mut copy = [sentinel; 3];
            assert_eq!(
                fy_result_copy(result, copy.as_mut_ptr(), 2, &mut count, null_mut()),
                FY_BUFFER_TOO_SMALL
            );
            assert_eq!(copy, [sentinel; 3]);
            assert_eq!(
                fy_result_copy(result, copy.as_mut_ptr(), 3, &mut count, null_mut()),
                FY_OK
            );
            assert_eq!(copy[0].score, 0.9);
        }
        assert_eq!(execution_count(), executions + 1);
        assert_eq!(fy_result_release(&mut result, null_mut()), FY_OK);
        assert_eq!(fy_result_release(&mut result, null_mut()), FY_OK);
    }
}
