// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "contract-fixtures")]
use facts_yolo::{
    ffi::{handles::*, results::*, types::*},
    fixtures::{BASIC, FY_BACKEND_TEST_FIXTURE, execution_count, resource_counts},
};
use std::ptr::null_mut;

// A separate one-test executable gives these process-global instrumentation
// counters a quiescent baseline, independent of parallel integration tests.
#[test]
fn result_lifetimes_and_copy_retries_have_exact_resource_and_execution_counts() {
    let baseline = resource_counts();
    let mut model = null_mut();
    let mut session = null_mut();
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
        byte_length: 12,
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
    unsafe {
        assert_eq!(fy_test_model_create(1, BASIC, &mut model, null_mut()), FY_OK);
        assert_eq!(fy_session_create(1, model, &settings, &mut session, null_mut()), FY_OK);
        assert_eq!(fy_model_release(&mut model, null_mut()), FY_OK);
        let active = resource_counts();
        let executions = execution_count();
        for index in 0..10_000 {
            let mut result = null_mut();
            assert_eq!(fy_detect(session, &image, &options, &mut result, null_mut()), FY_OK);
            assert_eq!(resource_counts()[3], baseline[3] + 1);
            for _ in 0..3 {
                let mut required = 77;
                assert_eq!(fy_result_copy(result, null_mut(), 0, &mut required, null_mut()), FY_OK);
                assert_eq!(required, 3);
            }
            assert_eq!(execution_count(), executions + index + 1);
            assert_eq!(fy_result_release(&mut result, null_mut()), FY_OK);
            assert_eq!(resource_counts(), active);
        }
        assert_eq!(fy_session_release(&mut session, null_mut()), FY_OK);
        assert_eq!(resource_counts(), baseline);
    }
}
