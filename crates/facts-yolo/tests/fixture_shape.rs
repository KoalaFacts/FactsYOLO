// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "contract-fixtures")]
use facts_yolo::{
    ffi::{handles::*, results::*, types::*},
    fixtures::{BASIC, FY_BACKEND_TEST_FIXTURE, execution_count, resource_counts},
};
use std::ptr::null_mut;

#[test]
fn invalid_fixture_shape_never_executes_or_poisons_the_session() {
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
        max_image_bytes: 128,
        max_pixels: 16,
        max_detections: 16,
    };
    let opts = FyDetectOptions {
        struct_size: size_of::<FyDetectOptions>() as u32,
        struct_version: 1,
        required_flags: 0,
        score_threshold: 0.5,
        max_detections: 16,
    };
    let pixels = [0_u8; 24];
    let mut image = FyImage {
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
    unsafe {
        assert_eq!(
            fy_test_model_create(1, BASIC, &mut model, null_mut()),
            FY_OK
        );
        assert_eq!(
            fy_session_create(1, model, &settings, &mut session, null_mut()),
            FY_OK
        );
        let active = resource_counts();
        let executions = execution_count();
        for (width, height) in [(1, 2), (2, 1), (3, 2), (2, 3)] {
            image.width = width;
            image.height = height;
            image.row_stride = width as usize * 3;
            let mut result = null_mut();
            let status = fy_detect(session, &image, &opts, &mut result, null_mut());
            // Clean an unexpected success before failing so the test itself
            // does not retain native resources at the RED checkpoint.
            if !result.is_null() {
                assert_eq!(fy_result_release(&mut result, null_mut()), FY_OK);
            }
            assert_eq!(status, FY_INVALID_ARGUMENT);
            assert_eq!(execution_count(), executions);
            assert_eq!(resource_counts(), active);
        }
        image.width = 2;
        image.height = 2;
        image.row_stride = 6;
        let mut result = null_mut();
        assert_eq!(
            fy_detect(session, &image, &opts, &mut result, null_mut()),
            FY_OK
        );
        assert_eq!(fy_result_release(&mut result, null_mut()), FY_OK);
        assert_eq!(fy_session_release(&mut session, null_mut()), FY_OK);
        assert_eq!(fy_model_release(&mut model, null_mut()), FY_OK);
        assert_eq!(resource_counts(), baseline);
    }
}
