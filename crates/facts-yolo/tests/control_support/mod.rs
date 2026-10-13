// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]
use facts_yolo::ffi::{handles::*, results::*, types::*};
use std::ptr::null_mut;
use std::sync::Mutex;

pub static SERIAL: Mutex<()> = Mutex::new(());

pub struct Owner {
    pub session: *mut FySession,
}

// SAFETY: test methods use only shared native operations and per-call buffers.
// Scoped threads borrow Owner; Drop cannot release until those borrows end.
unsafe impl Sync for Owner {}

impl Owner {
    pub fn new(fixture: u32) -> Self {
        let mut model = null_mut();
        let mut session = null_mut();
        let options = FySessionOptions {
            struct_size: size_of::<FySessionOptions>() as u32,
            struct_version: 1,
            required_flags: 0,
            backend_id: facts_yolo::fixtures::FY_BACKEND_TEST_FIXTURE,
            device_kind: FY_DEVICE_CPU,
            device_index: 0,
            precision: FY_PRECISION_F32,
            fallback_policy: FY_FALLBACK_NONE,
            cpu_threads: 1,
            max_image_bytes: 1024,
            max_pixels: 64,
            max_detections: 16,
        };
        unsafe {
            assert_eq!(
                fy_test_model_create(1, fixture, &mut model, null_mut()),
                FY_OK
            );
            let status = fy_session_create(1, model, &options, &mut session, null_mut());
            assert_eq!(fy_model_release(&mut model, null_mut()), FY_OK);
            assert_eq!(status, FY_OK);
        }
        Self { session }
    }

    pub fn call(&self, threshold: f32) -> (i32, usize, String) {
        let pixels = [42_u8; 12];
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
            score_threshold: threshold,
            max_detections: 16,
        };
        let mut text = [0xA5_u8; 128];
        let mut diagnostic = FyDiagnostic {
            struct_size: size_of::<FyDiagnostic>() as u32,
            struct_version: 1,
            required_flags: 0,
            data: text.as_mut_ptr().cast(),
            capacity: text.len(),
            required_bytes: usize::MAX,
        };
        let mut result = null_mut();
        let mut count = 0;
        let status =
            unsafe { fy_detect(self.session, &image, &options, &mut result, &mut diagnostic) };
        if status == FY_OK {
            assert!(!result.is_null());
            unsafe {
                assert_eq!(
                    fy_result_copy(result, null_mut(), 0, &mut count, null_mut()),
                    FY_OK
                );
                assert_eq!(fy_result_release(&mut result, null_mut()), FY_OK);
            }
        } else {
            assert!(result.is_null(), "failed detection transferred ownership");
        }
        let end = text.iter().position(|byte| *byte == 0).unwrap();
        let message = std::str::from_utf8(&text[..end]).unwrap().to_owned();
        (status, count, message)
    }

    pub fn wait(&self) -> i32 {
        unsafe { facts_yolo::ffi::testing::fy_test_session_wait_running(self.session, 3000) }
    }

    pub fn resume(&self) -> i32 {
        unsafe { facts_yolo::ffi::testing::fy_test_session_resume(self.session) }
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        // SAFETY: Owner has exclusive destruction after scoped threads join.
        let status = unsafe { fy_session_release(&mut self.session, null_mut()) };
        assert_eq!(status, FY_OK);
    }
}
