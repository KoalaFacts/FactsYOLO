// SPDX-License-Identifier: Apache-2.0
//! RED checkpoint for independently owned results; no behavior yet.
#![allow(clippy::missing_safety_doc)]
use super::{handles::FySession, types::*};

#[repr(C)]
pub struct FyResult {
    _private: [u8; 0],
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_detect(
    _session: *mut FySession,
    _image: *const FyImage,
    _options: *const FyDetectOptions,
    _out: *mut *mut FyResult,
    _diag: *mut FyDiagnostic,
) -> i32 {
    FY_INTERNAL_ERROR
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_result_describe(
    _result: *const FyResult,
    _out: *mut FyResultInfo,
    _diag: *mut FyDiagnostic,
) -> i32 {
    FY_INTERNAL_ERROR
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_result_copy(
    _result: *const FyResult,
    _dst: *mut FyDetection,
    _capacity: usize,
    _required: *mut usize,
    _diag: *mut FyDiagnostic,
) -> i32 {
    FY_INTERNAL_ERROR
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_result_release(
    _slot: *mut *mut FyResult,
    _diag: *mut FyDiagnostic,
) -> i32 {
    FY_INTERNAL_ERROR
}
