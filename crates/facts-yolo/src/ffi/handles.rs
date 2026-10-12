// SPDX-License-Identifier: Apache-2.0
//! RED checkpoint: deliberately nonfunctional ownership entry points.
use super::types::*;

#[repr(C)]
pub struct FyModel {
    _private: [u8; 0],
}
#[repr(C)]
pub struct FySession {
    _private: [u8; 0],
}

macro_rules! stub {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        /// Deliberately rejecting RED checkpoint.
        /// # Safety
        /// Callers must obey the plan's raw allocation, lifetime and disjointness contract.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name($($arg: $ty),*) -> i32 {
            FY_INTERNAL_ERROR
        }
    };
}
stub!(fy_model_open(_abi: u32, _bytes: *const u8, _length: usize, _options: *const FyModelOptions, _out: *mut *mut FyModel, _diag: *mut FyDiagnostic));
stub!(fy_model_release(_slot: *mut *mut FyModel, _diag: *mut FyDiagnostic));
stub!(fy_session_create(_abi: u32, _model: *const FyModel, _options: *const FySessionOptions, _out: *mut *mut FySession, _diag: *mut FyDiagnostic));
stub!(fy_session_describe(_session: *const FySession, _out: *mut FySessionInfo, _diag: *mut FyDiagnostic));
stub!(fy_session_release(_slot: *mut *mut FySession, _diag: *mut FyDiagnostic));
stub!(fy_capabilities_copy(_dst: *mut FyCapability, _capacity: usize, _required: *mut usize, _diag: *mut FyDiagnostic));
#[cfg(feature = "contract-fixtures")]
stub!(fy_test_model_create(_abi: u32, _fixture: u32, _out: *mut *mut FyModel, _diag: *mut FyDiagnostic));
