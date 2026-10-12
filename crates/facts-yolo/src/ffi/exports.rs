// SPDX-License-Identifier: Apache-2.0
use super::types::{FY_INTERNAL_ERROR, FyAbiInfo, FyDiagnostic};

/// Query the development ABI. This RED-stage stub deliberately fails tests.
///
/// # Safety
/// Callers must provide live, aligned, disjoint storage for advertised prefixes
/// and buffers, with initialized input fields, for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_abi_query(_out: *mut FyAbiInfo, _diag: *mut FyDiagnostic) -> i32 {
    FY_INTERNAL_ERROR
}
