// SPDX-License-Identifier: Apache-2.0
//! RED checkpoint: test-only controls deliberately reject pending implementation.
use super::{handles::FySession, types::FY_INTERNAL_ERROR};

/// # Safety
/// A non-null session must be live and correctly typed, with no concurrent release.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_test_session_wait_running(
    _session: *const FySession,
    _timeout_ms: u32,
) -> i32 {
    FY_INTERNAL_ERROR
}

/// # Safety
/// A non-null session must be live and correctly typed, with no concurrent release.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_test_session_resume(_session: *const FySession) -> i32 {
    FY_INTERNAL_ERROR
}
