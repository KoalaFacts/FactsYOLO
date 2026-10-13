// SPDX-License-Identifier: Apache-2.0
//! Explicit fixture controls. This entire module is absent from normal builds.
use super::handles::{FySession, SessionHandle, entry};
use crate::error::INVALID;
use std::ptr::null_mut;

/// Wait for the blocking test fixture, without taking its execution lock.
///
/// # Safety
/// A non-null aligned session must be a live correctly typed handle from this
/// library and cannot be released concurrently. This is not a production API.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_test_session_wait_running(
    session: *const FySession,
    timeout_ms: u32,
) -> i32 {
    unsafe {
        entry(null_mut(), || {
            let raw = session.cast::<SessionHandle>();
            if raw.is_null() || !raw.is_aligned() {
                return Err(INVALID);
            }
            // SAFETY: shared access only; the caller holds the live session.
            (*raw).session.test_wait_running(timeout_ms)
        })
    }
}

/// Allow one blocked fixture invocation to continue; pre-entry cleanup is allowed.
///
/// # Safety
/// The pointer obeys the same live-handle rules as wait_running. The operation
/// does not release or cancel a session and is absent from production builds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_test_session_resume(session: *const FySession) -> i32 {
    unsafe {
        entry(null_mut(), || {
            let raw = session.cast::<SessionHandle>();
            if raw.is_null() || !raw.is_aligned() {
                return Err(INVALID);
            }
            (*raw).session.test_resume()
        })
    }
}
