// SPDX-License-Identifier: Apache-2.0
//! Explicit test-only support. Not a model format or production backend.
pub const FY_BACKEND_TEST_FIXTURE: u32 = 0xffff0001;
pub const BASIC: u32 = 1;
pub const EMPTY: u32 = 2;
pub const INVALID_OUTPUT: u32 = 3;
pub const BLOCKING: u32 = 4;
pub const PANIC: u32 = 5;
pub const PAYLOAD_DROP_PANIC: u32 = 6;

/// RED checkpoint; actual resource instrumentation follows the failing tests.
pub fn live_counts() -> [usize; 3] {
    [0; 3]
}
