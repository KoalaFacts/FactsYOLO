// SPDX-License-Identifier: Apache-2.0
//! Explicit test-only support. Not a model format or production backend.
use std::sync::atomic::{AtomicUsize, Ordering};

pub const FY_BACKEND_TEST_FIXTURE: u32 = 0xffff0001;
pub const BASIC: u32 = 1;
pub const EMPTY: u32 = 2;
pub const INVALID_OUTPUT: u32 = 3;
pub const BLOCKING: u32 = 4;
pub const PANIC: u32 = 5;
pub const PAYLOAD_DROP_PANIC: u32 = 6;

static LIVE: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

/// Model handles, retained model states, and sessions, respectively.
/// A multi-counter snapshot is only coherent while allocations are quiescent.
/// This measures live objects, not RSS, allocator caches, or GPU memory.
pub fn live_counts() -> [usize; 3] {
    std::array::from_fn(|index| LIVE[index].load(Ordering::SeqCst))
}

/// RAII instrumentation, compiled out entirely from normal builds.
pub(crate) struct Live {
    index: usize,
}

impl Live {
    pub(crate) fn new(index: usize) -> Self {
        LIVE[index].fetch_add(1, Ordering::SeqCst);
        Self { index }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        LIVE[self.index].fetch_sub(1, Ordering::SeqCst);
    }
}
