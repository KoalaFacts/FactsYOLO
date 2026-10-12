// SPDX-License-Identifier: Apache-2.0
//! Explicit test-only support. Not a model format or production backend.
use crate::result::{Candidate, ImageTransform};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const FY_BACKEND_TEST_FIXTURE: u32 = 0xffff0001;
pub const BASIC: u32 = 1;
pub const EMPTY: u32 = 2;
pub const INVALID_OUTPUT: u32 = 3;
pub const BLOCKING: u32 = 4;
pub const PANIC: u32 = 5;
pub const PAYLOAD_DROP_PANIC: u32 = 6;

static LIVE: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];
static EXECUTIONS: AtomicUsize = AtomicUsize::new(0);

/// Model handles, retained model states, and sessions, respectively.
/// Multi-counter snapshots require quiescent allocations, not concurrent tests.
/// This measures live objects, not RSS, allocator caches, or GPU memory.
pub fn live_counts() -> [usize; 3] {
    std::array::from_fn(|index| LIVE[index].load(Ordering::SeqCst))
}

/// Model handles, retained states, sessions and completed results.
pub fn resource_counts() -> [usize; 4] {
    std::array::from_fn(|index| LIVE[index].load(Ordering::SeqCst))
}

pub fn execution_count() -> usize {
    EXECUTIONS.load(Ordering::SeqCst)
}

pub(crate) fn record_execution() {
    EXECUTIONS.fetch_add(1, Ordering::SeqCst);
}

pub(crate) const TRANSFORM: ImageTransform = ImageTransform {
    scale_x: 2.0,
    scale_y: 2.0,
    pad_x: 1.0,
    pad_y: 1.0,
};

pub(crate) const BASIC_CANDIDATES: [Candidate; 4] = [
    Candidate { x_min: -1.0, y_min: -1.0, x_max: 3.0, y_max: 3.0, score: 0.9, class_id: 1, source_index: 0 },
    Candidate { x_min: 1.0, y_min: 1.0, x_max: 5.0, y_max: 5.0, score: 0.9, class_id: 0, source_index: 1 },
    Candidate { x_min: 3.0, y_min: 3.0, x_max: 5.0, y_max: 5.0, score: 0.5, class_id: 0, source_index: 2 },
    Candidate { x_min: 3.0, y_min: 3.0, x_max: 3.0, y_max: 5.0, score: 0.7, class_id: 0, source_index: 3 },
];
pub(crate) const INVALID_CANDIDATES: [Candidate; 1] = [Candidate {
    x_min: f32::NAN,
    score: 0.1,
    ..BASIC_CANDIDATES[0]
}];

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
