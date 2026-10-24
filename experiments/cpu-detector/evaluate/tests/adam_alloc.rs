// SPDX-License-Identifier: Apache-2.0
//! Isolate the optimizer's own allocations from test-harness and other threads.
use fy_cpu_evaluate::training::adam_step;
use fy_tiny_model::PARAMETERS;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

std::thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn note_allocation() {
    if MEASURING.try_with(|flag| flag.get()).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

// SAFETY: Every operation delegates to System with the original layout/pointer.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_allocation();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn successful_adam_step_does_not_allocate_per_parameter_staging() {
    let mut weights = vec![0.1; PARAMETERS];
    let gradient = vec![0.01; PARAMETERS];
    let mut m = vec![0.0; PARAMETERS];
    let mut v = vec![0.0; PARAMETERS];

    // Initialize the allocator and thread-local counters before measurement.
    adam_step(&mut weights, &gradient, &mut m, &mut v, 1).unwrap();
    ALLOCATIONS.with(|count| count.set(0));
    MEASURING.with(|flag| flag.set(true));
    let result = adam_step(&mut weights, &gradient, &mut m, &mut v, 2);
    MEASURING.with(|flag| flag.set(false));
    result.unwrap();
    assert_eq!(
        ALLOCATIONS.with(|count| count.get()),
        0,
        "Adam must not allocate a per-step staging Vec"
    );
}
