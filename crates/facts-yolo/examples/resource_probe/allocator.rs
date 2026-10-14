// SPDX-License-Identifier: Apache-2.0
//! Instrument only this executable, never the distributed library allocator.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct CountedSystem;
static BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn live_bytes() -> usize {
    BYTES.load(Ordering::SeqCst)
}

// SAFETY: each request is delegated unchanged to System. Counters allocate
// nothing and cannot unwind. Failed allocations/reallocations leave counts alone.
unsafe impl GlobalAlloc for CountedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            BYTES.fetch_add(layout.size(), Ordering::SeqCst);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            BYTES.fetch_add(layout.size(), Ordering::SeqCst);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        BYTES.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        if !pointer.is_null() {
            if size >= layout.size() {
                BYTES.fetch_add(size - layout.size(), Ordering::SeqCst);
            } else {
                BYTES.fetch_sub(layout.size() - size, Ordering::SeqCst);
            }
        }
        pointer
    }
}
