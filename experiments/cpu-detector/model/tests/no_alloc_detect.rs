// SPDX-License-Identifier: Apache-2.0
// Single-test executable: count only allocations performed by compact TD1 inference.
use fy_tiny_model::{Model, PARAMETERS};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct TrackedSystem;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: TrackedSystem = TrackedSystem;

fn record(ptr: *mut u8) {
    if !ptr.is_null() && COUNTING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        record(ptr);
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        record(ptr);
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let resized = unsafe { System.realloc(ptr, layout, new_size) };
        record(resized);
        resized
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[test]
fn compact_rgb_detection_matches_borrowed_view_without_heap_requests() {
    // Setup and assertions happen outside the counted section. There is only
    // one test in this executable, so no other test thread can pollute counts.
    let mut model = Model {
        weights: vec![0.0; PARAMETERS],
    };
    model.weights[PARAMETERS - 6] = 8.0;
    let pixels: Vec<u8> = (0_usize..64 * 64 * 3)
        .map(|i| (i.wrapping_mul(31).wrapping_add(17) % 256) as u8)
        .collect();

    let expected = model
        .detect_with_rgb(|x, y| {
            let at = (y * 64 + x) * 3;
            Some([pixels[at], pixels[at + 1], pixels[at + 2]])
        })
        .unwrap();
    assert!(expected.is_some());

    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::SeqCst);
    let observed = model.detect(&pixels);
    COUNTING.store(false, Ordering::SeqCst);

    assert_eq!(observed.unwrap(), expected);
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        0,
        "compact RGB inference must not allocate a redundant 192-float buffer"
    );

    assert_eq!(
        model.detect(&pixels[..pixels.len() - 1]).unwrap_err(),
        "TD1 requires exactly 64x64 RGB bytes"
    );
    let mut oversized = pixels;
    oversized.push(0);
    assert_eq!(
        model.detect(&oversized).unwrap_err(),
        "TD1 requires exactly 64x64 RGB bytes"
    );
}
