// SPDX-License-Identifier: Apache-2.0
//! Reproducible paired TD1 portable forward baseline: mutable reference versus
//! artifact-validated immutable inference. Excludes training, preprocessing,
//! FFI and C#/Go/TypeScript SDKs; not a production detector benchmark.
use fy_tiny_model::{InferenceModel, Model, INPUTS, PARAMETERS};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

const CALLS: usize = 4_096;
const WARMUP: usize = 1_024;
const REPEATS: usize = 7;
static REQUESTS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record(ptr: *mut u8, bytes: usize) {
    if !ptr.is_null() {
        REQUESTS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        record(ptr, layout.size());
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        record(ptr, layout.size());
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, size) };
        record(new_ptr, size);
        new_ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn measure(
    label: &str,
    trial: usize,
    model: &Model,
    frozen: &InferenceModel,
    features: &[f32; INPUTS],
) {
    let before_requests = REQUESTS.load(Ordering::Relaxed);
    let before_bytes = BYTES.load(Ordering::Relaxed);
    let start = Instant::now();
    let mut digest = 0_u64;
    for _ in 0..CALLS {
        let out = match label {
            "mutable" => black_box(model).forward(black_box(features)).unwrap(),
            "immutable" => black_box(frozen).forward(black_box(features)).unwrap(),
            _ => unreachable!(),
        };
        for value in out {
            digest = digest.wrapping_add(value.to_bits() as u64);
        }
        black_box(digest);
    }
    let ns = start.elapsed().as_nanos();
    let allocations = REQUESTS.load(Ordering::Relaxed) - before_requests;
    let bytes = BYTES.load(Ordering::Relaxed) - before_bytes;
    println!(
        "{label},{trial},{CALLS},{ns},{:.3},{digest},{allocations},{bytes}",
        ns as f64 / CALLS as f64
    );
}

fn main() {
    if std::env::args().skip(1).collect::<Vec<_>>() != ["--measure"] {
        eprintln!("usage: validated_forward --measure");
        std::process::exit(1);
    }
    let model = Model::initialized(0xFACC_FEED);
    assert_eq!(model.weights.len(), PARAMETERS);
    let frozen = InferenceModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    let features = [0.5_f32; INPUTS];

    // Ensure bit-for-bit identical output on the same model, features and host.
    assert_eq!(
        model.forward(&features).unwrap(),
        frozen.forward(&features).unwrap()
    );
    for _ in 0..WARMUP {
        black_box(black_box(&model).forward(black_box(&features)).unwrap());
        black_box(black_box(&frozen).forward(black_box(&features)).unwrap());
    }
    println!(
        "mode,trial,calls,ns_total,ns_per_call,result_digest,allocation_requests,requested_bytes"
    );
    for trial in 0..REPEATS {
        if trial % 2 == 0 {
            measure("mutable", trial, &model, &frozen, &features);
            measure("immutable", trial, &model, &frozen, &features);
        } else {
            measure("immutable", trial, &model, &frozen, &features);
            measure("mutable", trial, &model, &frozen, &features);
        }
    }
}
