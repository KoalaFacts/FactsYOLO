// SPDX-License-Identifier: Apache-2.0
//! Compare the historical allocating TD1 compact inference path with Model::detect.
//! Measures portable reference inference only, not native FFI, SDKs or training.
use fy_tiny_model::{decode, features, Model, Rng, PARAMETERS};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

const IMAGES: usize = 8;
const ITERATIONS: usize = 256;
const REPEATS: usize = 7;

// Counts successful allocation/reallocation requests and requested bytes,
// not RSS, peak allocation, or allocator internals. Measurements are serial.
static REQUESTS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record(ptr: *mut u8, size: usize) {
    if !ptr.is_null() {
        REQUESTS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
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
        let result = unsafe { System.realloc(ptr, layout, size) };
        record(result, size);
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Historical,
    Current,
}

// The pre-PR #33 portable Model::detect implementation, without changes to
// feature order, forward math, decoding, model or sample values.
fn historical(model: &Model, pixels: &[u8]) -> Result<Option<fy_tiny_model::Detection>, String> {
    decode(&model.forward(&features(pixels)?)?)
}

#[derive(Serialize)]
struct Measurement {
    elapsed_ns: u128,
    allocation_requests: u64,
    requested_bytes: u64,
}

fn measure(mode: Mode, model: &Model, images: &[Vec<u8>]) -> Result<Measurement, String> {
    let requests_before = REQUESTS.load(Ordering::Relaxed);
    let bytes_before = BYTES.load(Ordering::Relaxed);
    let started = Instant::now();

    match mode {
        Mode::Historical => {
            for _ in 0..ITERATIONS {
                for image in images {
                    black_box(historical(model, black_box(image))?);
                }
            }
        }
        Mode::Current => {
            for _ in 0..ITERATIONS {
                for image in images {
                    black_box(model.detect(black_box(image))?);
                }
            }
        }
    }

    let elapsed_ns = started.elapsed().as_nanos();
    Ok(Measurement {
        elapsed_ns,
        allocation_requests: REQUESTS.load(Ordering::Relaxed) - requests_before,
        requested_bytes: BYTES.load(Ordering::Relaxed) - bytes_before,
    })
}

fn run() -> Result<(), String> {
    if std::env::args().skip(1).collect::<Vec<_>>() != ["--measure"] {
        return Err("usage: fy-inference-benchmark --measure".into());
    }

    let mut rng = Rng::new(2026);
    let images: Vec<Vec<u8>> = (0..IMAGES)
        .map(|_| {
            (0..64 * 64 * 3)
                .map(|_| (rng.next_u32() & 255) as u8)
                .collect()
        })
        .collect();
    let mut model = Model::initialized(404);
    // Force a nonempty classification so that all detections execute decode,
    // while leaving the original seeded network and both paths identical.
    model.weights[PARAMETERS - 6] = 8.0;

    // Check each distinct image before measuring; equality includes the
    // winning class, score and all four floating-point box corners.
    for (i, image) in images.iter().enumerate() {
        let old = historical(&model, image)?;
        let current = model.detect(image)?;
        if old.is_none() || old != current {
            return Err(format!("TD1 compact inference parity failed for image {i}"));
        }
    }
    let mut background = model.clone();
    background.weights[PARAMETERS - 7] = 16.0;
    for (i, image) in images.iter().enumerate() {
        let old = historical(&background, image)?;
        if old.is_some() || old != background.detect(image)? {
            return Err(format!("TD1 background inference parity failed for image {i}"));
        }
    }

    // Warm both paths outside the measured interval.
    for image in &images {
        black_box(historical(&model, image)?);
        black_box(model.detect(image)?);
    }

    let mut historical_runs = Vec::with_capacity(REPEATS);
    let mut current_runs = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        if repeat % 2 == 0 {
            historical_runs.push(measure(Mode::Historical, &model, &images)?);
            current_runs.push(measure(Mode::Current, &model, &images)?);
        } else {
            current_runs.push(measure(Mode::Current, &model, &images)?);
            historical_runs.push(measure(Mode::Historical, &model, &images)?);
        }
    }

    let mut inputs_hash = Sha256::new();
    for image in &images {
        inputs_hash.update(image);
    }
    let mut model_hash = Sha256::new();
    for weight in &model.weights {
        model_hash.update(weight.to_le_bytes());
    }

    println!(
        "{}",
        json!({
            "scope": "portable TD1 compact RGB inference, preprocessing + forward + decode; excludes FFI, SDK and training",
            "source_commit": option_env!("FACTSYOLO_BUILD_ID").unwrap_or("unknown"),
            "build": "release",
            "image_seed": 2026,
            "initialization_seed": 404,
            "images": IMAGES,
            "iterations_per_repeat": ITERATIONS,
            "repetitions": REPEATS,
            "inputs_sha256": format!("{:x}", inputs_hash.finalize()),
            "model_sha256": format!("{:x}", model_hash.finalize()),
            "exact_positive_and_background_parity": true,
            "allocator_metric": "successful alloc/alloc_zeroed/realloc requests and requested bytes, not peak RSS",
            "historical": historical_runs,
            "current": current_runs
        })
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
