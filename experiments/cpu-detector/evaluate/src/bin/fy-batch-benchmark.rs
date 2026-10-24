// SPDX-License-Identifier: Apache-2.0
//! Measures the exact training batch assembly (not model training).
//! The allocating reference reproduces the loop from before PR #27, while
//! the reused implementation calls the production fill_training_batch function.
use fy_cpu_evaluate::training::fill_training_batch;
use fy_tiny_model::{dataset, features, Rng, Sample, INPUTS};
use serde_json::json;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

const STEPS: usize = 2_000;
const BATCH: usize = 64;
const REPEATS: usize = 7;

// These count successful allocation requests and requested bytes, not live
// bytes, peak RSS or allocator-internal work. Benchmark runs are single-threaded.
static ALLOCATION_REQUESTS: AtomicU64 = AtomicU64::new(0);
static REQUESTED_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            ALLOCATION_REQUESTS.fetch_add(1, Ordering::Relaxed);
            REQUESTED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        result
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            ALLOCATION_REQUESTS.fetch_add(1, Ordering::Relaxed);
            REQUESTED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        result
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, new_size) };
        if !result.is_null() {
            ALLOCATION_REQUESTS.fetch_add(1, Ordering::Relaxed);
            REQUESTED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

// Historical implementation copied exactly from the pre-#27 training loop.
fn allocating_batch(train: &[Sample], cached: &[f32], rng: &mut Rng) -> (Vec<f32>, Vec<Sample>) {
    let mut batch = Vec::with_capacity(BATCH * INPUTS);
    let mut labels = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let index = rng.next_u32() as usize % train.len();
        batch.extend_from_slice(&cached[index * INPUTS..(index + 1) * INPUTS]);
        labels.push(Sample {
            pixels: vec![],
            class: train[index].class,
            box_xywh: train[index].box_xywh,
        });
    }
    (batch, labels)
}

fn verify_exact_batches(train: &[Sample], cached: &[f32]) -> Result<(), String> {
    let mut historical_rng = Rng::new(707);
    let mut reused_rng = Rng::new(707);
    let mut batch = Vec::with_capacity(BATCH * INPUTS);
    let mut labels = Vec::with_capacity(BATCH);
    for step in 1..=STEPS {
        let (expected_batch, expected_labels) =
            allocating_batch(train, cached, &mut historical_rng);
        fill_training_batch(train, cached, &mut reused_rng, &mut batch, &mut labels);
        let exact_features = expected_batch.len() == batch.len()
            && expected_batch
                .iter()
                .zip(&batch)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let exact_labels = expected_labels.len() == labels.len()
            && expected_labels.iter().zip(&labels).all(|(a, b)| {
                a.pixels.is_empty()
                    && b.pixels.is_empty()
                    && a.class == b.class
                    && a.box_xywh.map(f32::to_bits) == b.box_xywh.map(f32::to_bits)
            });
        if !exact_features || !exact_labels {
            return Err(format!(
                "historical and reused batches differ at step {step}"
            ));
        }
    }
    if historical_rng.next_u32() != reused_rng.next_u32() {
        return Err("batch sampling consumed a different RNG sequence".into());
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct Measurement {
    elapsed_ns: u128,
    allocation_requests: u64,
    requested_bytes: u64,
}

#[derive(Clone, Copy)]
enum Mode {
    Historical,
    Reused,
}

fn measure(mode: Mode, train: &[Sample], cached: &[f32]) -> Measurement {
    let mut rng = Rng::new(707);
    let before_requests = ALLOCATION_REQUESTS.load(Ordering::Relaxed);
    let before_bytes = REQUESTED_BYTES.load(Ordering::Relaxed);
    let started = Instant::now();
    match mode {
        Mode::Historical => {
            for _ in 0..STEPS {
                let (batch, labels) = allocating_batch(train, cached, &mut rng);
                black_box((&batch, &labels));
            }
        }
        Mode::Reused => {
            let mut batch = Vec::with_capacity(BATCH * INPUTS);
            let mut labels = Vec::with_capacity(BATCH);
            for _ in 0..STEPS {
                fill_training_batch(train, cached, &mut rng, &mut batch, &mut labels);
                black_box((&batch, &labels));
            }
        }
    }
    Measurement {
        elapsed_ns: started.elapsed().as_nanos(),
        allocation_requests: ALLOCATION_REQUESTS.load(Ordering::Relaxed) - before_requests,
        requested_bytes: REQUESTED_BYTES.load(Ordering::Relaxed) - before_bytes,
    }
}

fn run() -> Result<(), String> {
    if std::env::args().skip(1).collect::<Vec<_>>() != ["--measure"] {
        return Err("usage: fy-batch-benchmark --measure".into());
    }

    let train = dataset(101, 4_096);
    let mut cached = Vec::with_capacity(train.len() * INPUTS);
    for sample in &train {
        cached.extend(features(&sample.pixels)?);
    }
    // Validate every batch and RNG draw against the original algorithm
    // outside timed sections before recording any result.
    verify_exact_batches(&train, &cached)?;

    let mut historical = Vec::with_capacity(REPEATS);
    let mut reused = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        // Alternate order to reduce one-sided warm-up/thermal effects.
        if repeat % 2 == 0 {
            historical.push(measure(Mode::Historical, &train, &cached));
            reused.push(measure(Mode::Reused, &train, &cached));
        } else {
            reused.push(measure(Mode::Reused, &train, &cached));
            historical.push(measure(Mode::Historical, &train, &cached));
        }
    }
    println!(
        "{}",
        json!({
            "scope": "TD1 64-sample batch preparation only; no model forward/backward/optimizer",
            "source_commit": option_env!("FACTSYOLO_BUILD_ID").unwrap_or("unknown"),
            "build": "release",
            "training_seed": 101,
            "sampler_seed": 707,
            "training_samples": train.len(),
            "equivalent_steps": STEPS,
            "batch_size": BATCH,
            "repetitions": REPEATS,
            "bitwise_batch_and_rng_parity": true,
            "allocator_metric": "successful alloc/alloc_zeroed/realloc calls and requested bytes, not peak memory",
            "historical": historical,
            "reused": reused
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
