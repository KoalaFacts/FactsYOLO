// SPDX-License-Identifier: Apache-2.0
//! Paired Release-only live-allocation measurement for the PR #62 TD1 change.
//! Unlike RSS, allocator-requested live bytes can prove that the original
//! synthetic RGB Vec payload is released *after* feature construction.
//! Paired successful allocation requests/bytes also test the stack-backed
//! feature caching path against the preceding temporary-Vec path. Feature-build
//! peak, complete training, CPU use and wall-clock speedups are not inferred.
use fy_cpu_evaluate::training::cache_training_features_and_release_pixels;
use fy_tiny_model::{dataset, features, Sample, INPUTS};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

const SAMPLES: usize = 4_096;
const REPEATS: usize = 3;

// This benchmark makes no concurrent allocations. It tracks requested live
// payload sizes, not committed allocator pages, RSS or operating-system peaks.
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_REQUESTS: AtomicU64 = AtomicU64::new(0);
static REQUESTED_BYTES: AtomicU64 = AtomicU64::new(0);

struct LiveAllocator;

#[global_allocator]
static ALLOCATOR: LiveAllocator = LiveAllocator;

fn allocated(ptr: *mut u8, bytes: usize) {
    if !ptr.is_null() {
        ALLOCATION_REQUESTS.fetch_add(1, Ordering::Relaxed);
        REQUESTED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        let live = LIVE.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
        PEAK.fetch_max(live, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for LiveAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        allocated(ptr, layout.size());
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        allocated(ptr, layout.size());
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, size) };
        if !result.is_null() {
            LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
            allocated(result, size);
        }
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Retained,
    Released,
}

#[derive(Serialize)]
struct Measurement {
    elapsed_ns: u128,
    allocation_requests: u64,
    requested_bytes: u64,
    source_capacity_bytes: u64,
    live_after_cache_bytes: u64,
    peak_live_during_preparation_bytes: u64,
    feature_label_sha256: String,
}

fn historical_feature_cache(train: &[Sample]) -> Result<Vec<f32>, String> {
    // Exact preceding train path: cache features, retain every RGB Vec.
    let mut cached = Vec::with_capacity(train.len() * INPUTS);
    for sample in train {
        cached.extend(features(&sample.pixels)?);
    }
    Ok(cached)
}

fn measure(mode: Mode) -> Result<Measurement, String> {
    let before = LIVE.load(Ordering::Relaxed);
    let before_requests = ALLOCATION_REQUESTS.load(Ordering::Relaxed);
    let before_bytes = REQUESTED_BYTES.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let started = Instant::now();
    let mut train = dataset(101, SAMPLES);
    let source_capacity_bytes = train
        .iter()
        .map(|sample| sample.pixels.capacity() as u64)
        .sum::<u64>();
    let cached = match mode {
        Mode::Retained => historical_feature_cache(&train)?,
        Mode::Released => cache_training_features_and_release_pixels(&mut train)?,
    };
    if cached.len() != SAMPLES * INPUTS {
        return Err("incomplete source-cache benchmark feature buffer".into());
    }
    black_box((&train, &cached));
    // End measurement with both allocations still alive. Hashing, JSON
    // formatting and deallocation are outside the scoped clock/readings.
    let elapsed_ns = started.elapsed().as_nanos();
    let after = LIVE.load(Ordering::Relaxed);
    let peak = PEAK.load(Ordering::Relaxed);
    let allocation_requests = ALLOCATION_REQUESTS.load(Ordering::Relaxed)
        .checked_sub(before_requests)
        .ok_or("source-cache allocation request counter underflow")?;
    let requested_bytes = REQUESTED_BYTES.load(Ordering::Relaxed)
        .checked_sub(before_bytes)
        .ok_or("source-cache requested byte counter underflow")?;
    let live_after_cache_bytes = after
        .checked_sub(before)
        .ok_or("source-cache live allocation counter underflow")?;
    let peak_live_during_preparation_bytes = peak
        .checked_sub(before)
        .ok_or("source-cache peak allocation counter underflow")?;

    // Compare all feature bits, original source labels and box-coordinate
    // bits across modes and repeats, not merely aggregate model outcomes.
    let mut digest = Sha256::new();
    for sample in &train {
        digest.update([sample.class.map_or(255, |class| class as u8)]);
        for coordinate in sample.box_xywh {
            digest.update(coordinate.to_le_bytes());
        }
    }
    for value in &cached {
        digest.update(value.to_le_bytes());
    }
    let feature_label_sha256 = format!("{:x}", digest.finalize());
    drop(cached);
    drop(train);
    Ok(Measurement {
        elapsed_ns,
        allocation_requests,
        requested_bytes,
        source_capacity_bytes,
        live_after_cache_bytes,
        peak_live_during_preparation_bytes,
        feature_label_sha256,
    })
}

fn run() -> Result<(), String> {
    if std::env::args().skip(1).collect::<Vec<_>>() != ["--measure"] {
        return Err("usage: fy-source-cache-benchmark --measure".into());
    }
    if cfg!(debug_assertions) {
        return Err("source-cache measurement requires Release build".into());
    }
    let source = option_env!("FACTSYOLO_BUILD_ID").unwrap_or("unknown");
    if source == "unknown" {
        return Err("source commit must be supplied at compile time".into());
    }
    let mut runs = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        let (retained, released) = if repeat % 2 == 0 {
            (measure(Mode::Retained)?, measure(Mode::Released)?)
        } else {
            let released = measure(Mode::Released)?;
            let retained = measure(Mode::Retained)?;
            (retained, released)
        };
        let source_bytes = SAMPLES as u64 * 64 * 64 * 3;
        if retained.source_capacity_bytes != source_bytes
            || released.source_capacity_bytes != source_bytes
            || retained.feature_label_sha256 != released.feature_label_sha256
            || retained
                .live_after_cache_bytes
                .checked_sub(released.live_after_cache_bytes)
                != Some(source_bytes)
            || retained.allocation_requests <= released.allocation_requests
            || retained.requested_bytes <= released.requested_bytes
        {
            return Err("source-cache allocation or feature/label parity failed".into());
        }
        runs.push(json!({
            "first": if repeat % 2 == 0 { "retained" } else { "released" },
            "retained": retained,
            "released": released
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "fy-td1-source-cache-allocation-v2",
            "source_commit": source,
            "build": "release",
            "scope": "feature caching and source buffer residency only; excludes training",
            "allocator_metric": "System allocator live payload plus successful allocation requests/bytes, not RSS",
            "model_training_executed": false,
            "training_seed": 101,
            "training_samples": SAMPLES,
            "features_per_sample": INPUTS,
            "repetitions": REPEATS,
            "runs": runs
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
