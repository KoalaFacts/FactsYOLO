// SPDX-License-Identifier: Apache-2.0
//! Paired historical-vs-reused TD1 *complete training*, not just batch assembly.
//! The historical batch loop is the pre-#27 implementation. No model, RNG,
//! optimizer, validation dataset, or training budget is changed.
use fy_cpu_evaluate::{
    backend,
    metrics::{measure, Metrics},
    training::{adam_step, fill_training_batch},
    Backend,
};
use fy_tiny_model::{dataset, features, loss_gradient, Model, Rng, Sample, INPUTS, PARAMETERS};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

const STEPS: usize = 2_000;
const BATCH: usize = 64;
const REPEATS: usize = 3;
static REQUESTS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

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
fn record(ptr: *mut u8, size: usize) {
    if !ptr.is_null() {
        REQUESTS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

// Retain the exact previous allocation/sampling contract, including draw order.
fn historical_batch(train: &[Sample], cached: &[f32], rng: &mut Rng) -> (Vec<f32>, Vec<Sample>) {
    let mut inputs = Vec::with_capacity(BATCH * INPUTS);
    let mut labels = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let index = rng.next_u32() as usize % train.len();
        inputs.extend_from_slice(&cached[index * INPUTS..(index + 1) * INPUTS]);
        labels.push(Sample {
            pixels: Vec::new(),
            class: train[index].class,
            box_xywh: train[index].box_xywh,
        });
    }
    (inputs, labels)
}

#[derive(Clone, Copy)]
enum Mode {
    Historical,
    Reused,
}

#[derive(Serialize)]
struct Measurement {
    training_ns: u128,
    batch_preparation_ns: u128,
    allocation_requests: u64,
    requested_bytes: u64,
    model_sha256: String,
    sampler_next_draw: u32,
    last_batch_loss_bits: u32,
    validation: Metrics,
}

fn update(
    b: &dyn Backend,
    model: &mut Model,
    first: &mut [f32],
    second: &mut [f32],
    inputs: &[f32],
    labels: &[Sample],
    step: usize,
) -> Result<u32, String> {
    let output = b.forward(&model.weights, inputs, BATCH)?;
    let (loss, output_gradient) = loss_gradient(&output, labels)?;
    let gradient = b.backward(&model.weights, inputs, BATCH, &output_gradient)?;
    adam_step(&mut model.weights, &gradient, first, second, step)?;
    Ok(loss.to_bits())
}

fn train_one(
    b: &dyn Backend,
    mode: Mode,
    train: &[Sample],
    cached: &[f32],
    validation: &[Sample],
    validation_features: &[f32],
) -> Result<(Vec<f32>, Measurement), String> {
    let mut model = Model::initialized(404);
    let mut first = vec![0.0; PARAMETERS];
    let mut second = vec![0.0; PARAMETERS];
    let mut rng = Rng::new(707);
    let mut reused_inputs = Vec::with_capacity(BATCH * INPUTS);
    let mut reused_labels = Vec::with_capacity(BATCH);
    let mut preparation_ns = 0;
    let mut last_batch_loss_bits = 0;
    let before_requests = REQUESTS.load(Ordering::Relaxed);
    let before_bytes = BYTES.load(Ordering::Relaxed);
    let started = Instant::now();
    for step in 1..=STEPS {
        if started.elapsed().as_secs_f64() > 600.0 {
            return Err("full-training benchmark exceeded the registered 600s arm budget".into());
        }
        let batch_started = Instant::now();
        last_batch_loss_bits = match mode {
            Mode::Historical => {
                let (inputs, labels) = historical_batch(train, cached, &mut rng);
                preparation_ns += batch_started.elapsed().as_nanos();
                update(b, &mut model, &mut first, &mut second, &inputs, &labels, step)?
            }
            Mode::Reused => {
                fill_training_batch(
                    train,
                    cached,
                    &mut rng,
                    &mut reused_inputs,
                    &mut reused_labels,
                );
                preparation_ns += batch_started.elapsed().as_nanos();
                update(
                    b,
                    &mut model,
                    &mut first,
                    &mut second,
                    &reused_inputs,
                    &reused_labels,
                    step,
                )?
            }
        };
    }
    let training_ns = started.elapsed().as_nanos();
    let allocation_requests = REQUESTS.load(Ordering::Relaxed) - before_requests;
    let requested_bytes = BYTES.load(Ordering::Relaxed) - before_bytes;

    // Validation is deliberately outside both training and allocation intervals.
    // Never load the final seed909 holdout for this performance comparison.
    let output = b.forward(&model.weights, validation_features, validation.len())?;
    let validation = measure(&output, validation)?;
    if !validation.passes() {
        return Err("full-training benchmark failed the unchanged validation gate".into());
    }
    let mut digest = Sha256::new();
    for weight in &model.weights {
        digest.update(weight.to_le_bytes());
    }
    let measurement = Measurement {
        training_ns,
        batch_preparation_ns: preparation_ns,
        allocation_requests,
        requested_bytes,
        model_sha256: format!("{:x}", digest.finalize()),
        sampler_next_draw: rng.next_u32(),
        last_batch_loss_bits,
        validation,
    };
    Ok((model.weights, measurement))
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 || args[1] != "--measure" {
        return Err("usage: fy-full-training-benchmark <burn|candle> --measure".into());
    }
    let source = option_env!("FACTSYOLO_BUILD_ID").unwrap_or("unknown");
    if source == "unknown" {
        return Err("source commit must be supplied at compile time".into());
    }
    let b = backend(&args[0])?;
    let train = dataset(101, 4_096);
    let validation = dataset(202, 128);
    let mut cached = Vec::with_capacity(train.len() * INPUTS);
    for sample in &train {
        cached.extend(features(&sample.pixels)?);
    }
    let mut validation_features = Vec::with_capacity(validation.len() * INPUTS);
    for sample in &validation {
        validation_features.extend(features(&sample.pixels)?);
    }

    let mut expected_hash = None;
    let mut runs = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        let first = if repeat % 2 == 0 { "historical" } else { "reused" };
        let do_train = |mode| train_one(&*b, mode, &train, &cached, &validation, &validation_features);
        let (old_weights, historical, new_weights, reused) = if repeat % 2 == 0 {
            let (old_weights, historical) = do_train(Mode::Historical)?;
            let (new_weights, reused) = do_train(Mode::Reused)?;
            (old_weights, historical, new_weights, reused)
        } else {
            let (new_weights, reused) = do_train(Mode::Reused)?;
            let (old_weights, historical) = do_train(Mode::Historical)?;
            (old_weights, historical, new_weights, reused)
        };
        if old_weights.len() != new_weights.len()
            || !old_weights.iter().zip(&new_weights).all(|(a, b)| a.to_bits() == b.to_bits())
            || historical.sampler_next_draw != reused.sampler_next_draw
            || historical.last_batch_loss_bits != reused.last_batch_loss_bits
            || serde_json::to_value(&historical.validation).map_err(|e| e.to_string())?
                != serde_json::to_value(&reused.validation).map_err(|e| e.to_string())?
            || historical.model_sha256 != reused.model_sha256
        {
            return Err(format!("historical/reused complete-training parity failed at pair {repeat}"));
        }
        if let Some(hash) = &expected_hash {
            if &historical.model_sha256 != hash {
                return Err("full-training repeated seed produced different trained weights".into());
            }
        } else {
            expected_hash = Some(historical.model_sha256.clone());
        }
        runs.push(json!({"first": first, "historical": historical, "reused": reused}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "scope": "full TD1 Burn/Candle training (2000 optimizer steps) in one process",
            "source_commit": source,
            "backend": b.name(),
            "build": "release",
            "os": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "rayon_threads": std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
            "training_seed": 101,
            "initialization_seed": 404,
            "sampler_seed": 707,
            "steps_per_arm": STEPS,
            "batch_size": BATCH,
            "training_samples": train.len(),
            "repetitions": REPEATS,
            "data_preparation_excluded": true,
            "evaluation_excluded_from_timing": true,
            "exact_weight_rng_loss_validation_parity": true,
            "resource_metric": "successful process allocation requests and requested bytes; not peak RSS",
            "timing_scope": "complete forward/loss/backward/Adam + batch assembly; excludes data generation, validation and builds",
            "runs": runs
        }))
        .map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
