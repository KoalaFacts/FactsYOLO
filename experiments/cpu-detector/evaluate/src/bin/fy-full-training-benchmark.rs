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
    io::{self, Write},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

const STEPS: usize = 2_000;
const BATCH: usize = 64;
// Four historical-first and four reused-first pairs reduce fixed-order bias.
// Timing remains exploratory and never gates on a claimed speedup.
const REPEATS: usize = 8;
const ARM_BUDGET: Duration = Duration::from_secs(600);

fn over_arm_budget(elapsed: Duration) -> bool {
    elapsed > ARM_BUDGET
}

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

// Write each event immediately and flush it before running another arm.
// On any later failure the separate JSONL diagnostic keeps completed measurements,
// while stdout remains reserved for a single fully validated successful report.
fn emit_progress(event: &serde_json::Value) -> Result<(), String> {
    let mut stderr = io::stderr().lock();
    serde_json::to_writer(&mut stderr, event).map_err(|e| e.to_string())?;
    stderr.write_all(b"\n").map_err(|e| e.to_string())?;
    stderr.flush().map_err(|e| e.to_string())
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
    // Include the reuse buffers' one-time allocation in both the full-training
    // clock and requested-allocation counters. Previously these buffers were
    // allocated before measurement, understating the reused arm's resources.
    let mut reused_inputs = Vec::new();
    let mut reused_labels = Vec::new();
    let mut preparation_ns = 0;
    let mut last_batch_loss_bits = 0;
    let before_requests = REQUESTS.load(Ordering::Relaxed);
    let before_bytes = BYTES.load(Ordering::Relaxed);
    let started = Instant::now();
    if matches!(mode, Mode::Reused) {
        let setup_started = Instant::now();
        reused_inputs.reserve_exact(BATCH * INPUTS);
        reused_labels.reserve_exact(BATCH);
        preparation_ns += setup_started.elapsed().as_nanos();
    }
    for step in 1..=STEPS {
        if over_arm_budget(started.elapsed()) {
            return Err("full-training benchmark exceeded the registered 600s arm budget".into());
        }
        let batch_started = Instant::now();
        last_batch_loss_bits = match mode {
            Mode::Historical => {
                let (inputs, labels) = historical_batch(train, cached, &mut rng);
                preparation_ns += batch_started.elapsed().as_nanos();
                update(
                    b,
                    &mut model,
                    &mut first,
                    &mut second,
                    &inputs,
                    &labels,
                    step,
                )?
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
    let elapsed = started.elapsed();
    // A slow final optimizer step can cross 600s with no next iteration to check.
    if over_arm_budget(elapsed) {
        return Err("full-training benchmark exceeded the registered 600s arm budget".into());
    }
    let training_ns = elapsed.as_nanos();
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
        let first = if repeat % 2 == 0 {
            "historical"
        } else {
            "reused"
        };
        let do_train = |mode: Mode, arm: &str| -> Result<(Vec<f32>, Measurement), String> {
            emit_progress(&json!({
                "schema": "fy-full-training-progress-v1", "source_commit": source,
                "backend": b.name(), "pair": repeat, "arm": arm, "event": "started"
            }))?;
            // A test-only controlled error after the first *real* completed arm
            // proves that flushed measurements survive a subsequent failure.
            // This is outside measured training and has no effect unless opted in.
            let result = if repeat == 0
                && arm == "reused"
                && std::env::var("FACTSYOLO_TEST_FAIL_SECOND_ARM").as_deref() == Ok("1")
            {
                Err("controlled benchmark failure after one completed arm".to_string())
            } else {
                train_one(
                    &*b,
                    mode,
                    &train,
                    &cached,
                    &validation,
                    &validation_features,
                )
            };
            match result {
                Ok((weights, measurement)) => {
                    emit_progress(&json!({
                        "schema": "fy-full-training-progress-v1", "source_commit": source,
                        "backend": b.name(), "pair": repeat, "arm": arm,
                        "event": "completed", "measurement": &measurement
                    }))?;
                    Ok((weights, measurement))
                }
                Err(error) => {
                    emit_progress(&json!({
                        "schema": "fy-full-training-progress-v1", "source_commit": source,
                        "backend": b.name(), "pair": repeat, "arm": arm,
                        "event": "failed", "error": &error
                    }))?;
                    Err(error)
                }
            }
        };
        let (old_weights, historical, new_weights, reused) = if repeat % 2 == 0 {
            let (old_weights, historical) = do_train(Mode::Historical, "historical")?;
            let (new_weights, reused) = do_train(Mode::Reused, "reused")?;
            (old_weights, historical, new_weights, reused)
        } else {
            let (new_weights, reused) = do_train(Mode::Reused, "reused")?;
            let (old_weights, historical) = do_train(Mode::Historical, "historical")?;
            (old_weights, historical, new_weights, reused)
        };
        if old_weights.len() != new_weights.len()
            || !old_weights
                .iter()
                .zip(&new_weights)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            || historical.sampler_next_draw != reused.sampler_next_draw
            || historical.last_batch_loss_bits != reused.last_batch_loss_bits
            || serde_json::to_value(&historical.validation).map_err(|e| e.to_string())?
                != serde_json::to_value(&reused.validation).map_err(|e| e.to_string())?
            || historical.model_sha256 != reused.model_sha256
        {
            let error =
                format!("historical/reused complete-training parity failed at pair {repeat}");
            emit_progress(&json!({
                "schema": "fy-full-training-progress-v1", "source_commit": source,
                "backend": b.name(), "pair": repeat, "arm": "pair",
                "event": "failed", "error": &error
            }))?;
            return Err(error);
        }
        if let Some(hash) = &expected_hash {
            if &historical.model_sha256 != hash {
                let error = "full-training repeated seed produced different trained weights";
                emit_progress(&json!({
                    "schema": "fy-full-training-progress-v1", "source_commit": source,
                    "backend": b.name(), "pair": repeat, "arm": "pair",
                    "event": "failed", "error": error
                }))?;
                return Err(error.into());
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

#[cfg(test)]
mod tests {
    use super::{over_arm_budget, ARM_BUDGET};
    use std::time::Duration;

    #[test]
    fn registered_arm_budget_is_inclusive() {
        assert!(!over_arm_budget(ARM_BUDGET));
        assert!(over_arm_budget(ARM_BUDGET + Duration::from_nanos(1)));
    }
}

fn main() {
    if let Err(error) = run() {
        // Preserve a machine-readable JSONL failure record after the already
        // flushed per-arm diagnostics, without printing a success report.
        if emit_progress(&json!({
            "schema": "fy-full-training-progress-v1",
            "event": "process_failed", "error": &error
        }))
        .is_err()
        {
            eprintln!("error: {error}");
        }
        std::process::exit(1);
    }
}
