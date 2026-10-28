// SPDX-License-Identifier: Apache-2.0
//! Experimental TD1 training intervention: fixed-frequency edge-positive sampling.
//! No final holdout, changed model format, native ABI, or production inference path.
use fy_cpu_evaluate::{
    backend,
    metrics::{iou, measure, target_corners, Metrics},
    training::{adam_step, fill_training_batch},
    Backend,
};
use fy_tiny_model::{dataset, features, loss_gradient, Model, Rng, Sample, INPUTS, PARAMETERS};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::time::Instant;

const STEPS: usize = 2000;
const BATCH: usize = 64;
const EDGE_FREQUENCY: usize = 16;
const QUOTA: usize = 48;
const GROUPS: [&str; 5] = [
    "small_edge",
    "small_interior",
    "larger_edge",
    "larger_interior",
    "background",
];
const FROZEN_DEVELOPMENT_FINGERPRINT: &str =
    "04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4";
// Independently frozen against the untrained generator, before model predictions.
const PROSPECTIVE_SEED: u64 = 20261101;
const PROSPECTIVE_CANDIDATES: usize = 8192;
const PROSPECTIVE_FINGERPRINT: &str =
    "0239bac3429d484db479974c10e5a038cef8829762a7d5533f8cd915e3a6ede1";

fn stratum(sample: &Sample) -> usize {
    if sample.class.is_none() {
        return 4;
    }
    let [x1, y1, x2, y2] = target_corners(sample);
    let small = (x2 - x1) * (y2 - y1) <= 256.0;
    let edge = x1.min(y1).min(64.0 - x2).min(64.0 - y2) <= 2.0;
    match (small, edge) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

fn edge_pool(train: &[Sample]) -> Result<Vec<usize>, String> {
    let pool: Vec<usize> = train
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| (stratum(sample) == 0).then_some(index))
        .collect();
    if pool.is_empty() {
        return Err("training source has no small edge positives".into());
    }
    Ok(pool)
}

// Exactly one sampler draw per slot, just like the unchanged baseline.
// Four of 64 positions select a source index from the precomputed edge pool;
// all other positions use the historical modulo operation and RNG stream.
fn fill_edge_batch(
    train: &[Sample],
    cached: &[f32],
    pool: &[usize],
    rng: &mut Rng,
    inputs: &mut Vec<f32>,
    labels: &mut Vec<Sample>,
) {
    inputs.clear();
    labels.clear();
    for slot in 0..BATCH {
        let draw = rng.next_u32() as usize;
        let index = if slot % EDGE_FREQUENCY == 0 {
            pool[draw % pool.len()]
        } else {
            draw % train.len()
        };
        inputs.extend_from_slice(&cached[index * INPUTS..(index + 1) * INPUTS]);
        labels.push(Sample {
            pixels: Vec::new(),
            class: train[index].class,
            box_xywh: train[index].box_xywh,
        });
    }
}

fn train_model(
    b: &dyn Backend,
    train: &[Sample],
    cached: &[f32],
    pool: &[usize],
    boosted: bool,
) -> Result<(Model, f64), String> {
    let mut model = Model::initialized(404);
    let mut first = vec![0.0; PARAMETERS];
    let mut second = vec![0.0; PARAMETERS];
    let mut rng = Rng::new(707);
    let mut inputs = Vec::with_capacity(BATCH * INPUTS);
    let mut labels = Vec::with_capacity(BATCH);
    let started = Instant::now();

    for step in 1..=STEPS {
        if started.elapsed().as_secs_f64() > 600.0 {
            return Err("controlled training exceeded the registered 600s budget".into());
        }
        if boosted {
            fill_edge_batch(train, cached, pool, &mut rng, &mut inputs, &mut labels);
        } else {
            fill_training_batch(train, cached, &mut rng, &mut inputs, &mut labels);
        }
        let predictions = b.forward(&model.weights, &inputs, BATCH)?;
        let (_, output_gradient) = loss_gradient(&predictions, &labels)?;
        let gradient = b.backward(&model.weights, &inputs, BATCH, &output_gradient)?;
        adam_step(&mut model.weights, &gradient, &mut first, &mut second, step)?;
    }
    if model.weights == Model::initialized(404).weights {
        return Err("training left initial weights unchanged".into());
    }
    Ok((model, started.elapsed().as_secs_f64()))
}

fn registered_split(
    seed: u64,
    candidates: usize,
    expected_fingerprint: &str,
) -> Result<Vec<Sample>, String> {
    let mut counts = [0_usize; 5];
    let mut selected = Vec::with_capacity(QUOTA * GROUPS.len());
    for sample in dataset(seed, candidates) {
        let group = stratum(&sample);
        if counts[group] < QUOTA {
            selected.push(sample);
            counts[group] += 1;
        }
        if counts.iter().all(|&n| n == QUOTA) {
            break;
        }
    }
    if counts.iter().any(|&n| n != QUOTA) {
        return Err("registered development/prospective split incomplete".into());
    }

    let mut digest = Sha256::new();
    digest.update(b"TD1-synthetic-v1");
    for sample in &selected {
        digest.update([sample.class.map_or(255, |class| class as u8)]);
        for coordinate in sample.box_xywh {
            digest.update(coordinate.to_le_bytes());
        }
        digest.update(&sample.pixels);
    }
    if format!("{:x}", digest.finalize()) != expected_fingerprint {
        return Err("registered development/prospective fingerprint drift".into());
    }
    Ok(selected)
}

fn frozen_development() -> Result<Vec<Sample>, String> {
    registered_split(20261030, 4096, FROZEN_DEVELOPMENT_FINGERPRINT)
}

fn frozen_prospective() -> Result<Vec<Sample>, String> {
    registered_split(
        PROSPECTIVE_SEED,
        PROSPECTIVE_CANDIDATES,
        PROSPECTIVE_FINGERPRINT,
    )
}

#[derive(Default)]
struct Slice {
    count: usize,
    correct_class_iou50: usize,
    sum_iou: f32,
    missing: usize,
    wrong_class: usize,
    false_positives: usize,
}

fn evaluate(
    b: &dyn Backend,
    model: &Model,
    development: &[Sample],
    validation: &[Sample],
    validation_features: &[f32],
) -> Result<Value, String> {
    let outputs = b.forward(&model.weights, validation_features, validation.len())?;
    let validation_metrics: Metrics = measure(&outputs, validation)?;
    let mut groups: [Slice; 5] = std::array::from_fn(|_| Slice::default());
    for sample in development {
        let group = stratum(sample);
        let record = &mut groups[group];
        record.count += 1;
        let detected = model.detect(&sample.pixels)?;
        if sample.class.is_none() {
            record.false_positives += usize::from(detected.is_some());
            continue;
        }
        match detected {
            None => record.missing += 1,
            Some(detection) if Some(detection.class) != sample.class => {
                record.wrong_class += 1;
            }
            Some(detection) => {
                let overlap = iou(detection.corners, target_corners(sample));
                record.sum_iou += overlap;
                record.correct_class_iou50 += usize::from(overlap >= 0.5);
            }
        }
    }
    let mut grouped = Map::new();
    for (name, record) in GROUPS.into_iter().zip(groups) {
        if record.count != QUOTA {
            return Err(format!("invalid development stratum count: {name}"));
        }
        let value = if name == "background" {
            json!({"samples": record.count, "false_positives": record.false_positives})
        } else {
            json!({
                "samples": record.count,
                "correct_class_iou50": record.correct_class_iou50,
                "recall_correct_class_iou50": record.correct_class_iou50 as f64 / record.count as f64,
                "mean_iou_including_misses": record.sum_iou / record.count as f32,
                "missing_detections": record.missing,
                "wrong_class_detections": record.wrong_class
            })
        };
        grouped.insert(name.to_string(), value);
    }
    Ok(json!({
        "validation": validation_metrics,
        "validation_passes_original_thresholds": validation_metrics.passes(),
        "development": grouped
    }))
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() != 2 {
        return Err("usage: fy-edge-training <burn|candle>".into());
    }
    let source = option_env!("FACTSYOLO_BUILD_ID").unwrap_or("unknown");
    if source == "unknown" {
        return Err("source commit must be supplied at compile time".into());
    }
    let b = backend(&arguments[1])?;
    let train = dataset(101, 4096);
    let pool = edge_pool(&train)?;
    let mut cached = Vec::with_capacity(train.len() * INPUTS);
    for sample in &train {
        cached.extend(features(&sample.pixels)?);
    }
    let development = frozen_development()?;
    let prospective = frozen_prospective()?;
    let validation = dataset(202, 128);
    let mut validation_features = Vec::with_capacity(validation.len() * INPUTS);
    for sample in &validation {
        validation_features.extend(features(&sample.pixels)?);
    }

    let (baseline_model, baseline_seconds) = train_model(&*b, &train, &cached, &pool, false)?;
    let baseline = evaluate(
        &*b,
        &baseline_model,
        &development,
        &validation,
        &validation_features,
    )?;
    let baseline_prospective = evaluate(
        &*b,
        &baseline_model,
        &prospective,
        &validation,
        &validation_features,
    )?;
    let (boosted_model, boosted_seconds) = train_model(&*b, &train, &cached, &pool, true)?;
    let boosted = evaluate(
        &*b,
        &boosted_model,
        &development,
        &validation,
        &validation_features,
    )?;
    let boosted_prospective = evaluate(
        &*b,
        &boosted_model,
        &prospective,
        &validation,
        &validation_features,
    )?;
    if baseline["validation_passes_original_thresholds"] != true {
        return Err("baseline failed the unchanged validation thresholds".into());
    }
    let recall = |run: &Value, group: &str| -> Result<f64, String> {
        run["development"][group]["recall_correct_class_iou50"]
            .as_f64()
            .ok_or_else(|| format!("missing development recall for {group}"))
    };
    let report = json!({
        "status": "controlled-training-experiment-only",
        "source_commit": source,
        "backend": b.name(),
        "model": "TD1-v1-192-64-7",
        "training_seed": 101,
        "validation_seed": 202,
        "initialization_seed": 404,
        "sampler_seed": 707,
        "steps_per_arm": STEPS,
        "batch_size": BATCH,
        "train_samples": train.len(),
        "edge_pool_count": pool.len(),
        "edge_override_slots_per_batch": BATCH / EDGE_FREQUENCY,
        "development_seed": 20261030,
        "development_samples": development.len(),
        "development_fingerprint": FROZEN_DEVELOPMENT_FINGERPRINT,
        "prospective_seed": PROSPECTIVE_SEED,
        "prospective_candidate_budget": PROSPECTIVE_CANDIDATES,
        "prospective_samples": prospective.len(),
        "prospective_fingerprint": PROSPECTIVE_FINGERPRINT,
        "final_holdout_evaluated": false,
        "baseline_seconds": baseline_seconds,
        "edge_sampling_seconds": boosted_seconds,
        "baseline": baseline,
        "edge_sampling": boosted,
        "baseline_prospective": baseline_prospective["development"],
        "edge_sampling_prospective": boosted_prospective["development"],
        "prospective_edge_minus_baseline_small_edge_iou50_recall":
            recall(&boosted_prospective, "small_edge")? - recall(&baseline_prospective, "small_edge")?,
        "prospective_edge_minus_baseline_small_interior_iou50_recall":
            recall(&boosted_prospective, "small_interior")? - recall(&baseline_prospective, "small_interior")?,
        "edge_minus_baseline_small_edge_iou50_recall":
            recall(&boosted, "small_edge")? - recall(&baseline, "small_edge")?,
        "edge_minus_baseline_small_interior_iou50_recall":
            recall(&boosted, "small_interior")? - recall(&baseline, "small_interior")?
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_selection_is_ground_truth_only_and_nonempty() {
        let train = dataset(101, 256);
        let pool = edge_pool(&train).unwrap();
        assert!(pool.iter().all(|&index| stratum(&train[index]) == 0));
        assert!(!pool.is_empty());
    }

    #[test]
    fn boosted_sampler_preserves_rng_and_every_non_reserved_draw() {
        let train = dataset(101, 256);
        let pool = edge_pool(&train).unwrap();
        let mut cached = Vec::new();
        for sample in &train {
            cached.extend(features(&sample.pixels).unwrap());
        }
        let mut uniform_rng = Rng::new(707);
        let mut boosted_rng = Rng::new(707);
        let (mut uniform_input, mut boosted_input) = (Vec::new(), Vec::new());
        let (mut uniform_labels, mut boosted_labels) = (Vec::new(), Vec::new());
        let mut changed = false;

        for _ in 0..5 {
            fill_training_batch(
                &train,
                &cached,
                &mut uniform_rng,
                &mut uniform_input,
                &mut uniform_labels,
            );
            fill_edge_batch(
                &train,
                &cached,
                &pool,
                &mut boosted_rng,
                &mut boosted_input,
                &mut boosted_labels,
            );
            for slot in 0..BATCH {
                if slot % EDGE_FREQUENCY == 0 {
                    assert!(pool.iter().any(|&index| {
                        boosted_labels[slot].class == train[index].class
                            && boosted_labels[slot].box_xywh == train[index].box_xywh
                            && boosted_input[slot * INPUTS..(slot + 1) * INPUTS]
                                == cached[index * INPUTS..(index + 1) * INPUTS]
                    }));
                    changed |= uniform_input[slot * INPUTS..(slot + 1) * INPUTS]
                        != boosted_input[slot * INPUTS..(slot + 1) * INPUTS];
                } else {
                    assert_eq!(
                        uniform_input[slot * INPUTS..(slot + 1) * INPUTS],
                        boosted_input[slot * INPUTS..(slot + 1) * INPUTS]
                    );
                    assert_eq!(uniform_labels[slot].class, boosted_labels[slot].class);
                    assert_eq!(uniform_labels[slot].box_xywh, boosted_labels[slot].box_xywh);
                }
            }
            assert_eq!(uniform_rng.next_u32(), boosted_rng.next_u32());
        }
        assert!(
            changed,
            "edge intervention did not change any reserved draw"
        );
    }

    #[test]
    fn frozen_development_identity_is_rechecked_before_training_comparison() {
        assert_eq!(frozen_development().unwrap().len(), QUOTA * GROUPS.len());
    }

    #[test]
    fn prospective_fingerprint_and_separation_are_frozen_before_inference() {
        let prospective = frozen_prospective().unwrap();
        let earlier = frozen_development().unwrap();
        assert_eq!(prospective.len(), QUOTA * GROUPS.len());
        let historical: std::collections::HashSet<&[u8]> =
            earlier.iter().map(|sample| sample.pixels.as_slice()).collect();
        assert!(prospective
            .iter()
            .all(|sample| !historical.contains(sample.pixels.as_slice())));
        for group in 0..GROUPS.len() {
            assert_eq!(
                prospective.iter().filter(|sample| stratum(sample) == group).count(),
                QUOTA
            );
        }
    }
}
