// SPDX-License-Identifier: Apache-2.0
//! Experimental TD1 training intervention: fixed-frequency edge-positive sampling.
//! No final holdout, changed model format, native ABI, or production inference path.

// The wide experimental evidence JSON objects exceed serde_json::json!'s default
// macro expansion depth; this changes compilation only, not the report schema.
#![recursion_limit = "256"]
use fy_cpu_evaluate::{
    backend,
    metrics::{iou, measure, target_corners, Metrics},
    training::{adam_step, fill_training_batch, mirror_reserved_edge_batch},
    Backend,
};
use fy_cpu_evaluate::training::illuminate_training_batch;
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
const PROSPECTIVE_CANDIDATES: usize = 4096;
const PROSPECTIVE_FINGERPRINT: &str =
    "0239bac3429d484db479974c10e5a038cef8829762a7d5533f8cd915e3a6ede1";

const FOUR_SIDE_SEED: u64 = 20261113;
const FOUR_SIDE_CANDIDATES: usize = 4096;
const FOUR_SIDE_FINGERPRINT: &str =
    "59a4a885ff0ca2c5c8e700f8ccaaefbdaf454f2f2d0e6a72d280026b8398b3bd";
const DISJOINT_SEED: u64 = 20261123;
const DISJOINT_CANDIDATES: usize = 8192;
const DISJOINT_FINGERPRINT: &str =
    "e156306394733b94c5d99daec4ae22647819cab6976bc3dd55938eb988500a56";
const DISJOINT_NAMES: [&str; 5] = ["left", "top", "right", "bottom", "background"];

const FOUR_SIDE_NAMES: [&str; 4] = ["left", "top", "right", "bottom"];

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
    mirrored: bool,
    textured: bool,
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
            if mirrored {
                mirror_reserved_edge_batch(&mut inputs, &mut labels)?;
            }
        } else {
            fill_training_batch(train, cached, &mut rng, &mut inputs, &mut labels);
        }
        if textured {
            illuminate_training_batch(&mut inputs, &labels, step)?;
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

fn reflect_four_side_sample(sample: &Sample, horizontal: bool, vertical: bool) -> Sample {
    let mut pixels = vec![0_u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let sx = if horizontal { 63 - x } else { x };
            let sy = if vertical { 63 - y } else { y };
            let source = (sy * 64 + sx) * 3;
            let destination = (y * 64 + x) * 3;
            pixels[destination..destination + 3]
                .copy_from_slice(&sample.pixels[source..source + 3]);
        }
    }
    let mut box_xywh = sample.box_xywh;
    if horizontal {
        box_xywh[0] = 1.0 - box_xywh[0];
    }
    if vertical {
        box_xywh[1] = 1.0 - box_xywh[1];
    }
    Sample {
        pixels,
        class: sample.class,
        box_xywh,
    }
}

/// New source images are chosen by geometry only, before either backend's
/// inference. The four views of each source image are correlated, *not*
/// four independent test sets. The seed909 final holdout is never read.
fn frozen_four_sides() -> Result<[Vec<Sample>; 4], String> {
    let mut groups: [Vec<Sample>; 4] = std::array::from_fn(|_| Vec::with_capacity(48));
    let (mut left_sources, mut top_sources) = (0_usize, 0_usize);

    for sample in dataset(FOUR_SIDE_SEED, FOUR_SIDE_CANDIDATES) {
        if sample.class.is_none() {
            continue;
        }
        let [x1, y1, x2, y2] = target_corners(&sample);
        if (x2 - x1) * (y2 - y1) > 256.0 {
            continue;
        }
        let left = x1 <= 2.0;
        let top = y1 <= 2.0;
        let right = 64.0 - x2 <= 2.0;
        let bottom = 64.0 - y2 <= 2.0;
        let origin = if left && !top && !right && !bottom && left_sources < 24 {
            left_sources += 1;
            Some(0)
        } else if top && !left && !right && !bottom && top_sources < 24 {
            top_sources += 1;
            Some(1)
        } else {
            None
        };
        if let Some(origin) = origin {
            for (horizontal, vertical) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let side = if origin == 0 {
                    if horizontal {
                        2
                    } else {
                        0
                    }
                } else if vertical {
                    3
                } else {
                    1
                };
                groups[side].push(reflect_four_side_sample(&sample, horizontal, vertical));
            }
        }
        if left_sources == 24 && top_sources == 24 {
            break;
        }
    }

    if left_sources != 24 || top_sources != 24 || groups.iter().any(|rows| rows.len() != 48) {
        return Err("frozen four-side source selection incomplete".into());
    }
    let mut digest = Sha256::new();
    digest.update(b"TD1-four-side-paired-v1");
    for rows in &groups {
        for row in rows {
            digest.update([row.class.map_or(255, |class| class as u8)]);
            for coordinate in row.box_xywh {
                digest.update(coordinate.to_le_bytes());
            }
            digest.update(&row.pixels);
        }
    }
    if format!("{:x}", digest.finalize()) != FOUR_SIDE_FINGERPRINT {
        return Err("frozen four-side source/geometry/pixel fingerprint drift".into());
    }
    Ok(groups)
}

/// A different frozen seed and 144 distinct source images: no positive source
/// is evaluated on more than one side. Right and bottom images are reflected
/// from *different* original samples, not the left/top evaluation examples.
/// The final seed909 holdout is never loaded or used for training selection.
fn frozen_disjoint_four_sides() -> Result<[Vec<Sample>; 5], String> {
    let mut groups: [Vec<Sample>; 5] = std::array::from_fn(|_| Vec::with_capacity(48));
    let (mut left_sources, mut top_sources) = (0_usize, 0_usize);

    for sample in dataset(DISJOINT_SEED, DISJOINT_CANDIDATES) {
        if sample.class.is_none() {
            if groups[4].len() < 48 {
                groups[4].push(sample);
            }
            continue;
        }
        let [x1, y1, x2, y2] = target_corners(&sample);
        if (x2 - x1) * (y2 - y1) > 256.0 {
            continue;
        }
        let left = x1 <= 2.0;
        let top = y1 <= 2.0;
        let right = 64.0 - x2 <= 2.0;
        let bottom = 64.0 - y2 <= 2.0;
        if left && !top && !right && !bottom && left_sources < 48 {
            let side = if left_sources < 24 { 0 } else { 2 };
            left_sources += 1;
            groups[side].push(if side == 2 {
                reflect_four_side_sample(&sample, true, false)
            } else {
                sample
            });
        } else if top && !left && !right && !bottom && top_sources < 48 {
            let side = if top_sources < 24 { 1 } else { 3 };
            top_sources += 1;
            groups[side].push(if side == 3 {
                reflect_four_side_sample(&sample, false, true)
            } else {
                sample
            });
        }
        if left_sources == 48 && top_sources == 48 && groups[4].len() == 48 {
            break;
        }
    }

    if left_sources != 48
        || top_sources != 48
        || groups[..4].iter().any(|rows| rows.len() != 24)
        || groups[4].len() != 48
    {
        return Err("frozen disjoint four-side source selection incomplete".into());
    }
    let mut digest = Sha256::new();
    digest.update(b"TD1-four-side-disjoint-v1");
    for rows in &groups {
        for row in rows {
            digest.update([row.class.map_or(255, |class| class as u8)]);
            for coordinate in row.box_xywh {
                digest.update(coordinate.to_le_bytes());
            }
            digest.update(&row.pixels);
        }
    }
    if format!("{:x}", digest.finalize()) != DISJOINT_FINGERPRINT {
        return Err("frozen disjoint four-side source/pixel fingerprint drift".into());
    }
    Ok(groups)
}

fn evaluate_disjoint_four_sides(model: &Model, groups: &[Vec<Sample>; 5]) -> Result<Value, String> {
    let mut result = Map::new();
    for (side, samples) in DISJOINT_NAMES.into_iter().zip(groups) {
        let mut record = Slice::default();
        for sample in samples {
            record.count += 1;
            let detection = model.detect(&sample.pixels)?;
            if side == "background" {
                if sample.class.is_some() {
                    return Err("disjoint background sample has a target".into());
                }
                record.false_positives += usize::from(detection.is_some());
                continue;
            }
            if sample.class.is_none() {
                return Err("disjoint four-side positive sample lacks a target".into());
            }
            match detection {
                None => record.missing += 1,
                Some(found) if Some(found.class) != sample.class => record.wrong_class += 1,
                Some(found) => {
                    let overlap = iou(found.corners, target_corners(sample));
                    record.sum_iou += overlap;
                    record.correct_class_iou50 += usize::from(overlap >= 0.5);
                }
            }
        }
        let quota = if side == "background" { 48 } else { 24 };
        if record.count != quota {
            return Err(format!("incomplete frozen disjoint group: {side}"));
        }
        let row = if side == "background" {
            json!({"samples": record.count, "false_positives": record.false_positives})
        } else {
            json!({
                "samples": record.count,
                "correct_class_iou50": record.correct_class_iou50,
                "recall_correct_class_iou50":
                    record.correct_class_iou50 as f64 / record.count as f64,
                "mean_iou_including_misses": record.sum_iou / record.count as f32,
                "missing_detections": record.missing,
                "wrong_class_detections": record.wrong_class
            })
        };
        result.insert(side.to_string(), row);
    }
    Ok(Value::Object(result))
}

fn evaluate_four_sides(model: &Model, groups: &[Vec<Sample>; 4]) -> Result<Value, String> {
    let mut output = Map::new();
    for (name, samples) in FOUR_SIDE_NAMES.into_iter().zip(groups) {
        let mut record = Slice::default();
        for sample in samples {
            record.count += 1;
            match model.detect(&sample.pixels)? {
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
        if record.count != 48 {
            return Err(format!("incomplete frozen four-side group: {name}"));
        }
        output.insert(
            name.into(),
            json!({
                "samples": record.count,
                "correct_class_iou50": record.correct_class_iou50,
                "recall_correct_class_iou50":
                    record.correct_class_iou50 as f64 / record.count as f64,
                "mean_iou_including_misses": record.sum_iou / record.count as f32,
                "missing_detections": record.missing,
                "wrong_class_detections": record.wrong_class
            }),
        );
    }
    Ok(Value::Object(output))
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
    let mirror_only = arguments.len() == 3 && arguments[2] == "--mirror";
    let texture_only = arguments.len() == 3 && arguments[2] == "--texture";
    if arguments.len() != 2 && !mirror_only && !texture_only {
        return Err("usage: fy-edge-training <burn|candle> [--mirror|--texture]".into());
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
    let four_sides = frozen_four_sides()?;
    let disjoint_sides = frozen_disjoint_four_sides()?;
    let validation = dataset(202, 128);
    let mut validation_features = Vec::with_capacity(validation.len() * INPUTS);
    for sample in &validation {
        validation_features.extend(features(&sample.pixels)?);
    }

    // A separately invoked, opt-in third training arm. The established two-arm
    // comparison remains unchanged; this branch produces no model artifact and
    // never reads the final seed909 holdout.
    if mirror_only || texture_only {
        let (model, seconds) = train_model(&*b, &train, &cached, &pool, true, true, texture_only)?;
        let development_result =
            evaluate(&*b, &model, &development, &validation, &validation_features)?;
        let prospective_result =
            evaluate(&*b, &model, &prospective, &validation, &validation_features)?;
        let four_side_result = evaluate_four_sides(&model, &four_sides)?;
        let disjoint_result = evaluate_disjoint_four_sides(&model, &disjoint_sides)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "status": if texture_only {
                    "textured-mirror-training-experiment-only"
                } else {
                    "mirrored-edge-training-experiment-only"
                },
                "feature_illumination": texture_only,
                "illumination_phases": [
                    "identity", "checkerboard", "horizontal-gradient", "vertical-gradient"
                ],
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
                "mirror_slots": [0, 16, 32, 48],
                "mirror_orientations": ["identity", "horizontal", "vertical", "both"],
                "development_seed": 20261030,
                "development_samples": development.len(),
                "development_fingerprint": FROZEN_DEVELOPMENT_FINGERPRINT,
                "prospective_seed": PROSPECTIVE_SEED,
                "prospective_candidate_budget": PROSPECTIVE_CANDIDATES,
                "prospective_samples": prospective.len(),
                "prospective_fingerprint": PROSPECTIVE_FINGERPRINT,
                "final_holdout_evaluated": false,
                "four_side_generator": "TD1-four-side-paired-v1",
                "four_side_seed": FOUR_SIDE_SEED,
                "four_side_candidate_budget": FOUR_SIDE_CANDIDATES,
                "four_side_source_rows": 48,
                "four_side_transformed_samples": 192,
                "four_side_fingerprint": FOUR_SIDE_FINGERPRINT,
                "mirrored_four_side": four_side_result,
                "disjoint_generator": "TD1-four-side-disjoint-v1",
                "disjoint_seed": DISJOINT_SEED,
                "disjoint_candidate_budget": DISJOINT_CANDIDATES,
                "disjoint_source_rows": 144,
                "disjoint_last_source_index": 6876,
                "disjoint_fingerprint": DISJOINT_FINGERPRINT,
                "mirrored_disjoint_four_side": disjoint_result,
                "training_seconds": seconds,
                "validation": development_result["validation"],
                "validation_passes_original_thresholds":
                    development_result["validation_passes_original_thresholds"],
                "development": development_result["development"],
                "prospective": prospective_result["development"]
            }))
            .map_err(|e| e.to_string())?
        );
        if development_result["validation_passes_original_thresholds"] != true {
            return Err("opt-in training failed original validation thresholds".into());
        }
        return Ok(());
    }

    let (baseline_model, baseline_seconds) =
        train_model(&*b, &train, &cached, &pool, false, false, false)?;
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
    let baseline_four_side = evaluate_four_sides(&baseline_model, &four_sides)?;
    let baseline_disjoint = evaluate_disjoint_four_sides(&baseline_model, &disjoint_sides)?;
    let (boosted_model, boosted_seconds) = train_model(&*b, &train, &cached, &pool, true, false, false)?;
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
    let boosted_four_side = evaluate_four_sides(&boosted_model, &four_sides)?;
    let boosted_disjoint = evaluate_disjoint_four_sides(&boosted_model, &disjoint_sides)?;
    let recall = |run: &Value, group: &str| -> Result<f64, String> {
        run["development"][group]["recall_correct_class_iou50"]
            .as_f64()
            .ok_or_else(|| format!("missing development recall for {group}"))
    };
    let prospective_edge_delta =
        recall(&boosted_prospective, "small_edge")? - recall(&baseline_prospective, "small_edge")?;
    let prospective_interior_delta = recall(&boosted_prospective, "small_interior")?
        - recall(&baseline_prospective, "small_interior")?;
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
        "four_side_generator": "TD1-four-side-paired-v1",
        "four_side_seed": FOUR_SIDE_SEED,
        "four_side_candidate_budget": FOUR_SIDE_CANDIDATES,
        "four_side_source_rows": 48,
        "four_side_transformed_samples": 192,
        "four_side_fingerprint": FOUR_SIDE_FINGERPRINT,
        "baseline_four_side": baseline_four_side,
        "edge_sampling_four_side": boosted_four_side,
        "disjoint_generator": "TD1-four-side-disjoint-v1",
        "disjoint_seed": DISJOINT_SEED,
        "disjoint_candidate_budget": DISJOINT_CANDIDATES,
        "disjoint_source_rows": 144,
        "disjoint_last_source_index": 6876,
        "disjoint_fingerprint": DISJOINT_FINGERPRINT,
        "baseline_disjoint_four_side": baseline_disjoint,
        "edge_sampling_disjoint_four_side": boosted_disjoint,
        "baseline_seconds": baseline_seconds,
        "edge_sampling_seconds": boosted_seconds,
        "baseline": baseline,
        "edge_sampling": boosted,
        "baseline_prospective": baseline_prospective["development"],
        "edge_sampling_prospective": boosted_prospective["development"],
        "prospective_edge_minus_baseline_small_edge_iou50_recall": prospective_edge_delta,
        "prospective_edge_minus_baseline_small_interior_iou50_recall": prospective_interior_delta,
        "edge_minus_baseline_small_edge_iou50_recall":
            recall(&boosted, "small_edge")? - recall(&baseline, "small_edge")?,
        "edge_minus_baseline_small_interior_iou50_recall":
            recall(&boosted, "small_interior")? - recall(&baseline, "small_interior")?
    });
    write_controlled_report(&report, &mut std::io::stdout().lock())
}

fn write_controlled_report(report: &Value, output: &mut impl std::io::Write) -> Result<(), String> {
    serde_json::to_writer_pretty(&mut *output, report).map_err(|e| e.to_string())?;
    writeln!(output).map_err(|e| e.to_string())?;
    if report["baseline"]["validation_passes_original_thresholds"] != true {
        return Err("baseline failed the unchanged validation thresholds".into());
    }
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
    fn controlled_report_retains_completed_diagnostics_on_validation_failure() {
        let report = json!({
            "baseline": { "validation_passes_original_thresholds": false },
            "baseline_four_side": { "completed_metric": 0.25 },
            "edge_sampling_four_side": { "completed_metric": 0.5 }
        });
        let mut output = Vec::new();
        let result = write_controlled_report(&report, &mut output);
        assert_eq!(
            result.unwrap_err(),
            "baseline failed the unchanged validation thresholds"
        );
        let emitted: Value = serde_json::from_slice(&output)
            .expect("completed measurements must be emitted before validation failure");
        assert_eq!(emitted, report);
        assert_eq!(output.last(), Some(&b'\n'));
    }

    #[test]
    fn controlled_report_preserves_successful_output() {
        let report = json!({
            "baseline": { "validation_passes_original_thresholds": true },
            "baseline_four_side": { "completed_metric": 0.25 },
            "edge_sampling_four_side": { "completed_metric": 0.5 }
        });
        let mut output = Vec::new();
        write_controlled_report(&report, &mut output).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&output).unwrap(), report);
        assert_eq!(output.last(), Some(&b'\n'));
    }

    #[test]
    fn edge_selection_is_ground_truth_only_and_nonempty() {
        let train = dataset(101, 256);
        let pool = edge_pool(&train).unwrap();
        assert!(pool.iter().all(|&index| stratum(&train[index]) == 0));
        assert!(!pool.is_empty());
    }

    #[test]
    fn texture_training_retains_mirrored_sampler_and_labels() {
        let train = dataset(101, 256);
        let pool = edge_pool(&train).unwrap();
        let mut cached = Vec::with_capacity(train.len() * INPUTS);
        for sample in &train {
            cached.extend(features(&sample.pixels).unwrap());
        }
        let mut base_rng = Rng::new(707);
        let mut texture_rng = Rng::new(707);
        let (mut base_input, mut texture_input) = (Vec::new(), Vec::new());
        let (mut base_labels, mut texture_labels) = (Vec::new(), Vec::new());
        for step in 1..=4 {
            fill_edge_batch(
                &train, &cached, &pool, &mut base_rng, &mut base_input, &mut base_labels,
            );
            fill_edge_batch(
                &train, &cached, &pool, &mut texture_rng, &mut texture_input,
                &mut texture_labels,
            );
            mirror_reserved_edge_batch(&mut base_input, &mut base_labels).unwrap();
            mirror_reserved_edge_batch(&mut texture_input, &mut texture_labels).unwrap();
            assert_eq!(base_input, texture_input);
            let before = texture_labels
                .iter()
                .map(|sample| (sample.class, sample.box_xywh))
                .collect::<Vec<_>>();
            illuminate_training_batch(&mut texture_input, &texture_labels, step).unwrap();
            assert_ne!(base_input, texture_input);
            assert_eq!(
                texture_labels
                    .iter()
                    .map(|sample| (sample.class, sample.box_xywh))
                    .collect::<Vec<_>>(),
                before
            );
            assert_eq!(base_rng.next_u32(), texture_rng.next_u32());
        }
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
    fn disjoint_four_side_split_has_unique_sources_and_correct_side_geometry() {
        let groups = frozen_disjoint_four_sides().unwrap();
        let mut unique = std::collections::HashSet::new();
        for (side, samples) in groups.iter().enumerate() {
            assert_eq!(samples.len(), if side == 4 { 48 } else { 24 });
            for sample in samples {
                let hash = Sha256::digest(&sample.pixels).to_vec();
                assert!(
                    unique.insert(hash),
                    "a source is repeated in the disjoint split"
                );
                if side == 4 {
                    assert!(sample.class.is_none());
                    continue;
                }
                assert!(sample.class.is_some());
                let [x1, y1, x2, y2] = target_corners(sample);
                assert!((x2 - x1) * (y2 - y1) <= 256.0);
                let borders = [x1 <= 2.0, y1 <= 2.0, 64.0 - x2 <= 2.0, 64.0 - y2 <= 2.0];
                assert_eq!(borders.iter().filter(|&&near| near).count(), 1);
                assert!(borders[side]);
            }
        }
        assert_eq!(unique.len(), 144);
    }

    #[test]
    fn new_four_side_split_is_frozen_unique_and_has_correct_geometry() {
        let groups = frozen_four_sides().unwrap();
        let mut unique = std::collections::HashSet::new();
        let training: std::collections::HashSet<Vec<u8>> = dataset(101, 4096)
            .iter()
            .map(|sample| Sha256::digest(&sample.pixels).to_vec())
            .collect();
        for (side, samples) in groups.iter().enumerate() {
            assert_eq!(samples.len(), 48);
            for sample in samples {
                assert!(sample.class.is_some());
                let [x1, y1, x2, y2] = target_corners(sample);
                assert!((x2 - x1) * (y2 - y1) <= 256.0);
                let near_requested_side = match side {
                    0 => x1 <= 2.0,
                    1 => y1 <= 2.0,
                    2 => 64.0 - x2 <= 2.0,
                    3 => 64.0 - y2 <= 2.0,
                    _ => unreachable!(),
                };
                assert!(near_requested_side);
                let hash = Sha256::digest(&sample.pixels).to_vec();
                assert!(unique.insert(hash.clone()), "repeated transformed image");
                assert!(
                    !training.contains(&hash),
                    "training image leaked into probe"
                );
            }
        }
        assert_eq!(unique.len(), 192);
    }

    #[test]
    fn prospective_fingerprint_and_separation_are_frozen_before_inference() {
        let prospective = frozen_prospective().unwrap();
        let earlier = frozen_development().unwrap();
        assert_eq!(prospective.len(), QUOTA * GROUPS.len());
        let historical: std::collections::HashSet<&[u8]> = earlier
            .iter()
            .map(|sample| sample.pixels.as_slice())
            .collect();
        assert!(prospective
            .iter()
            .all(|sample| !historical.contains(sample.pixels.as_slice())));
        for group in 0..GROUPS.len() {
            assert_eq!(
                prospective
                    .iter()
                    .filter(|sample| stratum(sample) == group)
                    .count(),
                QUOTA
            );
        }
    }
}
