// SPDX-License-Identifier: Apache-2.0
use fy_cpu_evaluate::{
    backend,
    metrics::{iou, measure, target_corners},
    training::{adam_step, fill_training_batch},
    Backend,
};
use fy_tiny_model::{
    dataset, decode, features, loss_gradient, Model, Rng, Sample, INPUTS, PARAMETERS,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
    time::Instant,
};

const USAGE: &str = "usage: fy-cpu-evaluate train <burn|candle> <new-output-directory> | infer <artifact> <64x64.rgb>";
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > limit {
        return Err("input exceeds bounded file size".into());
    }
    Ok(bytes)
}
fn prediction(value: Option<fy_tiny_model::Detection>) -> Value {
    match value {
        Some(d) => json!({"class": d.class, "score": d.score, "corners": d.corners}),
        None => Value::Null,
    }
}
fn fingerprint(samples: &[Sample]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"TD1-synthetic-v1");
    for sample in samples {
        hash.update([sample.class.map_or(255, |c| c as u8)]);
        for v in sample.box_xywh {
            hash.update(v.to_le_bytes());
        }
        hash.update(&sample.pixels);
    }
    format!("{:x}", hash.finalize())
}
fn input(samples: &[Sample]) -> Result<Vec<f32>, String> {
    let mut result = Vec::with_capacity(samples.len() * INPUTS);
    for sample in samples {
        result.extend(features(&sample.pixels)?);
    }
    Ok(result)
}
fn forward(b: &dyn Backend, model: &Model, samples: &[Sample]) -> Result<Vec<f32>, String> {
    let mut output = Vec::with_capacity(samples.len() * 7);
    for chunk in samples.chunks(512) {
        output.extend(b.forward(&model.weights, &input(chunk)?, chunk.len())?);
    }
    Ok(output)
}
// A shared validation output avoids repeating the complete tensor forward pass
// for the same frozen model and dataset when collecting loss and metrics.
fn validation_loss_and_metrics(
    b: &dyn Backend,
    model: &Model,
    samples: &[Sample],
) -> Result<(f32, fy_cpu_evaluate::metrics::Metrics), String> {
    let output = forward(b, model, samples)?;
    let loss = loss_gradient(&output, samples)?.0;
    let metrics = measure(&output, samples)?;
    Ok((loss, metrics))
}

fn close(a: f32, b: f32) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-4 + 1e-4 * b.abs()
}
fn train(name: &str, destination: &Path) -> Result<(), String> {
    let b = backend(name)?;
    fs::create_dir(destination).map_err(|e| format!("create new output directory: {e}"))?;
    let train = dataset(101, 4096);
    let validation = dataset(202, 128);
    let test = dataset(909, 128);
    let cached = input(&train)?;
    let mut model = Model::initialized(404);
    let initial_weights = model.weights.clone();
    let initial_loss = loss_gradient(&forward(&*b, &model, &train)?, &train)?.0;
    let untrained = measure(&forward(&*b, &model, &test)?, &test)?;
    let mut m = vec![0.0; PARAMETERS];
    let mut v = vec![0.0; PARAMETERS];
    let mut rng = Rng::new(707);
    let mut batch = Vec::with_capacity(64 * INPUTS);
    let mut labels = Vec::with_capacity(64);
    let mut losses = File::create(destination.join("loss.tsv")).map_err(|e| e.to_string())?;
    writeln!(losses, "step\tbatch_loss").map_err(|e| e.to_string())?;
    let started = Instant::now();
    for step in 1..=2000 {
        if started.elapsed().as_secs_f64() > 600.0 {
            return Err("training exceeded 600 second budget".into());
        }
        fill_training_batch(&train, &cached, &mut rng, &mut batch, &mut labels);
        let outputs = b.forward(&model.weights, &batch, 64)?;
        let (loss, gradient) = loss_gradient(&outputs, &labels)?;
        let parameter_gradient = b.backward(&model.weights, &batch, 64, &gradient)?;
        adam_step(
            &mut model.weights,
            &parameter_gradient,
            &mut m,
            &mut v,
            step,
        )?;
        writeln!(losses, "{step}\t{loss:.9}").map_err(|e| e.to_string())?;
        if step % 200 == 0 {
            eprintln!(
                "{} step={step} loss={loss:.6} elapsed={:.3}s",
                b.name(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    let training_seconds = started.elapsed().as_secs_f64();
    let train_output = forward(&*b, &model, &train)?;
    let final_loss = loss_gradient(&train_output, &train)?.0;
    let train_metrics = measure(&train_output, &train)?;
    let (validation_loss, validation_metrics) =
        validation_loss_and_metrics(&*b, &model, &validation)?;
    let outputs = forward(&*b, &model, &test)?;
    let test_metrics = measure(&outputs, &test)?;
    let bytes = model.to_bytes()?;
    fs::write(destination.join("model.fytd"), &bytes).map_err(|e| e.to_string())?;
    let loaded = Model::from_bytes(&read_bounded(&destination.join("model.fytd"), 65536)?)?;
    let exact_reload = initial_weights != model.weights
        && model
            .weights
            .iter()
            .zip(&loaded.weights)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let mut max_reference_error = 0.0_f32;
    let mut reference_matches = true;
    let mut predictions = Vec::new();
    let mut reload_predictions_exact = true;
    for (index, (out, sample)) in outputs.as_chunks::<7>().0.iter().zip(&test).enumerate() {
        let feature = features(&sample.pixels)?;
        let reference = model.forward(&feature)?;
        let reloaded = loaded.forward(&feature)?;
        reload_predictions_exact &= reference
            .iter()
            .zip(reloaded)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        for (a, r) in out.iter().zip(reference) {
            max_reference_error = max_reference_error.max((a - r).abs());
            reference_matches &= close(*a, r);
        }
        let detection = decode(&reloaded)?;
        let overlap = match (&detection, sample.class) {
            (Some(d), Some(_)) => iou(d.corners, target_corners(sample)),
            _ => 0.0,
        };
        predictions.push(json!({"index":index,"expected_class":sample.class,"expected_corners":sample.class.map(|_|target_corners(sample)),"prediction":prediction(detection),"iou":overlap}));
        // Separate CLI process verification consumes both positive and background examples.
        if index < 6 {
            fs::write(
                destination.join(format!("sample-{index}.rgb")),
                &sample.pixels,
            )
            .map_err(|e| e.to_string())?;
        }
    }
    let mut failures = Vec::new();
    if final_loss > initial_loss * 0.25 {
        failures.push("training loss reduction");
    }
    if !validation_metrics.passes() {
        failures.push("validation metrics");
    }
    if !test_metrics.passes() {
        failures.push("test metrics");
    }
    if untrained.passes() {
        failures.push("untrained negative control unexpectedly passed");
    }
    if !exact_reload || !reload_predictions_exact {
        failures.push("exact parameter/prediction reload");
    }
    if !reference_matches {
        failures.push("backend/portable numerical consistency");
    }
    if training_seconds > 600.0 {
        failures.push("training time budget");
    }
    let report = json!({"architecture":"TD1","generator":"TD1-synthetic-v1","training_fingerprint":fingerprint(&train),"validation_fingerprint":fingerprint(&validation),"test_fingerprint":fingerprint(&test),"backend":b.name(),"source_commit":option_env!("FACTSYOLO_BUILD_ID").unwrap_or("unknown"),
        "training_samples":4096,"validation_samples":128,"test_samples":128,"training_seed":101,"validation_seed":202,"test_seed":909,"initialization_seed":404,"sampler_seed":707,
        "steps":2000,"batch":64,"parameters":PARAMETERS,"initial_loss":initial_loss,"final_loss":final_loss,"train":train_metrics,"validation_loss":validation_loss,
        "training_seconds":training_seconds,"untrained_test":untrained,"validation":validation_metrics,"test":test_metrics,
        "exact_weight_reload":exact_reload,"exact_reference_prediction_reload":reload_predictions_exact,
        "max_backend_reference_absolute_error":max_reference_error,"backend_reference_matches":reference_matches,
        "artifact_bytes":bytes.len(),"predictions":predictions,"passed":failures.is_empty(),"failures":failures});
    fs::write(
        destination.join("metrics.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"backend":b.name(),"initial_loss":initial_loss,"final_loss":final_loss,"train":train_metrics,"validation_loss":validation_loss,"test":test_metrics,"training_seconds":training_seconds,"passed":failures.is_empty(),"failures":failures})
    );
    if !failures.is_empty() {
        return Err(format!(
            "experiment acceptance failed: {}",
            failures.join(", ")
        ));
    }
    Ok(())
}
fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command, name, destination]
            if command == "train" && ["burn", "candle"].contains(&name.as_str()) =>
        {
            train(name, Path::new(destination))
        }
        [command, artifact, image] if command == "infer" => {
            let model = Model::from_bytes(&read_bounded(Path::new(artifact), 65536)?)?;
            let pixels = read_bounded(Path::new(image), 64 * 64 * 3)?;
            println!("{}", prediction(model.detect(&pixels)?));
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
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
    fn validation_summary_uses_one_forward_pass_per_chunk() {
        use std::cell::Cell;

        struct CountingBackend {
            calls: Cell<usize>,
        }
        impl Backend for CountingBackend {
            fn name(&self) -> &'static str {
                "counting"
            }
            fn forward(
                &self,
                _weights: &[f32],
                _input: &[f32],
                batch: usize,
            ) -> Result<Vec<f32>, String> {
                self.calls.set(self.calls.get() + 1);
                Ok(vec![0.0; batch * 7])
            }
            fn backward(
                &self,
                _weights: &[f32],
                _input: &[f32],
                _batch: usize,
                _output_gradient: &[f32],
            ) -> Result<Vec<f32>, String> {
                panic!("validation must not run a gradient pass")
            }
        }

        // 513 samples force two chunks. One positive and one background class
        // are necessary for measure() to accept the validation population.
        let samples: Vec<Sample> = (0..513)
            .map(|i| Sample {
                pixels: vec![0; 64 * 64 * 3],
                class: if i == 512 { None } else { Some(0) },
                box_xywh: [0.5; 4],
            })
            .collect();
        let backend = CountingBackend {
            calls: Cell::new(0),
        };
        let model = Model::initialized(404);
        let (loss, metrics) = validation_loss_and_metrics(&backend, &model, &samples).unwrap();
        assert!(loss.is_finite());
        assert_eq!(metrics.positives, 512);
        assert_eq!(metrics.negatives, 1);
        assert_eq!(
            backend.calls.get(),
            2,
            "validation loss and metrics must share the same two chunk outputs"
        );
    }

    #[test]
    fn reused_batches_match_historical_sampling_and_keep_storage() {
        let train = dataset(101, 16);
        let cached = input(&train).unwrap();
        let mut rng = Rng::new(707);
        let mut reference_rng = Rng::new(707);
        let mut batch = Vec::with_capacity(64 * INPUTS);
        let mut labels = Vec::with_capacity(64);
        let batch_ptr = batch.as_ptr();
        let labels_ptr = labels.as_ptr();

        // The reference intentionally allocates fresh buffers each step, as the
        // former training loop did. Five sequential batches detect RNG drift.
        for _ in 0..5 {
            fill_training_batch(&train, &cached, &mut rng, &mut batch, &mut labels);
            let mut expected_batch = Vec::with_capacity(64 * INPUTS);
            let mut expected_labels = Vec::with_capacity(64);
            for _ in 0..64 {
                let index = reference_rng.next_u32() as usize % train.len();
                expected_batch.extend_from_slice(&cached[index * INPUTS..(index + 1) * INPUTS]);
                expected_labels.push((train[index].class, train[index].box_xywh.map(f32::to_bits)));
            }
            assert_eq!(batch, expected_batch);
            assert_eq!(
                labels
                    .iter()
                    .map(|sample| (sample.class, sample.box_xywh.map(f32::to_bits)))
                    .collect::<Vec<_>>(),
                expected_labels
            );
            assert!(labels.iter().all(|sample| sample.pixels.is_empty()));
            assert_eq!(batch.as_ptr(), batch_ptr, "feature buffer reallocated");
            assert_eq!(labels.as_ptr(), labels_ptr, "label buffer reallocated");
        }
    }
}
