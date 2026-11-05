// SPDX-License-Identifier: Apache-2.0
//! The predeclared stage B gates (docs/testing/td2-framework-comparison.md).
use crate::Framework;
use fyp_core::data::{batch, pixels, Sample};
use fyp_core::loss::loss_and_gradient;
use fyp_core::net::{initial_parameters, SEED};
use fyp_core::reference::{finite_difference, reference_loss};
use fyp_core::sgd::Sgd;

/// G2 bound on the relative forward-loss error.
pub const G2_RELATIVE: f64 = 1e-5;
/// G3 bound: |g − g_fd| ≤ G3_ABSOLUTE + G3_RELATIVE·|g_fd|.
pub const G3_ABSOLUTE: f64 = 1e-4;
pub const G3_RELATIVE: f64 = 1e-2;
/// G4 bound on the relative L2 gradient difference between candidates.
pub const G4_RELATIVE: f64 = 1e-4;
/// G5: mean loss of the last 20 steps ≤ G5_RATIO × mean of the first 20.
pub const G5_RATIO: f64 = 0.6;
pub const BATCH: usize = 16;
pub const STEPS: usize = 300;

/// Loss and gradient of one taped step with the shared F64 loss.
pub fn loss_step(
    fw: &dyn Framework,
    parameters: &[f32],
    samples: &[Sample],
) -> Result<(f64, Vec<f32>), String> {
    let mut loss = f64::NAN;
    let grad = fw.step(parameters, &pixels(samples), samples.len(), &mut |head| {
        let head: Vec<f64> = head.iter().map(|v| f64::from(*v)).collect();
        let (l, g) = loss_and_gradient(&head, samples);
        loss = l;
        g.into_iter().map(|v| v as f32).collect()
    })?;
    Ok((loss, grad))
}

/// G2: relative error of the framework loss against the F64 reference.
pub fn forward_error(
    fw: &dyn Framework,
    parameters: &[f32],
    samples: &[Sample],
) -> Result<f64, String> {
    let head = fw.head(parameters, &pixels(samples), samples.len())?;
    let head: Vec<f64> = head.iter().map(|v| f64::from(*v)).collect();
    let loss = loss_and_gradient(&head, samples).0;
    let p64: Vec<f64> = parameters.iter().map(|v| f64::from(*v)).collect();
    let reference = reference_loss(&p64, samples);
    Ok((loss - reference).abs() / reference.abs())
}

/// Reference finite differences for the given parameter indices.
pub fn reference_gradients(parameters: &[f32], samples: &[Sample], indices: &[usize]) -> Vec<f64> {
    let p64: Vec<f64> = parameters.iter().map(|v| f64::from(*v)).collect();
    indices
        .iter()
        .map(|i| finite_difference(&p64, samples, *i))
        .collect()
}

/// G3 ratio for one sample: error over allowed error (≤ 1 passes).
pub fn g3_ratio(g: f64, fd: f64) -> f64 {
    (g - fd).abs() / (G3_ABSOLUTE + G3_RELATIVE * fd.abs())
}

pub fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let diff: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum();
    let norm: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (diff / norm).sqrt()
}

/// G5/G6: train with the shared optimizer; returns the per-step loss history.
pub fn train(fw: &dyn Framework, steps: usize, n: usize) -> Result<Vec<f64>, String> {
    let mut parameters = initial_parameters();
    let mut opt = Sgd::new(parameters.len());
    let mut history = Vec::with_capacity(steps);
    for step in 0..steps {
        let samples = batch(SEED, step as u64, n);
        let (loss, grad) = loss_step(fw, &parameters, &samples)?;
        if !loss.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return Err(format!("non-finite loss or gradient at step {step}"));
        }
        opt.step(&mut parameters, &grad);
        history.push(loss);
    }
    Ok(history)
}

/// G5 ratio: mean of the last 20 losses over the mean of the first 20.
pub fn g5_ratio(history: &[f64]) -> f64 {
    let window = 20.min(history.len());
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    mean(&history[history.len() - window..]) / mean(&history[..window])
}
