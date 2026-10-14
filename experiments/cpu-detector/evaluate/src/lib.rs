// SPDX-License-Identifier: Apache-2.0
mod burn_adapter;
mod candle_adapter;
use fy_tiny_model::{INPUTS, PARAMETERS};
pub trait Backend {
    fn name(&self) -> &'static str;
    fn forward(&self, weights: &[f32], input: &[f32], batch: usize) -> Result<Vec<f32>, String>;
    fn backward(
        &self,
        weights: &[f32],
        input: &[f32],
        batch: usize,
        output_gradient: &[f32],
    ) -> Result<Vec<f32>, String>;
}
pub fn backend(name: &str) -> Result<Box<dyn Backend>, String> {
    match name {
        "burn" => Ok(Box::new(burn_adapter::Burn)),
        "candle" => Ok(Box::new(candle_adapter::Candle)),
        _ => Err("backend must be burn or candle".into()),
    }
}
fn validate(
    weights: &[f32],
    input: &[f32],
    batch: usize,
    gradient: Option<&[f32]>,
) -> Result<(), String> {
    if batch == 0
        || batch > 512
        || weights.len() != PARAMETERS
        || weights.iter().any(|w| !w.is_finite() || w.abs() > 10000.0)
        || batch.checked_mul(INPUTS) != Some(input.len())
        || input
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err("invalid backend weights/input/batch".into());
    }
    if let Some(g) = gradient {
        if batch.checked_mul(7) != Some(g.len()) || g.iter().any(|v| !v.is_finite()) {
            return Err("invalid output gradient".into());
        }
    }
    Ok(())
}

pub mod training;

pub mod metrics;
