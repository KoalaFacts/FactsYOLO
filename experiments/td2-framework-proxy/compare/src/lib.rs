// SPDX-License-Identifier: Apache-2.0
//! Burn and Candle adapters for the stage B `fyp1` proxy, and its gates.
//!
//! Every parameter enters a framework as a flat 1-D leaf and is reshaped inside
//! the graph, so gradients come back in the `fyp_core::net::tensors` layout.
//! Batch norm is composed from tensor primitives in both candidates (neither
//! reviewed crate ships a batch-norm operation); no gradient is hand-written.
pub mod burn_adapter;
pub mod candle_adapter;
pub mod gates;

use fyp_core::data::IMAGE;
use fyp_core::loss::{HEAD_CHANNELS, HEAD_SIDE};
use fyp_core::net::PARAMETER_COUNT;

pub const BN_EPSILON: f64 = 1e-5;
pub const HEAD: usize = HEAD_CHANNELS * HEAD_SIDE * HEAD_SIDE;

/// One training framework under comparison.
pub trait Framework {
    fn name(&self) -> &'static str;
    /// Head output (N×5×16×16) for parameters and NCHW pixels.
    fn head(&self, parameters: &[f32], pixels: &[f32], n: usize) -> Result<Vec<f32>, String>;
    /// One taped forward: `upstream` maps the head output to dL/d(head), and the
    /// parameter gradient of `sum(head · upstream)` is returned in the flat layout.
    fn step(
        &self,
        parameters: &[f32],
        pixels: &[f32],
        n: usize,
        upstream: &mut dyn FnMut(&[f32]) -> Vec<f32>,
    ) -> Result<Vec<f32>, String>;
}

pub fn frameworks() -> Vec<Box<dyn Framework>> {
    vec![
        Box::new(burn_adapter::Burn),
        Box::new(candle_adapter::Candle),
    ]
}

pub(crate) fn check(parameters: &[f32], pixels: &[f32], n: usize) -> Result<(), String> {
    if parameters.len() != PARAMETER_COUNT {
        return Err(format!(
            "expected {PARAMETER_COUNT} parameters, got {}",
            parameters.len()
        ));
    }
    if n == 0 || pixels.len() != n * IMAGE {
        return Err(format!(
            "expected {} pixels for batch {n}, got {}",
            n * IMAGE,
            pixels.len()
        ));
    }
    Ok(())
}

pub(crate) fn check_upstream(upstream: &[f32], n: usize) -> Result<(), String> {
    if upstream.len() != n * HEAD {
        return Err(format!(
            "expected {} upstream values, got {}",
            n * HEAD,
            upstream.len()
        ));
    }
    Ok(())
}
