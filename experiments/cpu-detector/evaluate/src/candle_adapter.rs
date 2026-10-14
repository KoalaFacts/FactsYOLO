// SPDX-License-Identifier: Apache-2.0
use super::{validate, Backend};
use candle_core::{Device, Tensor, Var};
use fy_tiny_model::{B1, HIDDEN, INPUTS, W1, W2};
pub struct Candle;
struct Graph {
    parameters: Vec<Var>,
    output: Tensor,
}
fn graph(weights: &[f32], input: &[f32], batch: usize) -> candle_core::Result<Graph> {
    let device = Device::Cpu;
    let w1 = Var::from_slice(&weights[..W1], (INPUTS, HIDDEN), &device)?;
    let b1 = Var::from_slice(&weights[W1..B1], (1, HIDDEN), &device)?;
    let w2 = Var::from_slice(&weights[B1..W2], (HIDDEN, 7), &device)?;
    let b2 = Var::from_slice(&weights[W2..], (1, 7), &device)?;
    let x = Tensor::from_slice(input, (batch, INPUTS), &device)?;
    let output = x
        .matmul(&w1)?
        .broadcast_add(&b1)?
        .relu()?
        .matmul(&w2)?
        .broadcast_add(&b2)?;
    Ok(Graph {
        parameters: vec![w1, b1, w2, b2],
        output,
    })
}
impl Backend for Candle {
    fn name(&self) -> &'static str {
        "candle-cpu-0.11.0"
    }
    fn forward(&self, weights: &[f32], input: &[f32], batch: usize) -> Result<Vec<f32>, String> {
        validate(weights, input, batch, None)?;
        (|| {
            graph(weights, input, batch)?
                .output
                .flatten_all()?
                .to_vec1::<f32>()
        })()
        .map_err(|e: candle_core::Error| e.to_string())
    }
    fn backward(
        &self,
        weights: &[f32],
        input: &[f32],
        batch: usize,
        output_gradient: &[f32],
    ) -> Result<Vec<f32>, String> {
        validate(weights, input, batch, Some(output_gradient))?;
        (|| {
            let g = graph(weights, input, batch)?;
            let upstream = Tensor::from_slice(output_gradient, (batch, 7), &Device::Cpu)?;
            let gradients = g.output.mul(&upstream)?.sum_all()?.backward()?;
            let mut values = Vec::with_capacity(weights.len());
            for p in g.parameters {
                let grad = gradients.get(&p).ok_or_else(|| {
                    candle_core::Error::Msg("Candle did not retain parameter gradient".into())
                })?;
                values.extend(grad.flatten_all()?.to_vec1::<f32>()?);
            }
            Ok(values)
        })()
        .map_err(|e: candle_core::Error| e.to_string())
    }
}
