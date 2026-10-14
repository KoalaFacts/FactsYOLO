// SPDX-License-Identifier: Apache-2.0
use super::{validate, Backend};
use burn::tensor::{activation::relu, Device, Tensor, TensorData};
use fy_tiny_model::{B1, HIDDEN, INPUTS, W1, W2};
pub struct Burn;
struct Graph {
    w1: Tensor<2>,
    b1: Tensor<2>,
    w2: Tensor<2>,
    b2: Tensor<2>,
    output: Tensor<2>,
    device: Device,
}
fn graph(weights: &[f32], input: &[f32], batch: usize, track: bool) -> Graph {
    let device = if track {
        Device::flex().autodiff()
    } else {
        Device::flex()
    };
    let make = |slice: &[f32], shape: [usize; 2]| {
        let tensor = Tensor::<2>::from_data(TensorData::new(slice.to_vec(), shape), &device);
        if track {
            tensor.require_grad()
        } else {
            tensor
        }
    };
    let w1 = make(&weights[..W1], [INPUTS, HIDDEN]);
    let b1 = make(&weights[W1..B1], [1, HIDDEN]);
    let w2 = make(&weights[B1..W2], [HIDDEN, 7]);
    let b2 = make(&weights[W2..], [1, 7]);
    let x = Tensor::<2>::from_data(TensorData::new(input.to_vec(), [batch, INPUTS]), &device);
    let output = relu(x.matmul(w1.clone()) + b1.clone()).matmul(w2.clone()) + b2.clone();
    Graph {
        w1,
        b1,
        w2,
        b2,
        output,
        device,
    }
}
impl Backend for Burn {
    fn name(&self) -> &'static str {
        "burn-flex-0.22.0"
    }
    fn forward(&self, weights: &[f32], input: &[f32], batch: usize) -> Result<Vec<f32>, String> {
        validate(weights, input, batch, None)?;
        graph(weights, input, batch, false)
            .output
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))
    }
    fn backward(
        &self,
        weights: &[f32],
        input: &[f32],
        batch: usize,
        output_gradient: &[f32],
    ) -> Result<Vec<f32>, String> {
        validate(weights, input, batch, Some(output_gradient))?;
        let g = graph(weights, input, batch, true);
        let upstream = Tensor::<2>::from_data(
            TensorData::new(output_gradient.to_vec(), [batch, 7]),
            &g.device,
        );
        let gradients = (g.output * upstream).sum().backward();
        let mut values = Vec::with_capacity(weights.len());
        for parameter in [g.w1, g.b1, g.w2, g.b2] {
            let grad = parameter
                .grad(&gradients)
                .ok_or("Burn did not retain parameter gradient")?;
            values.extend(
                grad.to_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| format!("{e:?}"))?,
            );
        }
        Ok(values)
    }
}
