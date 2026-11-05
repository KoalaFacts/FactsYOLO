// SPDX-License-Identifier: Apache-2.0
use crate::{check, check_upstream, Framework, BN_EPSILON};
use burn::tensor::activation::silu;
use burn::tensor::module::{conv2d, interpolate};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Tensor, TensorData};
use fyp_core::data::SIDE;
use fyp_core::net::tensors;

pub struct Burn;

struct Graph {
    leaves: Vec<Tensor<1>>,
    head: Tensor<4>,
    device: Device,
}

fn batch_norm(x: Tensor<4>, gamma: Tensor<1>, beta: Tensor<1>) -> Tensor<4> {
    let c = gamma.dims()[0];
    let mean = x.clone().mean_dims(&[0, 2, 3]);
    let centred = x - mean;
    let var = centred.clone().powi_scalar(2).mean_dims(&[0, 2, 3]);
    let normalised = centred / (var + BN_EPSILON).sqrt();
    normalised * gamma.reshape([1, c, 1, 1]) + beta.reshape([1, c, 1, 1])
}

fn graph(parameters: &[f32], pixels: &[f32], n: usize, track: bool) -> Graph {
    let device = if track {
        Device::flex().autodiff()
    } else {
        Device::flex()
    };
    let leaves: Vec<Tensor<1>> = tensors()
        .iter()
        .map(|t| {
            let leaf = Tensor::<1>::from_data(
                TensorData::new(parameters[t.offset..t.offset + t.len].to_vec(), [t.len]),
                &device,
            );
            if track {
                leaf.require_grad()
            } else {
                leaf
            }
        })
        .collect();
    let p = |i: usize| leaves[i].clone();
    let kernel = |i: usize, shape: [usize; 4]| p(i).reshape(shape);
    let options = |stride: usize, pad: usize, groups: usize| {
        ConvOptions::new([stride, stride], [pad, pad], [1, 1], groups)
    };
    let x = Tensor::<4>::from_data(
        TensorData::new(pixels.to_vec(), [n, 3, SIDE, SIDE]),
        &device,
    );
    let a = silu(batch_norm(
        conv2d(x, kernel(0, [16, 3, 3, 3]), None, options(2, 1, 1)),
        p(1),
        p(2),
    ));
    let a = silu(batch_norm(
        conv2d(a, kernel(3, [16, 1, 3, 3]), None, options(2, 1, 16)),
        p(4),
        p(5),
    ));
    let pw = silu(batch_norm(
        conv2d(a, kernel(6, [32, 16, 1, 1]), None, options(1, 0, 1)),
        p(7),
        p(8),
    ));
    let a = silu(batch_norm(
        conv2d(
            pw.clone(),
            kernel(9, [64, 32, 3, 3]),
            None,
            options(2, 1, 1),
        ),
        p(10),
        p(11),
    ));
    let lateral = conv2d(a, kernel(12, [32, 64, 1, 1]), Some(p(13)), options(1, 0, 1));
    let up = interpolate(
        lateral,
        InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([16, 16]),
    );
    let head = conv2d(
        up + pw,
        kernel(14, [5, 32, 1, 1]),
        Some(p(15)),
        options(1, 0, 1),
    );
    Graph {
        leaves,
        head,
        device,
    }
}

fn values<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>, String> {
    t.to_data()
        .try_to_vec::<f32>()
        .map_err(|e| format!("{e:?}"))
}

impl Framework for Burn {
    fn name(&self) -> &'static str {
        "burn-flex-0.22.0"
    }
    fn head(&self, parameters: &[f32], pixels: &[f32], n: usize) -> Result<Vec<f32>, String> {
        check(parameters, pixels, n)?;
        values(graph(parameters, pixels, n, false).head)
    }
    fn step(
        &self,
        parameters: &[f32],
        pixels: &[f32],
        n: usize,
        upstream: &mut dyn FnMut(&[f32]) -> Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        check(parameters, pixels, n)?;
        let g = graph(parameters, pixels, n, true);
        let upstream = upstream(&values(g.head.clone())?);
        check_upstream(&upstream, n)?;
        let u = Tensor::<4>::from_data(TensorData::new(upstream, [n, 5, 16, 16]), &g.device);
        let grads = (g.head * u).sum().backward();
        let mut out = Vec::with_capacity(parameters.len());
        for leaf in g.leaves {
            out.extend(values(
                leaf.grad(&grads)
                    .ok_or("Burn did not retain a parameter gradient")?,
            )?);
        }
        Ok(out)
    }
}
