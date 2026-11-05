// SPDX-License-Identifier: Apache-2.0
use crate::{check, check_upstream, Framework, BN_EPSILON};
use candle_core::{Device, Result as CResult, Tensor, Var};
use fyp_core::data::SIDE;
use fyp_core::net::tensors;

pub struct Candle;

struct Graph {
    leaves: Vec<Var>,
    head: Tensor,
}

fn batch_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> CResult<Tensor> {
    let c = gamma.dim(0)?;
    let mean = x.mean_keepdim(0)?.mean_keepdim(2)?.mean_keepdim(3)?;
    let centred = x.broadcast_sub(&mean)?;
    let var = centred
        .sqr()?
        .mean_keepdim(0)?
        .mean_keepdim(2)?
        .mean_keepdim(3)?;
    let normalised = centred.broadcast_div(&(var + BN_EPSILON)?.sqrt()?)?;
    normalised
        .broadcast_mul(&gamma.reshape((1, c, 1, 1))?)?
        .broadcast_add(&beta.reshape((1, c, 1, 1))?)
}

fn graph(parameters: &[f32], pixels: &[f32], n: usize) -> CResult<Graph> {
    let device = Device::Cpu;
    let leaves = tensors()
        .iter()
        .map(|t| Var::from_slice(&parameters[t.offset..t.offset + t.len], t.len, &device))
        .collect::<CResult<Vec<Var>>>()?;
    let p = |i: usize| leaves[i].as_tensor();
    let kernel = |i: usize, shape: (usize, usize, usize, usize)| p(i).reshape(shape);
    let block = |x: &Tensor, k: Tensor, pad, stride, groups, g: usize| -> CResult<Tensor> {
        batch_norm(&x.conv2d(&k, pad, stride, 1, groups)?, p(g), p(g + 1))?.silu()
    };
    let x = Tensor::from_slice(pixels, (n, 3, SIDE, SIDE), &device)?;
    let a = block(&x, kernel(0, (16, 3, 3, 3))?, 1, 2, 1, 1)?;
    let a = block(&a, kernel(3, (16, 1, 3, 3))?, 1, 2, 16, 4)?;
    let pw = block(&a, kernel(6, (32, 16, 1, 1))?, 0, 1, 1, 7)?;
    let a = block(&pw, kernel(9, (64, 32, 3, 3))?, 1, 2, 1, 10)?;
    let lateral = a
        .conv2d(&kernel(12, (32, 64, 1, 1))?, 0, 1, 1, 1)?
        .broadcast_add(&p(13).reshape((1, 32, 1, 1))?)?;
    let fused = (lateral.upsample_nearest2d(16, 16)? + pw)?;
    let head = fused
        .conv2d(&kernel(14, (5, 32, 1, 1))?, 0, 1, 1, 1)?
        .broadcast_add(&p(15).reshape((1, 5, 1, 1))?)?;
    Ok(Graph { leaves, head })
}

impl Framework for Candle {
    fn name(&self) -> &'static str {
        "candle-cpu-0.11.0"
    }
    fn head(&self, parameters: &[f32], pixels: &[f32], n: usize) -> Result<Vec<f32>, String> {
        check(parameters, pixels, n)?;
        (|| {
            graph(parameters, pixels, n)?
                .head
                .detach()
                .flatten_all()?
                .to_vec1::<f32>()
        })()
        .map_err(|e| e.to_string())
    }
    fn step(
        &self,
        parameters: &[f32],
        pixels: &[f32],
        n: usize,
        upstream: &mut dyn FnMut(&[f32]) -> Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        check(parameters, pixels, n)?;
        let g = graph(parameters, pixels, n).map_err(|e| e.to_string())?;
        let head = g
            .head
            .flatten_all()
            .and_then(|h| h.to_vec1::<f32>())
            .map_err(|e| e.to_string())?;
        let upstream = upstream(&head);
        check_upstream(&upstream, n)?;
        (|| {
            let u = Tensor::from_slice(&upstream, (n, 5, 16, 16), &Device::Cpu)?;
            let grads = g.head.mul(&u)?.sum_all()?.backward()?;
            let mut out = Vec::with_capacity(parameters.len());
            for leaf in &g.leaves {
                let grad = grads.get(leaf.as_tensor()).ok_or_else(|| {
                    candle_core::Error::Msg("Candle did not retain a parameter gradient".into())
                })?;
                out.extend(grad.flatten_all()?.to_vec1::<f32>()?);
            }
            Ok(out)
        })()
        .map_err(|e: candle_core::Error| e.to_string())
    }
}
