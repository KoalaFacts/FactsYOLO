// SPDX-License-Identifier: Apache-2.0
//! Shared proxy loss, computed outside every framework in F64.
//!
//! For a head output `Y` of shape N×5×16×16 (channel 0 objectness logit,
//! channels 1–4 left/top/right/bottom distance pre-activations):
//!
//! `L = (1/N) Σ_n [ (1/256) Σ_cells BCE(Y0, t0) + (1/4) Σ_k |softplus(Yk) − dk| ]`
//!
//! where `t0` is 1 only at the cell containing the box centre and `dk` are the
//! distances from that cell's centre to the box edges in stride units.
use crate::data::Sample;

pub const HEAD_CHANNELS: usize = 5;
pub const HEAD_SIDE: usize = 16;
pub const STRIDE: usize = 4;
const CELLS: usize = HEAD_SIDE * HEAD_SIDE;

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Positive cell `(row, col)` and its four distance targets.
pub fn target(s: &Sample) -> (usize, usize, [f64; 4]) {
    let [x0, y0, x1, y1] = s.bbox.map(|v| v as f64);
    let col = (((x0 + x1) / 2.0) / STRIDE as f64).floor() as usize;
    let row = (((y0 + y1) / 2.0) / STRIDE as f64).floor() as usize;
    let px = (col as f64 + 0.5) * STRIDE as f64;
    let py = (row as f64 + 0.5) * STRIDE as f64;
    let st = STRIDE as f64;
    (
        row,
        col,
        [
            (px - x0) / st,
            (py - y0) / st,
            (x1 - px) / st,
            (y1 - py) / st,
        ],
    )
}

/// Loss and its gradient with respect to the NCHW head output.
pub fn loss_and_gradient(head: &[f64], samples: &[Sample]) -> (f64, Vec<f64>) {
    let n = samples.len();
    assert_eq!(head.len(), n * HEAD_CHANNELS * CELLS, "head shape");
    let inv_n = 1.0 / n as f64;
    let mut grad = vec![0.0; head.len()];
    let mut loss = 0.0;
    for (i, s) in samples.iter().enumerate() {
        let (row, col, d) = target(s);
        let positive = row * HEAD_SIDE + col;
        let base = i * HEAD_CHANNELS * CELLS;
        for cell in 0..CELLS {
            let y = head[base + cell];
            let t = if cell == positive { 1.0 } else { 0.0 };
            loss += inv_n / CELLS as f64 * (y.max(0.0) - y * t + (-y.abs()).exp().ln_1p());
            grad[base + cell] = inv_n / CELLS as f64 * (sigmoid(y) - t);
        }
        for (k, dk) in d.iter().enumerate() {
            let at = base + (k + 1) * CELLS + positive;
            let e = softplus(head[at]) - dk;
            loss += inv_n * 0.25 * e.abs();
            let sign = if e > 0.0 {
                1.0
            } else if e < 0.0 {
                -1.0
            } else {
                0.0
            };
            grad[at] = inv_n * 0.25 * sign * sigmoid(head[at]);
        }
    }
    (loss, grad)
}
