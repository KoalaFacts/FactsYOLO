// SPDX-License-Identifier: Apache-2.0
//! Independent F64 reference forward of `fyp1`, written from the layer table in
//! `docs/testing/td2-framework-comparison.md`. It shares no code with any
//! framework adapter; its finite differences are the gradient reference.
use crate::data::{pixels, Sample, SIDE};
use crate::loss::loss_and_gradient;
use crate::net::offset;

pub const BN_EPSILON: f64 = 1e-5;
pub const FD_STEP: f64 = 1e-4;

/// A square convolution with padding `k / 2`.
#[derive(Clone, Copy, Debug)]
pub struct Conv {
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,
    pub stride: usize,
    pub groups: usize,
}

/// NCHW convolution of `n` square images of side `side`; kernel layout
/// `[c_out, c_in/groups, k, k]`. Returns the output and its side.
pub fn conv2d(
    x: &[f64],
    n: usize,
    side: usize,
    w: &[f64],
    bias: Option<&[f64]>,
    c: Conv,
) -> (Vec<f64>, usize) {
    let pad = c.k / 2;
    let out = (side + 2 * pad - c.k) / c.stride + 1;
    let in_per_group = c.c_in / c.groups;
    let out_per_group = c.c_out / c.groups;
    assert_eq!(x.len(), n * c.c_in * side * side, "conv input shape");
    assert_eq!(
        w.len(),
        c.c_out * in_per_group * c.k * c.k,
        "conv kernel shape"
    );
    let mut y = vec![0.0; n * c.c_out * out * out];
    for b in 0..n {
        for o in 0..c.c_out {
            let g = o / out_per_group;
            let start = bias.map_or(0.0, |v| v[o]);
            for oy in 0..out {
                for ox in 0..out {
                    let mut acc = start;
                    for ci in 0..in_per_group {
                        let ic = g * in_per_group + ci;
                        for ky in 0..c.k {
                            let iy = (oy * c.stride + ky) as isize - pad as isize;
                            if iy < 0 || iy >= side as isize {
                                continue;
                            }
                            for kx in 0..c.k {
                                let ix = (ox * c.stride + kx) as isize - pad as isize;
                                if ix < 0 || ix >= side as isize {
                                    continue;
                                }
                                let xv = x
                                    [((b * c.c_in + ic) * side + iy as usize) * side + ix as usize];
                                let wv = w[((o * in_per_group + ci) * c.k + ky) * c.k + kx];
                                acc += xv * wv;
                            }
                        }
                    }
                    y[((b * c.c_out + o) * out + oy) * out + ox] = acc;
                }
            }
        }
    }
    (y, out)
}

/// Training-mode batch norm: biased batch statistics per channel over N, H, W.
pub fn batch_norm_train(
    x: &[f64],
    n: usize,
    c: usize,
    hw: usize,
    gamma: &[f64],
    beta: &[f64],
) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    let count = (n * hw) as f64;
    for ch in 0..c {
        let values =
            || (0..n).flat_map(move |b| x[(b * c + ch) * hw..(b * c + ch + 1) * hw].iter());
        let mean = values().sum::<f64>() / count;
        let var = values().map(|v| (v - mean) * (v - mean)).sum::<f64>() / count;
        let scale = gamma[ch] / (var + BN_EPSILON).sqrt();
        for b in 0..n {
            for i in (b * c + ch) * hw..(b * c + ch + 1) * hw {
                y[i] = (x[i] - mean) * scale + beta[ch];
            }
        }
    }
    y
}

fn silu(x: &mut [f64]) {
    for v in x {
        *v /= 1.0 + (-*v).exp();
    }
}

/// Nearest-neighbour ×2 upsampling of `[.., c, side, side]` planes.
pub fn upsample2(x: &[f64], c: usize, side: usize) -> Vec<f64> {
    let planes = x.len() / (side * side);
    assert_eq!(planes % c, 0, "upsample shape");
    let up = side * 2;
    let mut y = vec![0.0; planes * up * up];
    for p in 0..planes {
        for oy in 0..up {
            for ox in 0..up {
                y[(p * up + oy) * up + ox] = x[(p * side + oy / 2) * side + ox / 2];
            }
        }
    }
    y
}

fn slice<'a>(p: &'a [f64], name: &str, len: usize) -> &'a [f64] {
    let at = offset(name);
    &p[at..at + len]
}

fn block(p: &[f64], x: &[f64], n: usize, side: usize, name: &str, c: Conv) -> (Vec<f64>, usize) {
    let w = slice(
        p,
        &format!("{name}.w"),
        c.c_out * c.c_in / c.groups * c.k * c.k,
    );
    let (mut y, out) = conv2d(x, n, side, w, None, c);
    y = batch_norm_train(
        &y,
        n,
        c.c_out,
        out * out,
        slice(p, &format!("{name}.bn.g"), c.c_out),
        slice(p, &format!("{name}.bn.b"), c.c_out),
    );
    silu(&mut y);
    (y, out)
}

/// Head output, N×5×16×16, for F64 parameters.
pub fn reference_head(p: &[f64], samples: &[Sample]) -> Vec<f64> {
    let n = samples.len();
    let x: Vec<f64> = pixels(samples).into_iter().map(f64::from).collect();
    let conv = |c_in, c_out, k, stride, groups| Conv {
        c_in,
        c_out,
        k,
        stride,
        groups,
    };
    let (a, s) = block(p, &x, n, SIDE, "stem", conv(3, 16, 3, 2, 1));
    let (a, s) = block(p, &a, n, s, "dw", conv(16, 16, 3, 2, 16));
    let (pw, s16) = block(p, &a, n, s, "pw", conv(16, 32, 1, 1, 1));
    let (a, s8) = block(p, &pw, n, s16, "down", conv(32, 64, 3, 2, 1));
    let (lat, _) = conv2d(
        &a,
        n,
        s8,
        slice(p, "lateral.w", 32 * 64),
        Some(slice(p, "lateral.b", 32)),
        conv(64, 32, 1, 1, 1),
    );
    let mut fused = upsample2(&lat, 32, s8);
    for (f, v) in fused.iter_mut().zip(&pw) {
        *f += v;
    }
    conv2d(
        &fused,
        n,
        s16,
        slice(p, "head.w", 5 * 32),
        Some(slice(p, "head.b", 5)),
        conv(32, 5, 1, 1, 1),
    )
    .0
}

pub fn reference_loss(p: &[f64], samples: &[Sample]) -> f64 {
    loss_and_gradient(&reference_head(p, samples), samples).0
}

/// Central difference of the reference loss for parameter `index`, step `FD_STEP`.
pub fn finite_difference(p: &[f64], samples: &[Sample], index: usize) -> f64 {
    let mut q = p.to_vec();
    q[index] = p[index] + FD_STEP;
    let up = reference_loss(&q, samples);
    q[index] = p[index] - FD_STEP;
    let down = reference_loss(&q, samples);
    (up - down) / (2.0 * FD_STEP)
}
