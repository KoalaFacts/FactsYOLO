// SPDX-License-Identifier: Apache-2.0
//! Frozen TD2 training-target assignment and loss reference (stage D2a).
//!
//! Original F64 implementation of the rule fixed in section 6 of the TD2
//! design, written before any training so the rule is not tuned against
//! results. Ideas only from published papers: anchor-free distance targets
//! and per-level scale ranges ([FCOS](https://arxiv.org/abs/1904.01355)),
//! centre sampling, sigmoid focal loss
//! ([Lin et al.](https://arxiv.org/abs/1708.02002)) and GIoU loss
//! ([Rezatofighi et al.](https://arxiv.org/abs/1902.09630)). No code was
//! ported. Development tooling only: never linked into the deployed library,
//! no I/O, no dependencies. The training framework chosen in stage B must
//! reproduce these values; this crate is the oracle it is checked against.
//!
//! Rule, per pyramid level with stride `s` and input size `S`:
//! 1. Locations are cell centres `((col + 0.5) s, (row + 0.5) s)` on an
//!    `S/s × S/s` grid.
//! 2. A location is a candidate for a box when it lies strictly inside the
//!    box and within `CENTER_RADIUS · s` of the box centre on both axes
//!    (strict inequality).
//! 3. With `l, t, r, b` the pixel distances from the location to the box
//!    edges, the candidate is kept only when `max(l, t, r, b)` is in the
//!    level's half-open range `(lo, hi]` from `SCALE_RANGES`.
//! 4. A location kept for several boxes goes to the smallest box area; equal
//!    areas go to the lowest box index.
//! 5. The regression target is `[l, t, r, b] / s` (stride units).

pub mod recipe;

/// Pyramid strides, finest first.
pub const STRIDES: [u32; 3] = [8, 16, 32];
/// Per-level `(lo, hi]` ranges in pixels for `max(l, t, r, b)`.
pub const SCALE_RANGES: [(f64, f64); 3] = [(0.0, 64.0), (64.0, 128.0), (128.0, f64::INFINITY)];
/// Centre-sampling radius in strides.
pub const CENTER_RADIUS: f64 = 2.5;
/// Focal-loss weight of the positive class.
pub const FOCAL_ALPHA: f64 = 0.25;
/// Focal-loss focusing exponent.
pub const FOCAL_GAMMA: f64 = 2.0;

/// Ground-truth box in letterboxed input pixels with its class index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtBox {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub class: u16,
}

/// Target of one positive location.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assigned {
    /// Index of the box in the caller's slice.
    pub gt_index: usize,
    pub class: u16,
    /// Distances to the left, top, right and bottom edges in stride units.
    pub ltrb: [f64; 4],
}

/// Targets of one pyramid level, row-major; `None` is a negative location.
#[derive(Debug)]
pub struct LevelTargets {
    pub stride: u32,
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Option<Assigned>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum TargetError {
    /// The input size is zero or not a multiple of the coarsest stride.
    InputSize(u32),
    /// The class count is zero.
    ClassCount,
    /// The box has nonfinite coordinates, nonpositive spans, or an unrepresentable F64 span/area.
    InvalidBox(usize),
    ClassOutOfRange {
        index: usize,
        class: u16,
    },
}

/// Assigns targets for one `input_size × input_size` image.
pub fn assign(
    input_size: u32,
    class_count: u16,
    boxes: &[GtBox],
) -> Result<Vec<LevelTargets>, TargetError> {
    let coarsest = STRIDES[STRIDES.len() - 1];
    if input_size == 0 || input_size % coarsest != 0 {
        return Err(TargetError::InputSize(input_size));
    }
    if class_count == 0 {
        return Err(TargetError::ClassCount);
    }
    for (index, b) in boxes.iter().enumerate() {
        let finite = [b.left, b.top, b.right, b.bottom]
            .iter()
            .all(|v| v.is_finite());
        let width = b.right - b.left;
        let height = b.bottom - b.top;
        // Finite endpoints can still overflow a span or area. Silently
        // accepting infinite areas drops their targets: every cell starts
        // with best_area = infinity, so an infinite area can never win.
        if !finite
            || width <= 0.0
            || height <= 0.0
            || !width.is_finite()
            || !height.is_finite()
            || !(width * height).is_finite()
        {
            return Err(TargetError::InvalidBox(index));
        }
        if b.class >= class_count {
            return Err(TargetError::ClassOutOfRange {
                index,
                class: b.class,
            });
        }
    }

    let levels = STRIDES
        .iter()
        .zip(SCALE_RANGES)
        .map(|(&stride, (lo, hi))| {
            let side = (input_size / stride) as usize;
            let s = f64::from(stride);
            let radius = CENTER_RADIUS * s;
            let mut cells = vec![None; side * side];
            // Best (area, index) per cell, so ties resolve without re-reading boxes.
            let mut best_area = vec![f64::INFINITY; side * side];
            for row in 0..side {
                let y = (row as f64 + 0.5) * s;
                for col in 0..side {
                    let x = (col as f64 + 0.5) * s;
                    let at = row * side + col;
                    for (gt_index, b) in boxes.iter().enumerate() {
                        let (l, t, r, bt) = (x - b.left, y - b.top, b.right - x, b.bottom - y);
                        if l <= 0.0 || t <= 0.0 || r <= 0.0 || bt <= 0.0 {
                            continue;
                        }
                        let (cx, cy) = ((b.left + b.right) / 2.0, (b.top + b.bottom) / 2.0);
                        if (x - cx).abs() >= radius || (y - cy).abs() >= radius {
                            continue;
                        }
                        let m = l.max(t).max(r).max(bt);
                        if m <= lo || m > hi {
                            continue;
                        }
                        let area = (b.right - b.left) * (b.bottom - b.top);
                        // Strict `<` keeps the lowest index on equal areas.
                        if area < best_area[at] {
                            best_area[at] = area;
                            cells[at] = Some(Assigned {
                                gt_index,
                                class: b.class,
                                ltrb: [l / s, t / s, r / s, bt / s],
                            });
                        }
                    }
                }
            }
            LevelTargets {
                stride,
                width: side,
                height: side,
                cells,
            }
        })
        .collect();
    Ok(levels)
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Binary cross-entropy of a logit against a target in `[0, 1]`, stable for
/// any finite logit.
pub fn bce_with_logits(logit: f64, target: f64) -> f64 {
    logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p()
}

/// Sigmoid focal loss with `FOCAL_ALPHA` and `FOCAL_GAMMA` for a binary target.
pub fn sigmoid_focal_loss(logit: f64, target: f64) -> f64 {
    let p = sigmoid(logit);
    let p_t = target * p + (1.0 - target) * (1.0 - p);
    let alpha_t = target * FOCAL_ALPHA + (1.0 - target) * (1.0 - FOCAL_ALPHA);
    alpha_t * (1.0 - p_t).powf(FOCAL_GAMMA) * bce_with_logits(logit, target)
}

/// `1 − GIoU` of two boxes given as `[l, t, r, b]` distances from the same
/// location. `None` when a distance is negative or nonfinite, both boxes
/// are empty, or intermediate geometry is not representable as finite F64.
/// In particular, finite distances must never produce a nonfinite loss.
pub fn giou_loss(pred: [f64; 4], target: [f64; 4]) -> Option<f64> {
    if pred
        .iter()
        .chain(&target)
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return None;
    }
    let area = |d: [f64; 4]| (d[0] + d[2]) * (d[1] + d[3]);
    let pred_area = area(pred);
    let target_area = area(target);
    let inter_w = pred[0].min(target[0]) + pred[2].min(target[2]);
    let inter_h = pred[1].min(target[1]) + pred[3].min(target[3]);
    let inter = inter_w * inter_h;
    if !pred_area.is_finite() || !target_area.is_finite() || !inter.is_finite() {
        return None;
    }
    let union = pred_area + target_area - inter;
    if !union.is_finite() || union <= 0.0 {
        return None;
    }
    let enclosing = (pred[0].max(target[0]) + pred[2].max(target[2]))
        * (pred[1].max(target[1]) + pred[3].max(target[3]));
    if !enclosing.is_finite() || enclosing <= 0.0 {
        return None;
    }
    let loss = 1.0 - inter / union + (enclosing - union) / enclosing;
    loss.is_finite().then_some(loss)
}
