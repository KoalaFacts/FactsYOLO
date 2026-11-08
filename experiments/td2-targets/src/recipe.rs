// SPDX-License-Identifier: Apache-2.0
//! Frozen TD2 loss weighting, optimizer settings and learning-rate schedule
//! (stage D2b2a).
//!
//! Fixed before any training so they are not tuned against the holdout
//! (design section 6). Ideas only from published papers: decoupled weight
//! decay ([AdamW](https://arxiv.org/abs/1711.05101)), linear warmup
//! ([Goyal et al.](https://arxiv.org/abs/1706.02677)), cosine decay
//! ([SGDR](https://arxiv.org/abs/1608.03983)) and a heavier IoU-type box term
//! ([YOLOX](https://arxiv.org/abs/2107.08430)). No code was ported.
//!
//! Per-image loss, with `n = max(1, positive locations)` summed over every
//! level:
//! - `cls`: sigmoid focal loss over every location and class (one-hot target
//!   at positives, zero elsewhere), divided by `n`;
//! - `obj`: binary cross-entropy over every location (1 at positives,
//!   0 elsewhere), divided by `n`;
//! - `boxes`: `1 − GIoU` over positive locations, divided by `n`;
//! - `total = W_CLS · cls + W_OBJ · obj + W_BOX · boxes`, an error when not
//!   finite (finite extreme logits can sum past `f64::MAX`).
//!
//! A batch loss is the mean of its per-image totals ([`batch_loss`]).

use crate::{LevelTargets, bce_with_logits, giou_loss, sigmoid_focal_loss};

/// Weight of the classification term.
pub const W_CLS: f64 = 1.0;
/// Weight of the objectness term.
pub const W_OBJ: f64 = 1.0;
/// Weight of the GIoU box term.
pub const W_BOX: f64 = 5.0;

/// Peak learning rate (AdamW).
pub const BASE_LR: f64 = 2e-3;
/// Final learning rate as a fraction of `BASE_LR`.
pub const MIN_LR_FRACTION: f64 = 0.05;
/// Shortest schedule accepted: warmup is `total / 20` steps, at least one.
pub const MIN_TOTAL_STEPS: u32 = 20;
/// AdamW first and second moment decay.
pub const ADAMW_BETAS: (f64, f64) = (0.9, 0.999);
/// AdamW denominator epsilon.
pub const ADAMW_EPSILON: f64 = 1e-8;
/// Decoupled weight decay, applied to convolution weights only (not biases).
pub const WEIGHT_DECAY: f64 = 0.05;
/// Global gradient L2-norm clip applied before every optimizer step.
pub const GRAD_CLIP_NORM: f64 = 10.0;

/// Raw head outputs of one level, row-major like `LevelTargets::cells`.
#[derive(Clone, Copy, Debug)]
pub struct LevelOutputs<'a> {
    /// `cells × class_count` class logits.
    pub cls: &'a [f64],
    /// `cells` objectness logits.
    pub obj: &'a [f64],
    /// `cells × 4` `[l, t, r, b]` distances in stride units. All must be
    /// finite; they must be non-negative only at positives (GIoU is taken
    /// there alone).
    pub ltrb: &'a [f64],
}

/// Normalized loss terms of one image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossParts {
    pub cls: f64,
    pub obj: f64,
    pub boxes: f64,
    pub total: f64,
    pub positives: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum LossError {
    ClassCount,
    LevelCount {
        targets: usize,
        outputs: usize,
    },
    /// Cell list or output lengths do not match the level's grid.
    Shape {
        level: usize,
    },
    /// A positive's class is not below the class count.
    TargetClass {
        level: usize,
        cell: usize,
    },
    /// A logit or distance output is NaN or infinite.
    NonFinite {
        level: usize,
        cell: usize,
    },
    /// A positive's predicted box has a negative distance or unrepresentable
    /// geometry, so GIoU is undefined.
    Box {
        level: usize,
        cell: usize,
    },
    /// Every output is finite but a summed term or the weighted total is not
    /// representable as finite F64.
    Overflow,
    /// A batch has no images.
    EmptyBatch,
}

/// Loss of one image from its assigned targets and raw head outputs.
pub fn image_loss(
    targets: &[LevelTargets],
    outputs: &[LevelOutputs],
    class_count: u16,
) -> Result<LossParts, LossError> {
    if class_count == 0 {
        return Err(LossError::ClassCount);
    }
    if targets.len() != outputs.len() {
        return Err(LossError::LevelCount {
            targets: targets.len(),
            outputs: outputs.len(),
        });
    }
    let classes = usize::from(class_count);
    let (mut cls, mut obj, mut boxes, mut positives) = (0.0, 0.0, 0.0, 0usize);
    for (level, (t, o)) in targets.iter().zip(outputs).enumerate() {
        let cells = t.cells.len();
        let shape_ok = t.width.checked_mul(t.height) == Some(cells)
            && cells.checked_mul(classes) == Some(o.cls.len())
            && o.obj.len() == cells
            && cells.checked_mul(4) == Some(o.ltrb.len());
        if !shape_ok {
            return Err(LossError::Shape { level });
        }
        for (cell, target) in t.cells.iter().enumerate() {
            let logits = &o.cls[cell * classes..(cell + 1) * classes];
            let distances = &o.ltrb[cell * 4..cell * 4 + 4];
            if !(logits.iter().chain(distances).all(|v| v.is_finite()) && o.obj[cell].is_finite()) {
                return Err(LossError::NonFinite { level, cell });
            }
            let class = match target {
                Some(a) if usize::from(a.class) >= classes => {
                    return Err(LossError::TargetClass { level, cell });
                }
                Some(a) => Some(usize::from(a.class)),
                None => None,
            };
            for (k, &logit) in logits.iter().enumerate() {
                cls += sigmoid_focal_loss(logit, if class == Some(k) { 1.0 } else { 0.0 });
            }
            obj += bce_with_logits(o.obj[cell], if target.is_some() { 1.0 } else { 0.0 });
            if let Some(a) = target {
                let pred = [distances[0], distances[1], distances[2], distances[3]];
                boxes += giou_loss(pred, a.ltrb).ok_or(LossError::Box { level, cell })?;
                positives += 1;
            }
        }
    }
    let n = positives.max(1) as f64;
    let (cls, obj, boxes) = (cls / n, obj / n, boxes / n);
    let total = W_CLS * cls + W_OBJ * obj + W_BOX * boxes;
    // Every term is finite and non-negative, so NaN cannot arise, but
    // finite extreme logits can still sum past F64::MAX.
    if !total.is_finite() {
        return Err(LossError::Overflow);
    }
    Ok(LossParts {
        cls,
        obj,
        boxes,
        total,
        positives,
    })
}

/// Batch loss: the mean of the per-image totals.
pub fn batch_loss(images: &[LossParts]) -> Result<f64, LossError> {
    if images.is_empty() {
        return Err(LossError::EmptyBatch);
    }
    let sum: f64 = images.iter().map(|p| p.total).sum();
    if !sum.is_finite() {
        return Err(LossError::Overflow);
    }
    Ok(sum / images.len() as f64)
}

#[derive(Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// Fewer than `MIN_TOTAL_STEPS` optimizer steps.
    TotalSteps(u32),
    /// The step index is not below the total.
    Step { step: u32, total: u32 },
}

/// Linear-warmup length for a run of `total` optimizer steps.
pub fn warmup_steps(total: u32) -> Result<u32, ScheduleError> {
    if total < MIN_TOTAL_STEPS {
        return Err(ScheduleError::TotalSteps(total));
    }
    Ok((total / 20).max(1))
}

/// Learning rate of zero-based optimizer step `step` in a run of `total`.
///
/// Steps `0..w` rise linearly to `BASE_LR` (`BASE_LR · (step + 1) / w`);
/// steps `w..total` follow a cosine from `BASE_LR` at step `w` to
/// `BASE_LR · MIN_LR_FRACTION` at the last step.
pub fn lr_at(step: u32, total: u32) -> Result<f64, ScheduleError> {
    let w = warmup_steps(total)?;
    if step >= total {
        return Err(ScheduleError::Step { step, total });
    }
    if step < w {
        return Ok(BASE_LR * f64::from(step + 1) / f64::from(w));
    }
    // total >= 20 and w <= total / 20, so the decay spans at least 18 steps.
    let progress = f64::from(step - w) / f64::from(total - 1 - w);
    let floor = BASE_LR * MIN_LR_FRACTION;
    Ok(floor + (BASE_LR - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()))
}
