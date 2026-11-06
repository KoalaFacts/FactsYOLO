// SPDX-License-Identifier: Apache-2.0
//! Reference TD2 post-processing (development only).
//!
//! Implements the decoding contract of the TD2 design, section 6 "Decoding",
//! as plain, dependency-free Rust so that stage D training code and the
//! stage E native backend can both be tested against one reference:
//!
//! 1. Letterbox geometry: the frame is scaled by `min(S / w, S / h)`, centred
//!    in an S×S square, and model-space boxes are mapped back to clipped
//!    original-image pixel edges.
//! 2. Anchor-free box decoding: every feature location at stride `s` has the
//!    centre `((x + 0.5) * s, (y + 0.5) * s)` and predicts non-negative
//!    left/top/right/bottom distances in stride units (FCOS idea,
//!    <https://arxiv.org/abs/1904.01355>; no code ported).
//! 3. Score = `sqrt(sigmoid(objectness) * sigmoid(class logit))`; candidates
//!    below the caller's threshold are dropped, the best `pre_nms_top_k`
//!    are kept, per-class greedy NMS suppresses a box whose IoU with a kept
//!    box of the same class is **above** `nms_iou`, then results are sorted
//!    and capped.
//!
//! Precision and order, fixed so a backend can match bit for bit:
//! - scores and IoU are compared in `f32` (IoU exactly `0.65f32` is kept);
//! - NMS works on the clipped original-image boxes; candidates entirely in
//!   the padding are dropped after the top-k cut;
//! - candidates are ranked for top-k and greedy NMS by score descending,
//!   class id, level order, then location index (row-major);
//! - the final sort and cap use the core's ABI comparator: score
//!   descending, class id, `left`, `top`, `right`, `bottom`, then rank.
//!
//! Nonfinite logits or distances, negative distances and kept candidates
//! whose model-space box overflows are [`DecodeError::OutputInvalid`],
//! matching the core's `FY_MODEL_OUTPUT_INVALID` rule.
//!
//! This crate is not part of the deployed library and changes no ABI or
//! artifact format.

/// A box as pixel edges (`left <= right`, `top <= bottom`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxEdges {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl BoxEdges {
    fn area(&self) -> f32 {
        (self.right - self.left).max(0.0) * (self.bottom - self.top).max(0.0)
    }

    fn iou(&self, other: &BoxEdges) -> f32 {
        let w = (self.right.min(other.right) - self.left.max(other.left)).max(0.0);
        let h = (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0.0);
        let inter = w * h;
        let union = self.area() + other.area() - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Frame or model input size is zero, or the input size is not a
    /// multiple of 32.
    InvalidSize,
    /// A level's stride or buffer lengths disagree with the input size, the
    /// class count differs between levels, or no level was given.
    ShapeMismatch,
    /// A logit or distance is nonfinite, or a distance is negative.
    OutputInvalid,
    /// An option is out of range.
    InvalidOptions,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeOptions {
    /// Minimum fused score kept, in `[0, 1]`.
    pub score_threshold: f32,
    /// Candidates kept before NMS (spec: 1,000).
    pub pre_nms_top_k: usize,
    /// Same-class IoU above which the lower-ranked box is suppressed, in
    /// `(0, 1]` (spec: 0.65).
    pub nms_iou: f32,
    /// Result cap applied after NMS; exceeding it sets `truncated`.
    pub max_detections: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            score_threshold: 0.25,
            pre_nms_top_k: 1000,
            nms_iou: 0.65,
            max_detections: 100,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub class_id: u32,
    pub score: f32,
    /// Clipped original-image pixel edges.
    pub edges: BoxEdges,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    pub detections: Vec<Detection>,
    /// More detections survived NMS than `max_detections`.
    pub truncated: bool,
}

/// Raw head outputs of one pyramid level, row-major over an
/// `(S / stride) × (S / stride)` grid. The class count is
/// `class_logits.len() / cells` and must be the same on every level.
#[derive(Clone, Copy, Debug)]
pub struct LevelOutput<'a> {
    pub stride: u32,
    /// One objectness logit per location.
    pub objectness: &'a [f32],
    /// `classes` logits per location.
    pub class_logits: &'a [f32],
    /// Left, top, right, bottom distances per location, in stride units.
    pub distances: &'a [f32],
}

/// Aspect-preserving, centred letterbox of a `width × height` frame into an
/// `size × size` model input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    width: u32,
    height: u32,
    size: u32,
    scale: f64,
    pad_x: f64,
    pad_y: f64,
}

impl Letterbox {
    pub fn new(width: u32, height: u32, size: u32) -> Result<Self, DecodeError> {
        if width == 0 || height == 0 || size == 0 || size % 32 != 0 {
            return Err(DecodeError::InvalidSize);
        }
        let s = f64::from(size);
        let scale = (s / f64::from(width)).min(s / f64::from(height));
        Ok(Self {
            width,
            height,
            size,
            scale,
            pad_x: (s - f64::from(width) * scale) / 2.0,
            pad_y: (s - f64::from(height) * scale) / 2.0,
        })
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn pad_x(&self) -> f64 {
        self.pad_x
    }

    pub fn pad_y(&self) -> f64 {
        self.pad_y
    }

    /// Maps a model-space box to original-image edges clipped to the frame;
    /// `None` when nothing of it lies inside the frame.
    pub fn to_source(&self, b: BoxEdges) -> Option<BoxEdges> {
        let w = f64::from(self.width);
        let h = f64::from(self.height);
        let x = |v: f32| ((f64::from(v) - self.pad_x) / self.scale).clamp(0.0, w) as f32;
        let y = |v: f32| ((f64::from(v) - self.pad_y) / self.scale).clamp(0.0, h) as f32;
        let out = BoxEdges {
            left: x(b.left),
            top: y(b.top),
            right: x(b.right),
            bottom: y(b.bottom),
        };
        (out.right > out.left && out.bottom > out.top).then_some(out)
    }
}

struct Candidate {
    score: f32,
    class_id: u32,
    level: usize,
    cell: usize,
}

fn sigmoid(v: f32) -> f64 {
    1.0 / (1.0 + (-f64::from(v)).exp())
}

fn validate(options: &DecodeOptions) -> Result<(), DecodeError> {
    let t = options.score_threshold;
    let iou = options.nms_iou;
    let ok = t.is_finite()
        && (0.0..=1.0).contains(&t)
        && iou.is_finite()
        && iou > 0.0
        && iou <= 1.0
        && options.pre_nms_top_k > 0
        && options.max_detections > 0;
    if ok {
        Ok(())
    } else {
        Err(DecodeError::InvalidOptions)
    }
}

/// Validates shapes and values; returns the shared class count.
fn check_levels(levels: &[LevelOutput<'_>], size: u32) -> Result<usize, DecodeError> {
    let mut classes = None;
    for level in levels {
        if level.stride == 0 || size % level.stride != 0 {
            return Err(DecodeError::ShapeMismatch);
        }
        let grid = (size / level.stride) as usize;
        let cells = grid.checked_mul(grid).ok_or(DecodeError::ShapeMismatch)?;
        let coords = cells.checked_mul(4).ok_or(DecodeError::ShapeMismatch)?;
        let n = level.class_logits.len();
        if level.objectness.len() != cells
            || level.distances.len() != coords
            || n == 0
            || n % cells != 0
        {
            return Err(DecodeError::ShapeMismatch);
        }
        if *classes.get_or_insert(n / cells) != n / cells {
            return Err(DecodeError::ShapeMismatch);
        }
    }
    let classes = classes.ok_or(DecodeError::ShapeMismatch)?;
    for level in levels {
        let finite = level
            .objectness
            .iter()
            .chain(level.class_logits)
            .all(|v| v.is_finite());
        let distances_ok = level.distances.iter().all(|d| d.is_finite() && *d >= 0.0);
        if !finite || !distances_ok {
            return Err(DecodeError::OutputInvalid);
        }
    }
    Ok(classes)
}

/// Decodes raw TD2 head outputs into sorted, NMS-filtered, capped detections
/// in original-image coordinates.
pub fn decode(
    levels: &[LevelOutput<'_>],
    letterbox: &Letterbox,
    options: &DecodeOptions,
) -> Result<Decoded, DecodeError> {
    validate(options)?;
    let classes = check_levels(levels, letterbox.size)?;

    let mut candidates = Vec::new();
    for (li, level) in levels.iter().enumerate() {
        for (cell, &obj) in level.objectness.iter().enumerate() {
            let p_obj = sigmoid(obj);
            let logits = &level.class_logits[cell * classes..(cell + 1) * classes];
            for (class, &c) in logits.iter().enumerate() {
                let score = (p_obj * sigmoid(c)).sqrt() as f32;
                if score >= options.score_threshold {
                    candidates.push(Candidate {
                        score,
                        class_id: class as u32,
                        level: li,
                        cell,
                    });
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.class_id.cmp(&b.class_id))
            .then(a.level.cmp(&b.level))
            .then(a.cell.cmp(&b.cell))
    });
    candidates.truncate(options.pre_nms_top_k);

    let mut kept: Vec<Detection> = Vec::new();
    for c in candidates {
        let level = &levels[c.level];
        let grid = (letterbox.size / level.stride) as usize;
        let stride = level.stride as f32;
        let cx = ((c.cell % grid) as f32 + 0.5) * stride;
        let cy = ((c.cell / grid) as f32 + 0.5) * stride;
        let d = &level.distances[c.cell * 4..c.cell * 4 + 4];
        let model = BoxEdges {
            left: cx - d[0] * stride,
            top: cy - d[1] * stride,
            right: cx + d[2] * stride,
            bottom: cy + d[3] * stride,
        };
        let finite = [model.left, model.top, model.right, model.bottom]
            .iter()
            .all(|v| v.is_finite());
        if !finite {
            return Err(DecodeError::OutputInvalid);
        }
        let Some(edges) = letterbox.to_source(model) else {
            continue;
        };
        let suppressed = kept
            .iter()
            .any(|k| k.class_id == c.class_id && k.edges.iou(&edges) > options.nms_iou);
        if !suppressed {
            kept.push(Detection {
                class_id: c.class_id,
                score: c.score,
                edges,
            });
        }
    }

    // The core's ABI comparator; `kept` is already in rank order, so a
    // stable sort makes rank the final tie-break.
    kept.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.class_id.cmp(&b.class_id))
            .then(a.edges.left.total_cmp(&b.edges.left))
            .then(a.edges.top.total_cmp(&b.edges.top))
            .then(a.edges.right.total_cmp(&b.edges.right))
            .then(a.edges.bottom.total_cmp(&b.edges.bottom))
    });
    let truncated = kept.len() > options.max_detections;
    kept.truncate(options.max_detections);
    Ok(Decoded {
        detections: kept,
        truncated,
    })
}
