// SPDX-License-Identifier: Apache-2.0
//! Original-image result normalization; no model-specific suppression.
pub(crate) mod owned;
use crate::{Error, error::INVALID, ffi::types::*};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub score: f32,
    pub class_id: u32,
    pub source_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ImageTransform {
    pub scale_x: f32,
    pub scale_y: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

#[derive(Debug)]
pub struct Finalized {
    pub detections: Vec<FyDetection>,
    pub truncated: bool,
}

const OUTPUT: Error = Error {
    status: FY_MODEL_OUTPUT_INVALID,
    message: "invalid model output or image transform",
};
const ALLOCATION: Error = Error {
    status: FY_LIMIT_EXCEEDED,
    message: "result allocation request could not be satisfied",
};

#[derive(Clone, Copy, Debug)]
struct Ranked {
    detection: FyDetection,
    source_index: usize,
}

// Better records compare Less; BinaryHeap keeps the worst retained record first.
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = &self.detection;
        let b = &other.detection;
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.class_id.cmp(&b.class_id))
            .then_with(|| a.x_min.total_cmp(&b.x_min))
            .then_with(|| a.y_min.total_cmp(&b.y_min))
            .then_with(|| a.x_max.total_cmp(&b.x_max))
            .then_with(|| a.y_max.total_cmp(&b.y_max))
            .then_with(|| self.source_index.cmp(&other.source_index))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}

fn canonical_zero(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

fn coordinate(value: f32, pad: f32, scale: f32, bound: u32) -> Result<f32, Error> {
    let original = (value - pad) / scale;
    if !original.is_finite() {
        return Err(OUTPUT);
    }
    Ok(canonical_zero(original.clamp(0.0, bound as f32)))
}

/// Validate and normalize post-suppression candidates, retaining at most the cap.
///
/// Coordinates are inverted, checked for finite arithmetic, clipped to original
/// image pixel edges, then zero-area boxes are discarded. Every candidate is
/// checked, including those below threshold or outside the retained top-k.
/// Ordering is score descending, class ascending, coordinates lexicographically,
/// and source_index ascending. Both coordinates and scores normalize signed zero.
/// Storage is O(min(candidate count, cap)); no model-specific NMS occurs here.
/// Fallible reservations do not constitute universal process-OOM recovery.
pub fn finalize(
    candidates_after_suppression: &[Candidate],
    transform: ImageTransform,
    source: (u32, u32),
    class_count: u32,
    threshold: f32,
    max_detections: usize,
) -> Result<Finalized, Error> {
    if !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
        || max_detections == 0
        || source.0 == 0
        || source.1 == 0
    {
        return Err(INVALID);
    }
    if class_count == 0
        || !transform.scale_x.is_finite()
        || !transform.scale_y.is_finite()
        || transform.scale_x <= 0.0
        || transform.scale_y <= 0.0
        || !transform.pad_x.is_finite()
        || !transform.pad_y.is_finite()
    {
        return Err(OUTPUT);
    }
    let mut retained = BinaryHeap::<Ranked>::new();
    retained
        .try_reserve_exact(candidates_after_suppression.len().min(max_detections))
        .map_err(|_| ALLOCATION)?;
    let mut truncated = false;
    for candidate in candidates_after_suppression {
        if !candidate.score.is_finite()
            || !(0.0..=1.0).contains(&candidate.score)
            || candidate.class_id >= class_count
            || !candidate.x_min.is_finite()
            || !candidate.y_min.is_finite()
            || !candidate.x_max.is_finite()
            || !candidate.y_max.is_finite()
            || candidate.x_min > candidate.x_max
            || candidate.y_min > candidate.y_max
        {
            return Err(OUTPUT);
        }
        let detection = FyDetection {
            x_min: coordinate(
                candidate.x_min,
                transform.pad_x,
                transform.scale_x,
                source.0,
            )?,
            y_min: coordinate(
                candidate.y_min,
                transform.pad_y,
                transform.scale_y,
                source.1,
            )?,
            x_max: coordinate(
                candidate.x_max,
                transform.pad_x,
                transform.scale_x,
                source.0,
            )?,
            y_max: coordinate(
                candidate.y_max,
                transform.pad_y,
                transform.scale_y,
                source.1,
            )?,
            score: canonical_zero(candidate.score),
            class_id: candidate.class_id,
        };
        if detection.x_min == detection.x_max
            || detection.y_min == detection.y_max
            || detection.score < threshold
        {
            continue;
        }
        let ranked = Ranked {
            detection,
            source_index: candidate.source_index,
        };
        if retained.len() < max_detections {
            retained.push(ranked);
        } else {
            truncated = true;
            if let Some(mut worst) = retained.peek_mut() {
                if ranked < *worst {
                    *worst = ranked;
                }
            }
        }
    }
    let mut ordered = retained.into_vec();
    ordered.sort_unstable();
    let mut detections = Vec::new();
    detections
        .try_reserve_exact(ordered.len())
        .map_err(|_| ALLOCATION)?;
    detections.extend(ordered.into_iter().map(|ranked| ranked.detection));
    Ok(Finalized {
        detections,
        truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_numeric_ties_are_ordered_by_original_candidate_index() {
        let first = Ranked {
            detection: FyDetection {
                x_min: 0.0,
                y_min: 0.0,
                x_max: 1.0,
                y_max: 1.0,
                score: 0.9,
                class_id: 0,
            },
            source_index: 7,
        };
        let second = Ranked {
            source_index: 9,
            ..first
        };
        assert!(first < second);
        assert_eq!(first.partial_cmp(&second), Some(Ordering::Less));
        assert_eq!(first, first);
    }
}
