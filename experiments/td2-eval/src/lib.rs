// SPDX-License-Identifier: Apache-2.0
//! Independent COCO-style bounding-box detection evaluator for TD2 evidence.
//!
//! Written from scratch to reproduce the bbox behaviour of the reference
//! COCO evaluator (pycocotools), including its tie-breaking and edge-case
//! constants, and checked against frozen pycocotools 2.0.11 vectors. The
//! metric is defined at <https://cocodataset.org/#detection-eval>: average precision over IoU
//! thresholds 0.50:0.05:0.95, 101-point interpolated precision, at most
//! `max_detections` per image and class, crowd regions ignored, and
//! small/medium/large area ranges. Development tooling only: never linked into
//! the deployed library, no I/O, no dependencies.
//!
//! Known difference from the reference evaluator: object area for the size
//! ranges is the ground-truth *box* area, because TD2 data carries boxes only.

use std::collections::BTreeMap;

/// Axis-aligned box in pixels: left, top, right, bottom edges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxXyxy {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

impl BoxXyxy {
    fn area(self) -> f64 {
        (self.right - self.left) * (self.bottom - self.top)
    }

    fn validate(self) -> Result<(), String> {
        let edges = [self.left, self.top, self.right, self.bottom];
        if edges.iter().any(|edge| !edge.is_finite()) {
            return Err("box edge is not finite".to_owned());
        }
        if self.right < self.left || self.bottom < self.top {
            return Err("box is inverted".to_owned());
        }
        if !self.area().is_finite() {
            return Err("box area is not finite".to_owned());
        }
        Ok(())
    }
}

/// One annotated object. `crowd` regions are neither required nor penalized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundTruth {
    pub image_id: u64,
    pub class_id: u32,
    pub bbox: BoxXyxy,
    pub crowd: bool,
}

/// One scored prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub image_id: u64,
    pub class_id: u32,
    pub bbox: BoxXyxy,
    pub score: f64,
}

/// Object size range, by area in square pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AreaRange {
    All,
    /// Area below 32².
    Small,
    /// Area from 32² to 96².
    Medium,
    /// Area above 96².
    Large,
}

impl AreaRange {
    pub const EVERY: [Self; 4] = [Self::All, Self::Small, Self::Medium, Self::Large];

    fn contains(self, area: f64) -> bool {
        let (low, high) = match self {
            Self::All => (0.0, 1e10),
            Self::Small => (0.0, 32.0 * 32.0),
            Self::Medium => (32.0 * 32.0, 96.0 * 96.0),
            Self::Large => (96.0 * 96.0, 1e10),
        };
        area >= low && area <= high
    }
}

/// The ten IoU thresholds 0.50, 0.55, ..., 0.95, spaced as `start + i * step`
/// with the end point exact, so threshold equality cases are reproducible.
pub fn iou_thresholds() -> [f64; 10] {
    let step = (0.95 - 0.5) / 9.0;
    std::array::from_fn(|index| {
        if index == 9 {
            0.95
        } else {
            0.5 + step * index as f64
        }
    })
}

/// Average precision for one class.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassResult {
    pub class_id: u32,
    /// Non-crowd ground truths of this class within the `All` area range.
    pub ground_truths: usize,
    /// AP averaged over the ten IoU thresholds, all areas.
    pub ap: f64,
    /// AP at IoU 0.50, all areas.
    pub ap50: f64,
}

/// Summary metrics. `None` means no class had ground truth in that range.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub map: Option<f64>,
    pub map50: Option<f64>,
    pub map75: Option<f64>,
    pub map_small: Option<f64>,
    pub map_medium: Option<f64>,
    pub map_large: Option<f64>,
    /// Classes with at least one non-crowd ground truth, ascending id.
    pub classes: Vec<ClassResult>,
}

/// Ground truths and input-ordered detections of one class in one image.
type Group = (Vec<GroundTruth>, Vec<(usize, Detection)>);

/// Evaluates `detections` against `ground_truths`. `max_detections` caps
/// detections per image and class; COCO uses 100, the value the frozen
/// reference vectors check. Classes are those present
/// in either list; a class without non-crowd ground truth is excluded from
/// the means. Rejects nonfinite or inverted boxes, boxes whose area overflows,
/// nonfinite scores and
/// `max_detections == 0`.
pub fn evaluate(
    ground_truths: &[GroundTruth],
    detections: &[Detection],
    max_detections: usize,
) -> Result<Summary, String> {
    if max_detections == 0 {
        return Err("max_detections must be positive".to_owned());
    }
    for (index, truth) in ground_truths.iter().enumerate() {
        truth
            .bbox
            .validate()
            .map_err(|error| format!("ground truth {index}: {error}"))?;
    }
    for (index, detection) in detections.iter().enumerate() {
        detection
            .bbox
            .validate()
            .map_err(|error| format!("detection {index}: {error}"))?;
        if !detection.score.is_finite() {
            return Err(format!("detection {index}: score is not finite"));
        }
    }
    // Group by (class, image), keeping input order for stable tie-breaking.
    let mut groups: BTreeMap<(u32, u64), Group> = BTreeMap::new();
    for truth in ground_truths {
        groups
            .entry((truth.class_id, truth.image_id))
            .or_default()
            .0
            .push(*truth);
    }
    for (order, detection) in detections.iter().enumerate() {
        groups
            .entry((detection.class_id, detection.image_id))
            .or_default()
            .1
            // `+ 0.0` maps -0.0 to 0.0 so equal scores tie as in the reference.
            .push((
                order,
                Detection {
                    score: detection.score + 0.0,
                    ..*detection
                },
            ));
    }
    let thresholds = iou_thresholds();
    // ap[class][area][threshold]; None when the class has no ground truth there.
    let mut table: BTreeMap<u32, [[Option<f64>; 10]; 4]> = BTreeMap::new();
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    let class_ids: Vec<u32> = {
        let mut ids: Vec<u32> = groups.keys().map(|(class, _)| *class).collect();
        ids.dedup();
        ids
    };
    for class_id in class_ids {
        let images: Vec<_> = groups
            .range((class_id, 0)..=(class_id, u64::MAX))
            .map(|(_, group)| group)
            .collect();
        let mut row = [[None; 10]; 4];
        for (area_index, area) in AreaRange::EVERY.into_iter().enumerate() {
            for (threshold_index, threshold) in thresholds.into_iter().enumerate() {
                let mut required = 0usize;
                // (score, true positive); ignored detections omitted.
                let mut scored: Vec<(f64, bool)> = Vec::new();
                for (truths, image_detections) in &images {
                    required += match_image(
                        truths,
                        image_detections,
                        max_detections,
                        area,
                        threshold,
                        &mut scored,
                    );
                }
                row[area_index][threshold_index] = average_precision(required, &mut scored);
                if area == AreaRange::All && threshold_index == 0 {
                    counts.insert(class_id, required);
                }
            }
        }
        table.insert(class_id, row);
    }
    let mean = |area: usize, selected: &[usize]| -> Option<f64> {
        let values: Vec<f64> = table
            .values()
            .flat_map(|row| selected.iter().filter_map(move |&index| row[area][index]))
            .collect();
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    let every: Vec<usize> = (0..10).collect();
    let classes = table
        .iter()
        .filter(|(_, row)| row[0][0].is_some())
        .map(|(&class_id, row)| ClassResult {
            class_id,
            ground_truths: counts[&class_id],
            ap: row[0].iter().flatten().sum::<f64>() / 10.0,
            ap50: row[0][0].unwrap_or(0.0),
        })
        .collect();
    Ok(Summary {
        map: mean(0, &every),
        map50: mean(0, &[0]),
        map75: mean(0, &[5]),
        map_small: mean(1, &every),
        map_medium: mean(2, &every),
        map_large: mean(3, &every),
        classes,
    })
}

/// Greedy matching for one image and class at one IoU threshold and area
/// range. Appends non-ignored detections to `scored` and returns the number of
/// required (non-ignored) ground truths.
fn match_image(
    truths: &[GroundTruth],
    detections: &[(usize, Detection)],
    max_detections: usize,
    area: AreaRange,
    threshold: f64,
    scored: &mut Vec<(f64, bool)>,
) -> usize {
    // Required ground truths first, then ignored ones (crowd or out of range).
    let mut ordered: Vec<(GroundTruth, bool)> = truths
        .iter()
        .map(|truth| (*truth, truth.crowd || !area.contains(truth.bbox.area())))
        .collect();
    ordered.sort_by_key(|(_, ignored)| *ignored);
    let required = ordered.iter().filter(|(_, ignored)| !ignored).count();
    let mut ranked: Vec<&(usize, Detection)> = detections.iter().collect();
    ranked.sort_by(|a, b| b.1.score.total_cmp(&a.1.score).then(a.0.cmp(&b.0)));
    ranked.truncate(max_detections);
    let mut taken = vec![false; ordered.len()];
    for (_, detection) in ranked {
        let mut best_iou = threshold.min(1.0 - 1e-10);
        let mut matched: Option<usize> = None;
        for (index, (truth, ignored)) in ordered.iter().enumerate() {
            if taken[index] && !truth.crowd {
                continue;
            }
            // Once a required match exists, ignored candidates cannot replace it.
            if let Some(current) = matched {
                if !ordered[current].1 && *ignored {
                    break;
                }
            }
            let overlap = iou(detection.bbox, truth.bbox, truth.crowd);
            if overlap < best_iou {
                continue;
            }
            best_iou = overlap;
            matched = Some(index);
        }
        match matched {
            Some(index) => {
                taken[index] = true;
                if !ordered[index].1 {
                    scored.push((detection.score, true));
                }
            }
            None if area.contains(detection.bbox.area()) => {
                scored.push((detection.score, false));
            }
            None => {}
        }
    }
    required
}

/// Intersection over union; for crowd regions, intersection over the
/// detection's own area.
fn iou(detection: BoxXyxy, truth: BoxXyxy, crowd: bool) -> f64 {
    let width = detection.right.min(truth.right) - detection.left.max(truth.left);
    let height = detection.bottom.min(truth.bottom) - detection.top.max(truth.top);
    if width <= 0.0 || height <= 0.0 {
        return 0.0;
    }
    let intersection = width * height;
    let union = if crowd {
        detection.area()
    } else {
        detection.area() + truth.area() - intersection
    };
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// 101-point interpolated average precision; `None` without ground truth.
fn average_precision(required: usize, scored: &mut [(f64, bool)]) -> Option<f64> {
    if required == 0 {
        return None;
    }
    // Stable: equal scores keep image order, then per-image rank.
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut recall = Vec::with_capacity(scored.len());
    let mut precision = Vec::with_capacity(scored.len());
    let (mut true_positives, mut false_positives) = (0.0f64, 0.0f64);
    for (_, positive) in scored.iter() {
        if *positive {
            true_positives += 1.0;
        } else {
            false_positives += 1.0;
        }
        recall.push(true_positives / required as f64);
        precision.push(true_positives / (true_positives + false_positives + f64::EPSILON));
    }
    // Precision envelope: best precision at any equal or higher recall.
    for index in (1..precision.len()).rev() {
        if precision[index] > precision[index - 1] {
            precision[index - 1] = precision[index];
        }
    }
    let mut total = 0.0;
    for step in 0..=100 {
        let level = if step == 100 { 1.0 } else { step as f64 * 0.01 };
        let position = recall.partition_point(|value| *value < level);
        if position < precision.len() {
            total += precision[position];
        }
    }
    Some(total / 101.0)
}
