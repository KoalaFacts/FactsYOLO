// SPDX-License-Identifier: Apache-2.0
//! Frozen loss weights and per-batch normalization (stage D2b2a).

use td2_targets::{
    FOCAL_ALPHA, FOCAL_GAMMA, GtBox, LevelPrediction, LevelTargets, LossError, LossParts,
    WEIGHT_BOX, WEIGHT_CLASS, WEIGHT_OBJECTNESS, assign, total_loss,
};

const LN2: f64 = std::f64::consts::LN_2;

fn gt(left: f64, top: f64, right: f64, bottom: f64, class: u16) -> GtBox {
    GtBox {
        left,
        top,
        right,
        bottom,
        class,
    }
}

/// All-zero logits; box predictions equal to their targets (zero elsewhere).
fn neutral(levels: &[LevelTargets], classes: usize) -> Vec<LevelPrediction> {
    levels
        .iter()
        .map(|l| LevelPrediction {
            class_logits: vec![0.0; l.cells.len() * classes],
            objectness: vec![0.0; l.cells.len()],
            boxes: l
                .cells
                .iter()
                .map(|c| c.map_or([0.0; 4], |a| a.ltrb))
                .collect(),
        })
        .collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

#[test]
fn weights_are_frozen() {
    assert_eq!(
        (WEIGHT_CLASS, WEIGHT_OBJECTNESS, WEIGHT_BOX),
        (1.0, 1.0, 5.0)
    );
}

#[test]
fn zero_logits_match_the_closed_form() {
    // S = 64: 8×8 + 4×4 + 2×2 = 84 locations.
    let classes = 3usize;
    let levels = assign(64, classes as u16, &[gt(10.0, 12.0, 40.0, 44.0, 2)]).unwrap();
    let cells: usize = levels.iter().map(|l| l.cells.len()).sum();
    assert_eq!(cells, 84);
    let pos = levels
        .iter()
        .flat_map(|l| &l.cells)
        .filter(|c| c.is_some())
        .count();
    assert!(pos > 0);
    let parts = total_loss(&[(&levels, &neutral(&levels, classes))], classes).unwrap();
    // focal(0, 0) = (1 − α)·0.5^γ·ln2 and focal(0, 1) = α·0.5^γ·ln2.
    let quarter = 0.5f64.powf(FOCAL_GAMMA);
    let cls_sum = ((cells * classes - pos) as f64) * (1.0 - FOCAL_ALPHA) * quarter * LN2
        + pos as f64 * FOCAL_ALPHA * quarter * LN2;
    let n = pos as f64;
    assert_eq!(parts.positives, pos);
    assert!(close(parts.class, cls_sum / n), "{parts:?}");
    assert!(close(parts.objectness, cells as f64 * LN2 / n));
    assert_eq!(parts.boxes, 0.0);
    let total = WEIGHT_CLASS * parts.class + WEIGHT_OBJECTNESS * parts.objectness;
    assert!(close(parts.total, total));
}

#[test]
fn no_positives_normalizes_by_one() {
    let classes = 2usize;
    let levels = assign(32, classes as u16, &[]).unwrap();
    let parts = total_loss(&[(&levels, &neutral(&levels, classes))], classes).unwrap();
    let cells = (16 + 4 + 1) as f64;
    assert_eq!(parts.positives, 0);
    assert!(close(parts.objectness, cells * LN2));
    assert!(close(
        parts.class,
        cells * classes as f64 * (1.0 - FOCAL_ALPHA) * 0.25 * LN2
    ));
    assert_eq!(parts.boxes, 0.0);
}

#[test]
fn box_term_is_weighted_mean_giou_over_positives() {
    let classes = 1usize;
    let levels = assign(64, 1, &[gt(8.0, 8.0, 56.0, 56.0, 0)]).unwrap();
    let mut preds = neutral(&levels, classes);
    let mut pos = 0usize;
    for (p, l) in preds.iter_mut().zip(&levels) {
        for (b, c) in p.boxes.iter_mut().zip(&l.cells) {
            if let Some(a) = c {
                // Same left/top/bottom, width doubled: the prediction
                // contains the target and is the enclosing box, so
                // 1 − GIoU = 1 − IoU = 1/2.
                let [l0, t0, r0, b0] = a.ltrb;
                *b = [l0, t0, l0 + 2.0 * r0, b0];
                pos += 1;
            }
        }
    }
    assert!(pos > 0);
    let parts = total_loss(&[(&levels, &preds)], classes).unwrap();
    let mut want = 0.0;
    for (p, l) in preds.iter().zip(&levels) {
        for (b, c) in p.boxes.iter().zip(&l.cells) {
            if let Some(a) = c {
                let w = a.ltrb[0] + a.ltrb[2];
                let pw = b[0] + b[2];
                let iou = w / pw;
                want += 1.0 - iou;
            }
        }
    }
    assert!(
        close(parts.boxes, want / pos as f64),
        "{} vs {}",
        parts.boxes,
        want / pos as f64
    );
    assert!(close(
        parts.total,
        parts.class + parts.objectness + WEIGHT_BOX * parts.boxes
    ));
}

#[test]
fn normalization_is_per_batch_not_per_image() {
    let classes = 2usize;
    let a = assign(64, 2, &[gt(4.0, 4.0, 30.0, 28.0, 1)]).unwrap();
    let b = assign(64, 2, &[]).unwrap();
    let pa = neutral(&a, classes);
    let pb = neutral(&b, classes);
    let one = total_loss(&[(&a, &pa)], classes).unwrap();
    let two = total_loss(&[(&a, &pa), (&a, &pa)], classes).unwrap();
    let mixed = total_loss(&[(&a, &pa), (&b, &pb)], classes).unwrap();
    assert_eq!(two.positives, 2 * one.positives);
    assert!(close(two.total, one.total));
    // An empty image adds negatives but no positives to the same divisor.
    assert_eq!(mixed.positives, one.positives);
    assert!(mixed.objectness > one.objectness);
    let alone = total_loss(&[(&b, &pb)], classes).unwrap();
    let sum = one.objectness * one.positives as f64 + alone.objectness;
    assert!(close(mixed.objectness, sum / one.positives as f64));
}

#[test]
fn rejects_malformed_predictions() {
    let classes = 2usize;
    let levels = assign(64, 2, &[gt(4.0, 4.0, 30.0, 28.0, 1)]).unwrap();
    let good = neutral(&levels, classes);
    let run = |p: &Vec<LevelPrediction>, c: usize| -> Result<LossParts, LossError> {
        total_loss(&[(&levels, p)], c)
    };
    assert!(run(&good, classes).is_ok());
    assert_eq!(total_loss(&[], classes), Err(LossError::EmptyBatch));
    assert_eq!(run(&good, 0), Err(LossError::ClassCount));
    assert_eq!(run(&good, 3), Err(LossError::Shape { image: 0, level: 0 }));
    let mut fewer = neutral(&levels, classes);
    fewer.pop();
    assert_eq!(run(&fewer, classes), Err(LossError::Levels { image: 0 }));
    let mut short = neutral(&levels, classes);
    short[1].objectness.pop();
    assert_eq!(
        run(&short, classes),
        Err(LossError::Shape { image: 0, level: 1 })
    );
    let mut nan = neutral(&levels, classes);
    nan[2].class_logits[0] = f64::NAN;
    assert_eq!(
        run(&nan, classes),
        Err(LossError::NonFinite { image: 0, level: 2 })
    );
    let mut inf = neutral(&levels, classes);
    inf[0].objectness[3] = f64::INFINITY;
    assert_eq!(
        run(&inf, classes),
        Err(LossError::NonFinite { image: 0, level: 0 })
    );
    // A negative distance at a positive location has no GIoU.
    let (level, cell) = levels
        .iter()
        .enumerate()
        .find_map(|(i, l)| l.cells.iter().position(|c| c.is_some()).map(|c| (i, c)))
        .unwrap();
    let mut neg = neutral(&levels, classes);
    neg[level].boxes[cell][0] = -1.0;
    assert_eq!(
        run(&neg, classes),
        Err(LossError::InvalidBox {
            image: 0,
            level,
            cell
        })
    );
    // Box values at negative locations are ignored, even when nonfinite.
    let mut ignored = neutral(&levels, classes);
    let free = levels[0].cells.iter().position(|c| c.is_none()).unwrap();
    ignored[0].boxes[free] = [f64::NAN; 4];
    assert_eq!(run(&ignored, classes), run(&good, classes));
    // A target class beyond the prediction's class count is an error, not a
    // silently all-negative location.
    let wide = assign(64, 5, &[gt(4.0, 4.0, 30.0, 28.0, 4)]).unwrap();
    let narrow = neutral(&wide, classes);
    assert!(matches!(
        total_loss(&[(&wide, &narrow)], classes),
        Err(LossError::ClassOutOfRange { image: 0, .. })
    ));
}

#[test]
fn objectness_targets_are_one_at_positives_only() {
    let classes = 1usize;
    let levels = assign(64, 1, &[gt(10.0, 12.0, 40.0, 44.0, 0)]).unwrap();
    let mut preds = neutral(&levels, classes);
    for p in &mut preds {
        p.objectness.iter_mut().for_each(|v| *v = 2.0);
    }
    let parts = total_loss(&[(&levels, &preds)], classes).unwrap();
    let cells = 84.0;
    let pos = parts.positives as f64;
    // BCE(2, 1) = ln(1 + e^-2); BCE(2, 0) = 2 + ln(1 + e^-2).
    let soft = (-2.0f64).exp().ln_1p();
    let want = (pos * soft + (cells - pos) * (2.0 + soft)) / pos;
    assert!(
        close(parts.objectness, want),
        "{} vs {want}",
        parts.objectness
    );
}

#[test]
fn rejects_short_box_vectors() {
    let levels = assign(64, 1, &[gt(10.0, 12.0, 40.0, 44.0, 0)]).unwrap();
    for level in 0..levels.len() {
        let mut preds = neutral(&levels, 1);
        preds[level].boxes.pop();
        assert_eq!(
            total_loss(&[(&levels, &preds)], 1),
            Err(LossError::Shape { image: 0, level })
        );
    }
}

#[test]
fn rejects_finite_logits_whose_loss_sum_overflows() {
    // Each logit is finite and each per-element BCE (≈ 1e308) is finite,
    // but their sum is not: the batch loss must be an error, never ∞.
    let classes = 1usize;
    let levels = assign(64, 1, &[]).unwrap();
    let mut preds = neutral(&levels, classes);
    preds[0].objectness[0] = f64::MAX;
    preds[0].objectness[1] = f64::MAX;
    assert_eq!(
        total_loss(&[(&levels, &preds)], classes),
        Err(LossError::Overflow)
    );
    // The same for the class term.
    let mut preds = neutral(&levels, classes);
    preds[0].class_logits[0] = f64::MAX;
    preds[0].class_logits[1] = f64::MAX;
    assert_eq!(
        total_loss(&[(&levels, &preds)], classes),
        Err(LossError::Overflow)
    );
    // One huge logit alone is still a finite, valid loss.
    let mut preds = neutral(&levels, classes);
    preds[0].objectness[0] = 1e300;
    let parts = total_loss(&[(&levels, &preds)], classes).unwrap();
    assert!(parts.total.is_finite() && parts.objectness >= 1e300);
}
