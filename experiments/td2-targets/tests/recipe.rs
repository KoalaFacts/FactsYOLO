// SPDX-License-Identifier: Apache-2.0
//! Hand-computed cases for the frozen TD2 loss weighting and schedule (D2b2a).

use std::f64::consts::LN_2;

use td2_targets::recipe::{
    ADAMW_BETAS, ADAMW_EPSILON, BASE_LR, GRAD_CLIP_NORM, LevelOutputs, LossError, MIN_LR_FRACTION,
    MIN_TOTAL_STEPS, ScheduleError, W_BOX, W_CLS, W_OBJ, WEIGHT_DECAY, image_loss, lr_at,
    warmup_steps,
};
use td2_targets::{Assigned, GtBox, LevelTargets, assign, giou_loss, sigmoid_focal_loss};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

fn pos(class: u16, ltrb: [f64; 4]) -> Option<Assigned> {
    Some(Assigned {
        gt_index: 0,
        class,
        ltrb,
    })
}

fn level(width: usize, cells: Vec<Option<Assigned>>) -> LevelTargets {
    LevelTargets {
        stride: 8,
        width,
        height: cells.len() / width,
        cells,
    }
}

fn out<'a>(cls: &'a [f64], obj: &'a [f64], ltrb: &'a [f64]) -> LevelOutputs<'a> {
    LevelOutputs { cls, obj, ltrb }
}

#[test]
fn frozen_constants() {
    assert_eq!((W_CLS, W_OBJ, W_BOX), (1.0, 1.0, 5.0));
    assert_eq!(BASE_LR, 2e-3);
    assert_eq!(MIN_LR_FRACTION, 0.05);
    assert_eq!(MIN_TOTAL_STEPS, 20);
    assert_eq!(ADAMW_BETAS, (0.9, 0.999));
    assert_eq!(ADAMW_EPSILON, 1e-8);
    assert_eq!(WEIGHT_DECAY, 0.05);
    assert_eq!(GRAD_CLIP_NORM, 10.0);
}

#[test]
fn zero_logits_one_positive_matches_closed_form() {
    // Two cells, two classes; cell 0 is class 1 with a perfect box.
    let targets = [level(2, vec![pos(1, [1.0; 4]), None])];
    let cls = [0.0; 4];
    let obj = [0.0; 2];
    let ltrb = [1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0];
    let parts = image_loss(&targets, &[out(&cls, &obj, &ltrb)], 2).unwrap();
    // focal(0, 1) = 0.25 · 0.25 · ln 2, focal(0, 0) = 0.75 · 0.25 · ln 2.
    // Cell 0: 0.1875 + 0.0625; cell 1: 2 · 0.1875; sum 0.625 · ln 2.
    assert_eq!(parts.positives, 1);
    assert!(close(parts.cls, 0.625 * LN_2), "{parts:?}");
    assert!(close(parts.obj, 2.0 * LN_2), "{parts:?}");
    // The negative cell's box output is ignored.
    assert_eq!(parts.boxes, 0.0);
    assert!(close(parts.total, 2.625 * LN_2), "{parts:?}");
}

#[test]
fn box_term_is_weighted_and_normalized_by_positives() {
    // Two positives: one perfect box, one with GIoU loss 0.75.
    let targets = [level(2, vec![pos(0, [1.0; 4]), pos(0, [1.0; 4])])];
    let cls = [0.0; 2];
    let obj = [0.0; 2];
    let ltrb = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
    let parts = image_loss(&targets, &[out(&cls, &obj, &ltrb)], 1).unwrap();
    assert_eq!(parts.positives, 2);
    // inter 4, union 16, enclosing 16 -> 1 - 0.25 = 0.75; mean over 2 positives.
    assert!(close(parts.boxes, 0.375), "{parts:?}");
    let cls_sum = 2.0 * sigmoid_focal_loss(0.0, 1.0);
    assert!(close(parts.cls, cls_sum / 2.0));
    assert!(close(parts.obj, LN_2));
    assert!(close(
        parts.total,
        W_CLS * parts.cls + W_OBJ * parts.obj + W_BOX * parts.boxes
    ));
}

#[test]
fn no_positives_divides_by_one() {
    let targets = [level(1, vec![None])];
    let parts = image_loss(&targets, &[out(&[0.0], &[0.0], &[0.0; 4])], 1).unwrap();
    assert_eq!(parts.positives, 0);
    assert!(close(parts.cls, 0.1875 * LN_2));
    assert!(close(parts.obj, LN_2));
    assert_eq!(parts.boxes, 0.0);
}

#[test]
fn sums_run_over_every_level() {
    let targets = [
        level(1, vec![pos(0, [1.0; 4])]),
        level(1, vec![None]),
        level(1, vec![pos(0, [2.0; 4])]),
    ];
    let outputs = [
        out(&[0.0], &[0.0], &[1.0; 4]),
        out(&[0.0], &[0.0], &[0.0; 4]),
        out(&[0.0], &[0.0], &[1.0; 4]),
    ];
    let parts = image_loss(&targets, &outputs, 1).unwrap();
    assert_eq!(parts.positives, 2);
    assert!(close(parts.obj, 3.0 * LN_2 / 2.0));
    assert!(close(parts.boxes, 0.75 / 2.0));
}

#[test]
fn perfect_outputs_for_assigned_targets_have_zero_box_loss() {
    let boxes = [GtBox {
        left: 40.0,
        top: 60.0,
        right: 120.0,
        bottom: 150.0,
        class: 3,
    }];
    let targets = assign(320, 5, &boxes).unwrap();
    let mut cls = Vec::new();
    let mut obj = Vec::new();
    let mut ltrb = Vec::new();
    for t in &targets {
        let mut c = vec![-30.0; t.cells.len() * 5];
        let mut o = vec![-30.0; t.cells.len()];
        let mut d = vec![0.0; t.cells.len() * 4];
        for (i, cell) in t.cells.iter().enumerate() {
            if let Some(a) = cell {
                c[i * 5 + usize::from(a.class)] = 30.0;
                o[i] = 30.0;
                d[i * 4..i * 4 + 4].copy_from_slice(&a.ltrb);
            }
        }
        cls.push(c);
        obj.push(o);
        ltrb.push(d);
    }
    let outputs: Vec<_> = (0..targets.len())
        .map(|i| out(&cls[i], &obj[i], &ltrb[i]))
        .collect();
    let parts = image_loss(&targets, &outputs, 5).unwrap();
    assert!(parts.positives > 0);
    assert!(parts.boxes.abs() < 1e-12, "{parts:?}");
    assert!(parts.cls < 1e-9 && parts.obj < 1e-9, "{parts:?}");
}

#[test]
fn randomized_parts_are_consistent() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..200 {
        let cells: Vec<_> = (0..6)
            .map(|_| {
                (next() < 0.4).then(|| Assigned {
                    gt_index: 0,
                    class: (next() * 3.0) as u16,
                    ltrb: [
                        0.1 + next() * 4.0,
                        0.1 + next() * 4.0,
                        0.1 + next() * 4.0,
                        0.1 + next() * 4.0,
                    ],
                })
            })
            .collect();
        let targets = [level(3, cells.clone())];
        let cls: Vec<f64> = (0..18).map(|_| next() * 20.0 - 10.0).collect();
        let obj: Vec<f64> = (0..6).map(|_| next() * 20.0 - 10.0).collect();
        let ltrb: Vec<f64> = (0..24).map(|_| next() * 5.0).collect();
        let parts = image_loss(&targets, &[out(&cls, &obj, &ltrb)], 3).unwrap();
        // Independent recount from the D2a primitives.
        let npos = cells.iter().flatten().count();
        let mut c = 0.0;
        let mut o = 0.0;
        let mut b = 0.0;
        for (i, cell) in cells.iter().enumerate() {
            for k in 0..3 {
                let t = cell.is_some_and(|a| usize::from(a.class) == k);
                c += sigmoid_focal_loss(cls[i * 3 + k], f64::from(u8::from(t)));
            }
            o += td2_targets::bce_with_logits(obj[i], f64::from(u8::from(cell.is_some())));
            if let Some(a) = cell {
                let p = [
                    ltrb[i * 4],
                    ltrb[i * 4 + 1],
                    ltrb[i * 4 + 2],
                    ltrb[i * 4 + 3],
                ];
                b += giou_loss(p, a.ltrb).unwrap();
            }
        }
        let n = npos.max(1) as f64;
        assert_eq!(parts.positives, npos);
        assert!(close(parts.cls, c / n) && close(parts.obj, o / n) && close(parts.boxes, b / n));
        assert!(close(
            parts.total,
            W_CLS * parts.cls + W_OBJ * parts.obj + W_BOX * parts.boxes
        ));
        assert!(parts.total.is_finite() && parts.total >= 0.0);
    }
}

#[test]
fn rejects_mismatched_shapes() {
    let targets = [level(2, vec![pos(0, [1.0; 4]), None])];
    let ok_cls = [0.0; 2];
    let ok_obj = [0.0; 2];
    let ok_ltrb = [1.0; 8];
    assert_eq!(
        image_loss(&targets, &[], 1),
        Err(LossError::LevelCount {
            targets: 1,
            outputs: 0
        })
    );
    assert_eq!(
        image_loss(&targets, &[out(&ok_cls[..1], &ok_obj, &ok_ltrb)], 1),
        Err(LossError::Shape { level: 0 })
    );
    assert_eq!(
        image_loss(&targets, &[out(&ok_cls, &ok_obj[..1], &ok_ltrb)], 1),
        Err(LossError::Shape { level: 0 })
    );
    assert_eq!(
        image_loss(&targets, &[out(&ok_cls, &ok_obj, &ok_ltrb[..7])], 1),
        Err(LossError::Shape { level: 0 })
    );
    assert_eq!(
        image_loss(&targets, &[out(&[], &[], &[])], 0),
        Err(LossError::ClassCount)
    );
    // A cell list that does not fill width × height is malformed.
    let ragged = [LevelTargets {
        stride: 8,
        width: 2,
        height: 2,
        cells: vec![None, None],
    }];
    assert_eq!(
        image_loss(&ragged, &[out(&ok_cls, &ok_obj, &ok_ltrb)], 1),
        Err(LossError::Shape { level: 0 })
    );
    // A target class outside the class count.
    let bad_class = [level(2, vec![pos(1, [1.0; 4]), None])];
    assert_eq!(
        image_loss(&bad_class, &[out(&ok_cls, &ok_obj, &ok_ltrb)], 1),
        Err(LossError::TargetClass { level: 0, cell: 0 })
    );
}

#[test]
fn rejects_nonfinite_outputs_and_invalid_boxes() {
    let targets = [level(2, vec![pos(0, [1.0; 4]), None])];
    let obj = [0.0; 2];
    let ltrb = [1.0; 8];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            image_loss(&targets, &[out(&[0.0, bad], &obj, &ltrb)], 1),
            Err(LossError::NonFinite { level: 0, cell: 1 })
        );
        assert_eq!(
            image_loss(&targets, &[out(&[0.0; 2], &[bad, 0.0], &ltrb)], 1),
            Err(LossError::NonFinite { level: 0, cell: 0 })
        );
        // Box outputs of negative cells are still checked: a NaN anywhere
        // in the tensor is a broken forward pass.
        let mut l = ltrb;
        l[5] = bad;
        assert_eq!(
            image_loss(&targets, &[out(&[0.0; 2], &obj, &l)], 1),
            Err(LossError::NonFinite { level: 0, cell: 1 })
        );
    }
    let mut negative = ltrb;
    negative[2] = -0.5;
    assert_eq!(
        image_loss(&targets, &[out(&[0.0; 2], &obj, &negative)], 1),
        Err(LossError::Box { level: 0, cell: 0 })
    );
    // Finite distances whose geometry overflows F64.
    let mut huge = ltrb;
    huge[..4].copy_from_slice(&[f64::MAX; 4]);
    assert_eq!(
        image_loss(&targets, &[out(&[0.0; 2], &obj, &huge)], 1),
        Err(LossError::Box { level: 0, cell: 0 })
    );
}

#[test]
fn warmup_is_one_twentieth_with_a_floor_of_one() {
    assert_eq!(warmup_steps(20), Ok(1));
    assert_eq!(warmup_steps(39), Ok(1));
    assert_eq!(warmup_steps(40), Ok(2));
    assert_eq!(warmup_steps(1000), Ok(50));
    assert_eq!(warmup_steps(19), Err(ScheduleError::TotalSteps(19)));
    assert_eq!(warmup_steps(0), Err(ScheduleError::TotalSteps(0)));
}

#[test]
fn schedule_hand_points() {
    let total = 1000;
    assert!(close(lr_at(0, total).unwrap(), BASE_LR / 50.0));
    assert!(close(lr_at(24, total).unwrap(), BASE_LR / 2.0));
    assert!(close(lr_at(49, total).unwrap(), BASE_LR));
    // Cosine starts at the base rate right after warmup ...
    assert!(close(lr_at(50, total).unwrap(), BASE_LR));
    // ... is halfway between base and floor at the midpoint of the decay ...
    let mid = 50 + (total - 1 - 50) / 2;
    let floor = BASE_LR * MIN_LR_FRACTION;
    let expected_mid =
        floor + (BASE_LR - floor) * 0.5 * (1.0 + (std::f64::consts::PI * 474.0 / 949.0).cos());
    assert_eq!(mid, 524);
    assert!(close(lr_at(mid, total).unwrap(), expected_mid));
    // ... and ends exactly on the floor at the last step.
    assert!(close(lr_at(total - 1, total).unwrap(), floor));
}

#[test]
fn schedule_shape_holds_for_many_lengths() {
    for total in [20, 21, 39, 40, 41, 100, 777, 10_000] {
        let w = warmup_steps(total).unwrap();
        let lrs: Vec<f64> = (0..total).map(|s| lr_at(s, total).unwrap()).collect();
        assert!(
            lrs.iter()
                .all(|&v| v.is_finite() && v > 0.0 && v <= BASE_LR)
        );
        assert!(lrs[..w as usize].windows(2).all(|p| p[0] < p[1]), "{total}");
        assert!(
            lrs[w as usize..].windows(2).all(|p| p[0] >= p[1]),
            "{total}"
        );
        assert!(close(lrs[w as usize - 1], BASE_LR));
        assert!(close(lrs[total as usize - 1], BASE_LR * MIN_LR_FRACTION));
    }
}

#[test]
fn schedule_rejects_out_of_range_steps() {
    assert_eq!(
        lr_at(100, 100),
        Err(ScheduleError::Step {
            step: 100,
            total: 100
        })
    );
    assert_eq!(
        lr_at(u32::MAX, 100),
        Err(ScheduleError::Step {
            step: u32::MAX,
            total: 100
        })
    );
    assert_eq!(lr_at(0, 5), Err(ScheduleError::TotalSteps(5)));
    // The largest length is accepted without overflow.
    assert!(close(
        lr_at(u32::MAX - 1, u32::MAX).unwrap(),
        BASE_LR * MIN_LR_FRACTION
    ));
}

#[test]
fn rejects_finite_outputs_whose_sums_overflow() {
    // Each term is finite, but the sums or the weighted total are not.
    let targets = [level(3, vec![None, None, None])];
    let big = [1e308; 3];
    assert_eq!(
        image_loss(&targets, &[out(&big, &big, &[0.0; 12])], 1),
        Err(LossError::Overflow)
    );
    let one = [level(1, vec![None])];
    assert_eq!(
        image_loss(&one, &[out(&[1e308], &[1.7e308], &[0.0; 4])], 1),
        Err(LossError::Overflow)
    );
}
