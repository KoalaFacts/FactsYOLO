// SPDX-License-Identifier: Apache-2.0
//! Hand-computed cases for the frozen TD2 target-assignment rule and losses.

use td2_targets::{
    CENTER_RADIUS, FOCAL_ALPHA, FOCAL_GAMMA, GtBox, STRIDES, TargetError, assign, bce_with_logits,
    giou_loss, sigmoid_focal_loss,
};

fn gt(left: f64, top: f64, right: f64, bottom: f64, class: u16) -> GtBox {
    GtBox {
        left,
        top,
        right,
        bottom,
        class,
    }
}

fn positives(levels: &[td2_targets::LevelTargets], level: usize) -> usize {
    levels[level].cells.iter().filter(|c| c.is_some()).count()
}

fn cell(
    levels: &[td2_targets::LevelTargets],
    level: usize,
    x: f64,
    y: f64,
) -> Option<td2_targets::Assigned> {
    let l = &levels[level];
    let s = f64::from(l.stride);
    let col = ((x - s / 2.0) / s) as usize;
    let row = ((y - s / 2.0) / s) as usize;
    l.cells[row * l.width + col]
}

#[test]
fn frozen_constants() {
    assert_eq!(STRIDES, [8, 16, 32]);
    assert_eq!(CENTER_RADIUS, 2.5);
    assert_eq!(FOCAL_ALPHA, 0.25);
    assert_eq!(FOCAL_GAMMA, 2.0);
}

#[test]
fn grid_shapes_follow_input_size() {
    let levels = assign(320, 3, &[]).unwrap();
    let shapes: Vec<_> = levels
        .iter()
        .map(|l| (l.stride, l.width, l.height))
        .collect();
    assert_eq!(shapes, vec![(8, 40, 40), (16, 20, 20), (32, 10, 10)]);
    assert!(levels.iter().all(|l| l.cells.iter().all(Option::is_none)));
}

#[test]
fn small_box_lands_on_stride_8_centre_region() {
    // 40x40 box centred at (100, 100): centre radius 2.5 * 8 = 20 keeps
    // cell centres 84..116 on each axis, all inside the box: 5 x 5.
    let levels = assign(320, 3, &[gt(80.0, 80.0, 120.0, 120.0, 2)]).unwrap();
    assert_eq!(positives(&levels, 0), 25);
    assert_eq!(positives(&levels, 1), 0);
    assert_eq!(positives(&levels, 2), 0);
    let c = cell(&levels, 0, 100.0, 100.0).unwrap();
    assert_eq!(c.gt_index, 0);
    assert_eq!(c.class, 2);
    assert_eq!(c.ltrb, [2.5, 2.5, 2.5, 2.5]);
    let corner = cell(&levels, 0, 84.0, 116.0).unwrap();
    assert_eq!(corner.ltrb, [0.5, 4.5, 4.5, 0.5]);
    assert!(cell(&levels, 0, 76.0, 100.0).is_none());
}

#[test]
fn scale_range_is_per_location() {
    // 200x200 box at 60..260. Stride 16 centre cells 136..184 have max
    // distance <= 124 (range (64, 128]); stride 32 cells whose max distance
    // exceeds 128 are positive there; stride 8 needs max <= 64: none.
    let levels = assign(320, 1, &[gt(60.0, 60.0, 260.0, 260.0, 0)]).unwrap();
    assert_eq!(positives(&levels, 0), 0);
    assert_eq!(positives(&levels, 1), 16);
    assert_eq!(positives(&levels, 2), 12);
    let c = cell(&levels, 2, 112.0, 112.0).unwrap();
    assert_eq!(
        c.ltrb,
        [52.0 / 32.0, 52.0 / 32.0, 148.0 / 32.0, 148.0 / 32.0]
    );
    assert!(cell(&levels, 2, 144.0, 144.0).is_none());
}

#[test]
fn ambiguous_location_goes_to_smaller_box() {
    let boxes = [
        gt(80.0, 80.0, 120.0, 120.0, 0),
        gt(90.0, 90.0, 110.0, 110.0, 1),
    ];
    let levels = assign(320, 2, &boxes).unwrap();
    assert_eq!(cell(&levels, 0, 100.0, 100.0).unwrap().gt_index, 1);
    assert_eq!(cell(&levels, 0, 84.0, 84.0).unwrap().gt_index, 0);
}

#[test]
fn equal_area_tie_goes_to_lower_index() {
    let b = gt(80.0, 80.0, 120.0, 120.0, 0);
    let levels = assign(320, 1, &[b, b]).unwrap();
    assert!(levels[0].cells.iter().flatten().all(|c| c.gt_index == 0));
    assert_eq!(positives(&levels, 0), 25);
}

#[test]
fn box_outside_the_frame_assigns_nothing() {
    let levels = assign(320, 1, &[gt(400.0, 400.0, 440.0, 440.0, 0)]).unwrap();
    assert!(levels.iter().all(|l| l.cells.iter().all(Option::is_none)));
}

#[test]
fn rejects_invalid_inputs() {
    assert_eq!(assign(0, 1, &[]).unwrap_err(), TargetError::InputSize(0));
    assert_eq!(
        assign(330, 1, &[]).unwrap_err(),
        TargetError::InputSize(330)
    );
    assert_eq!(assign(320, 0, &[]).unwrap_err(), TargetError::ClassCount);
    for bad in [
        gt(f64::NAN, 0.0, 10.0, 10.0, 0),
        gt(0.0, 0.0, f64::INFINITY, 10.0, 0),
        gt(10.0, 0.0, 10.0, 10.0, 0),
        gt(0.0, 10.0, 10.0, 5.0, 0),
    ] {
        assert_eq!(
            assign(320, 1, &[gt(0.0, 0.0, 8.0, 8.0, 0), bad]).unwrap_err(),
            TargetError::InvalidBox(1)
        );
    }
    assert_eq!(
        assign(320, 3, &[gt(0.0, 0.0, 8.0, 8.0, 3)]).unwrap_err(),
        TargetError::ClassOutOfRange { index: 0, class: 3 }
    );
}

#[test]
fn target_limits_match_t1_artifact_contract() {
    // Both were accepted by the old oracle, despite being impossible to encode
    // as an FYTDv002 T1 header. Check before constructing any target grid.
    assert_eq!(
        assign(1056, 1, &[]).unwrap_err(),
        TargetError::InputSize(1056)
    );
    assert_eq!(assign(32, 121, &[]).unwrap_err(), TargetError::ClassCount);

    // The inclusive upper boundaries must remain usable.
    let levels = assign(1024, 1, &[]).unwrap();
    assert_eq!(
        levels.iter().map(|l| l.width).collect::<Vec<_>>(),
        vec![128, 64, 32]
    );
    assert_eq!(assign(32, 120, &[]).unwrap().len(), STRIDES.len());
}

#[test]
fn rejects_finite_box_endpoints_with_unrepresentable_geometry() {
    // Previously these boxes were accepted despite an infinite computed area.
    // A box centred on the image then silently had zero assigned positives,
    // because its area could never beat the initial best_area = infinity.
    for bad in [
        gt(-1e200, -1e200, 1e200, 1e200, 0), // finite spans, area overflows
        gt(-f64::MAX, -1.0, f64::MAX, 1.0, 0), // width itself overflows
        gt(80.0, 80.0, 1e200, 1e200, 0),     // finite endpoints, area overflows
    ] {
        assert_eq!(
            assign(320, 1, &[bad]).unwrap_err(),
            TargetError::InvalidBox(0),
            "{bad:?}"
        );
    }
    // A normal finite positive-area box must retain its existing targets.
    let valid = assign(320, 1, &[gt(80.0, 80.0, 120.0, 120.0, 0)]).unwrap();
    assert_eq!(positives(&valid, 0), 25);
}

#[test]
fn bce_is_stable_at_extremes() {
    assert!((bce_with_logits(0.0, 1.0) - std::f64::consts::LN_2).abs() < 1e-15);
    assert!((bce_with_logits(0.0, 0.0) - std::f64::consts::LN_2).abs() < 1e-15);
    assert!(bce_with_logits(100.0, 1.0) < 1e-40);
    assert!((bce_with_logits(-1000.0, 1.0) - 1000.0).abs() < 1e-9);
    assert!((bce_with_logits(1000.0, 0.0) - 1000.0).abs() < 1e-9);
}

#[test]
fn focal_matches_closed_form() {
    for &(x, t) in &[(0.3_f64, 1.0), (-1.7, 0.0), (2.2, 0.0), (-0.4, 1.0)] {
        let p: f64 = 1.0 / (1.0 + (-x).exp());
        let pt = if t == 1.0 { p } else { 1.0 - p };
        let at = if t == 1.0 { 0.25 } else { 0.75 };
        let expected = -at * (1.0 - pt).powi(2) * pt.ln();
        assert!(
            (sigmoid_focal_loss(x, t) - expected).abs() < 1e-14,
            "x={x} t={t}"
        );
    }
    assert!(sigmoid_focal_loss(-1000.0, 1.0).is_finite());
    assert!(sigmoid_focal_loss(1000.0, 1.0) < 1e-300);
}

#[test]
fn giou_hand_cases() {
    assert_eq!(
        giou_loss([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
        Some(0.0)
    );
    // Nested: IoU 4/16, enclosing box equals the union: loss 0.75.
    assert_eq!(giou_loss([1.0; 4], [2.0; 4]), Some(0.75));
    // Touching halves: IoU 0, enclosing area equals union area: loss 1.
    assert_eq!(
        giou_loss([2.0, 1.0, 0.0, 1.0], [0.0, 1.0, 2.0, 1.0]),
        Some(1.0)
    );
    // Corner-touching: IoU 0, union 2, enclosing 4: GIoU -0.5, loss 1.5.
    assert_eq!(
        giou_loss([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]),
        Some(1.5)
    );
    assert_eq!(giou_loss([-1.0, 1.0, 1.0, 1.0], [1.0; 4]), None);
    assert_eq!(giou_loss([f64::NAN, 1.0, 1.0, 1.0], [1.0; 4]), None);
    assert_eq!(giou_loss([0.0; 4], [0.0; 4]), None);
}

#[test]
fn giou_rejects_finite_inputs_that_overflow_intermediate_geometry() {
    // Inputs are finite; their intermediate areas do not fit in F64.
    // The old implementation returned Some(NaN), including for 1e160.
    for magnitude in [1e154, 1e160, 1e200, f64::MAX] {
        let huge = [magnitude; 4];
        assert_eq!(giou_loss(huge, [1.0; 4]), None, "magnitude={magnitude}");
        assert_eq!(giou_loss([1.0; 4], huge), None, "magnitude={magnitude}");
    }
    // Individual areas are finite, but their sum overflows.
    let near_limit = [5e153; 4];
    assert_eq!(giou_loss(near_limit, near_limit), None);
    // Individual areas and their union are finite; only the enclosing box
    // overflows. Without the enclosure guard this returned Some(NaN).
    assert_eq!(
        giou_loss([1e154, 1.0, 1e154, 1.0], [1.0, 1e154, 1.0, 1e154]),
        None
    );
    // Large-but-representable geometry remains valid.
    assert_eq!(giou_loss([1e153; 4], [1.0; 4]), Some(1.0));
}

#[test]
fn randomized_invariants_hold() {
    // Deterministic LCG; original synthetic boxes only.
    let mut state: u64 = 0x0005_eed0_f7d2;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..300 {
        let n = (next() * 6.0) as usize;
        let boxes: Vec<GtBox> = (0..n)
            .map(|i| {
                let (w, h) = (2.0 + next() * 300.0, 2.0 + next() * 300.0);
                let (x, y) = (next() * 360.0 - 40.0, next() * 360.0 - 40.0);
                gt(x, y, x + w, y + h, (i % 4) as u16)
            })
            .collect();
        let levels = assign(320, 4, &boxes).unwrap();
        // Completeness: a location is positive exactly when some box is eligible.
        for (k, level) in levels.iter().enumerate() {
            let s = f64::from(level.stride);
            let (lo, hi) = td2_targets::SCALE_RANGES[k];
            for (at, c) in level.cells.iter().enumerate() {
                let x = ((at % level.width) as f64 + 0.5) * s;
                let y = ((at / level.width) as f64 + 0.5) * s;
                let any = boxes.iter().any(|o| {
                    let d = [x - o.left, y - o.top, o.right - x, o.bottom - y];
                    let mo = d.iter().copied().fold(f64::MIN, f64::max);
                    d.iter().all(|v| *v > 0.0)
                        && (x - (o.left + o.right) / 2.0).abs() < CENTER_RADIUS * s
                        && (y - (o.top + o.bottom) / 2.0).abs() < CENTER_RADIUS * s
                        && mo > lo
                        && mo <= hi
                });
                assert_eq!(c.is_some(), any, "level {k} cell {at}");
            }
        }
        for (k, level) in levels.iter().enumerate() {
            let s = f64::from(level.stride);
            let (lo, hi) = td2_targets::SCALE_RANGES[k];
            for (at, c) in level.cells.iter().enumerate() {
                let Some(c) = c else { continue };
                let x = ((at % level.width) as f64 + 0.5) * s;
                let y = ((at / level.width) as f64 + 0.5) * s;
                let b = boxes[c.gt_index];
                assert_eq!(c.class, b.class);
                let px = c.ltrb.map(|d| d * s);
                assert!((x - px[0] - b.left).abs() < 1e-9 && (x + px[2] - b.right).abs() < 1e-9);
                assert!((y - px[1] - b.top).abs() < 1e-9 && (y + px[3] - b.bottom).abs() < 1e-9);
                assert!(px.iter().all(|d| *d > 0.0));
                let m = px.iter().copied().fold(0.0, f64::max);
                assert!(m > lo && m <= hi);
                assert!((x - (b.left + b.right) / 2.0).abs() < CENTER_RADIUS * s);
                // No other eligible box at this location is strictly smaller.
                let area = |g: &GtBox| (g.right - g.left) * (g.bottom - g.top);
                for (j, o) in boxes.iter().enumerate() {
                    let d = [x - o.left, y - o.top, o.right - x, o.bottom - y];
                    let mo = d.iter().copied().fold(f64::MIN, f64::max);
                    let eligible = d.iter().all(|v| *v > 0.0)
                        && (x - (o.left + o.right) / 2.0).abs() < CENTER_RADIUS * s
                        && (y - (o.top + o.bottom) / 2.0).abs() < CENTER_RADIUS * s
                        && mo > lo
                        && mo <= hi;
                    if eligible {
                        assert!(area(o) > area(&b) || (area(o) == area(&b) && j >= c.gt_index));
                    }
                }
            }
        }
    }
}

#[test]
fn scale_range_lower_bound_is_exclusive() {
    // At the stride-16 centre (104, 104) of box 40..168 every distance is
    // exactly 64: outside (64, 128], and 104 is not a stride-8 centre.
    let levels = assign(320, 1, &[gt(40.0, 40.0, 168.0, 168.0, 0)]).unwrap();
    assert!(cell(&levels, 1, 104.0, 104.0).is_none());
    assert_eq!(
        cell(&levels, 1, 120.0, 104.0).unwrap().ltrb,
        [5.0, 4.0, 3.0, 4.0]
    );
}

#[test]
fn scale_range_upper_bound_is_inclusive() {
    // Distances exactly 64 at a stride-8 centre stay on stride 8.
    let levels = assign(320, 1, &[gt(36.0, 36.0, 164.0, 164.0, 0)]).unwrap();
    assert_eq!(cell(&levels, 0, 100.0, 100.0).unwrap().ltrb, [8.0; 4]);
    // Distances exactly 128 at a stride-16 centre stay on stride 16.
    let levels = assign(320, 1, &[gt(-24.0, -24.0, 232.0, 232.0, 0)]).unwrap();
    assert_eq!(cell(&levels, 1, 104.0, 104.0).unwrap().ltrb, [8.0; 4]);
}

#[test]
fn location_on_a_box_edge_is_not_inside() {
    // Left and top edges pass through the stride-8 centres at 100.
    let levels = assign(320, 1, &[gt(100.0, 100.0, 120.0, 120.0, 0)]).unwrap();
    assert!(cell(&levels, 0, 100.0, 108.0).is_none());
    assert!(cell(&levels, 0, 108.0, 100.0).is_none());
    assert_eq!(positives(&levels, 0), 4);
    // Right and bottom edges through centres at 100, left and top through 84:
    // only (92, 92) is strictly inside.
    let levels = assign(320, 1, &[gt(84.0, 84.0, 100.0, 100.0, 0)]).unwrap();
    assert!(cell(&levels, 0, 100.0, 92.0).is_none());
    assert!(cell(&levels, 0, 92.0, 100.0).is_none());
    assert_eq!(positives(&levels, 0), 1);
}
