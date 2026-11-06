// SPDX-License-Identifier: Apache-2.0
use td2_eval::{BoxXyxy, Detection, GroundTruth, Summary, evaluate, iou_thresholds};

fn bbox(left: f64, top: f64, right: f64, bottom: f64) -> BoxXyxy {
    BoxXyxy {
        left,
        top,
        right,
        bottom,
    }
}

fn truth(image_id: u64, class_id: u32, bbox: BoxXyxy) -> GroundTruth {
    GroundTruth {
        image_id,
        class_id,
        bbox,
        crowd: false,
    }
}

fn detection(image_id: u64, class_id: u32, bbox: BoxXyxy, score: f64) -> Detection {
    Detection {
        image_id,
        class_id,
        bbox,
        score,
    }
}

fn close(actual: Option<f64>, expected: f64) {
    let actual = actual.expect("metric defined");
    assert!((actual - expected).abs() <= 1e-12, "{actual} != {expected}");
}

#[test]
fn thresholds_are_the_ten_coco_levels() {
    let thresholds = iou_thresholds();
    assert_eq!(thresholds[0], 0.5);
    assert_eq!(thresholds[9], 0.95);
    for pair in thresholds.windows(2) {
        assert!((pair[1] - pair[0] - 0.05).abs() < 1e-12);
    }
}

#[test]
fn perfect_detections_score_one_and_empty_detections_score_zero() {
    let truths = [
        truth(1, 0, bbox(0.0, 0.0, 10.0, 10.0)),
        truth(2, 0, bbox(50.0, 50.0, 200.0, 200.0)),
    ];
    let perfect: Vec<_> = truths
        .iter()
        .map(|t| detection(t.image_id, t.class_id, t.bbox, 0.9))
        .collect();
    let summary = evaluate(&truths, &perfect, 100).unwrap();
    close(summary.map, 1.0);
    close(summary.map50, 1.0);
    close(summary.map_small, 1.0);
    assert_eq!(summary.map_medium, None);
    close(summary.map_large, 1.0);
    let empty = evaluate(&truths, &[], 100).unwrap();
    close(empty.map, 0.0);
    assert_eq!(empty.classes[0].ground_truths, 2);
}

#[test]
fn higher_scored_false_positive_halves_precision_at_full_recall() {
    let truths = [truth(1, 0, bbox(0.0, 0.0, 100.0, 100.0))];
    let detections = [
        detection(1, 0, bbox(300.0, 300.0, 400.0, 400.0), 0.9),
        detection(1, 0, bbox(0.0, 0.0, 100.0, 100.0), 0.8),
    ];
    // Recall reaches 1 only at rank 2 with precision 1/2 at every level.
    close(evaluate(&truths, &detections, 100).unwrap().map50, 0.5);
}

#[test]
fn iou_threshold_is_inclusive_and_duplicates_are_false_positives() {
    let truths = [truth(1, 0, bbox(0.0, 0.0, 100.0, 100.0))];
    // IoU exactly 0.5: intersection 100x50 over union 100x100.
    let half = [detection(1, 0, bbox(0.0, 0.0, 100.0, 50.0), 0.9)];
    let summary = evaluate(&truths, &half, 100).unwrap();
    close(summary.map50, 1.0);
    close(summary.map75, 0.0);
    close(summary.map, 0.1);
    let duplicate = [
        detection(1, 0, bbox(0.0, 0.0, 100.0, 100.0), 0.9),
        detection(1, 0, bbox(0.0, 0.0, 100.0, 100.0), 0.8),
    ];
    // The second copy cannot reuse the matched truth, after full recall.
    close(evaluate(&truths, &duplicate, 100).unwrap().map, 1.0);
}

#[test]
fn crowd_regions_absorb_detections_without_requiring_them() {
    let truths = [
        truth(1, 0, bbox(0.0, 0.0, 50.0, 50.0)),
        GroundTruth {
            crowd: true,
            ..truth(1, 0, bbox(100.0, 100.0, 400.0, 400.0))
        },
    ];
    let detections = [
        detection(1, 0, bbox(150.0, 150.0, 200.0, 200.0), 0.95),
        detection(1, 0, bbox(160.0, 160.0, 210.0, 210.0), 0.94),
        detection(1, 0, bbox(0.0, 0.0, 50.0, 50.0), 0.5),
    ];
    let summary = evaluate(&truths, &detections, 100).unwrap();
    close(summary.map, 1.0);
    assert_eq!(summary.classes[0].ground_truths, 1);
    let only_crowd = evaluate(&truths[1..], &detections, 100).unwrap();
    assert_eq!(only_crowd.map, None);
    assert!(only_crowd.classes.is_empty());
}

#[test]
fn wrong_class_and_wrong_image_do_not_match() {
    let truths = [truth(1, 0, bbox(0.0, 0.0, 10.0, 10.0))];
    let detections = [
        detection(1, 1, bbox(0.0, 0.0, 10.0, 10.0), 0.9),
        detection(2, 0, bbox(0.0, 0.0, 10.0, 10.0), 0.9),
    ];
    let summary = evaluate(&truths, &detections, 100).unwrap();
    close(summary.map, 0.0);
    assert_eq!(summary.classes.len(), 1, "class 1 has no ground truth");
}

#[test]
fn per_image_class_cap_keeps_only_the_highest_scores() {
    let truths = [truth(1, 0, bbox(0.0, 0.0, 10.0, 10.0))];
    let detections = [
        detection(1, 0, bbox(0.0, 0.0, 10.0, 10.0), 0.1),
        detection(1, 0, bbox(50.0, 50.0, 60.0, 60.0), 0.9),
    ];
    close(evaluate(&truths, &detections, 1).unwrap().map, 0.0);
    close(evaluate(&truths, &detections, 2).unwrap().map50, 0.5);
}

#[test]
fn rejects_invalid_inputs() {
    let good = [truth(1, 0, bbox(0.0, 0.0, 1.0, 1.0))];
    let inverted = [truth(1, 0, bbox(5.0, 0.0, 1.0, 1.0))];
    let nan_box = [detection(1, 0, bbox(f64::NAN, 0.0, 1.0, 1.0), 0.5)];
    let inf_score = [detection(1, 0, bbox(0.0, 0.0, 1.0, 1.0), f64::INFINITY)];
    assert_eq!(
        evaluate(&inverted, &[], 100).unwrap_err(),
        "ground truth 0: box is inverted"
    );
    assert_eq!(
        evaluate(&good, &nan_box, 100).unwrap_err(),
        "detection 0: box edge is not finite"
    );
    assert_eq!(
        evaluate(&good, &inf_score, 100).unwrap_err(),
        "detection 0: score is not finite"
    );
    assert_eq!(
        evaluate(&good, &[], 0).unwrap_err(),
        "max_detections must be positive"
    );
    let inverted_detection = [detection(1, 0, bbox(0.0, 3.0, 1.0, 1.0), 0.5)];
    assert_eq!(
        evaluate(&good, &inverted_detection, 100).unwrap_err(),
        "detection 0: box is inverted"
    );
    let nan_truth = [truth(1, 0, bbox(0.0, 0.0, 1.0, f64::NAN))];
    assert_eq!(
        evaluate(&nan_truth, &[], 100).unwrap_err(),
        "ground truth 0: box edge is not finite"
    );
    let nan_score = [detection(1, 0, bbox(0.0, 0.0, 1.0, 1.0), f64::NAN)];
    assert_eq!(
        evaluate(&good, &nan_score, 100).unwrap_err(),
        "detection 0: score is not finite"
    );
    let overflow = [truth(1, 0, bbox(-1e308, 0.0, 1e308, 10.0))];
    assert_eq!(
        evaluate(&overflow, &[], 100).unwrap_err(),
        "ground truth 0: box area is not finite"
    );
}

#[test]
fn negative_zero_score_ties_with_zero_in_input_order() {
    let truths = [truth(1, 1, bbox(0.0, 0.0, 10.0, 10.0))];
    let detections = [
        detection(1, 1, bbox(50.0, 50.0, 60.0, 60.0), -0.0),
        detection(1, 1, bbox(0.0, 0.0, 10.0, 10.0), 0.0),
    ];
    // The reference keeps input order on equal scores: false positive first.
    close(evaluate(&truths, &detections, 100).unwrap().map, 0.5);
}

struct Case {
    name: String,
    truths: Vec<GroundTruth>,
    detections: Vec<Detection>,
    summary: [f64; 6],
    classes: Vec<(u32, f64, f64)>,
}

fn cases() -> Vec<Case> {
    let text = include_str!("data/pycocotools-crosscheck.txt");
    let mut cases = Vec::new();
    let mut current: Option<Case> = None;
    for line in text.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let number = |index: usize| fields[index].parse::<f64>().unwrap();
        let edges = || bbox(number(3), number(4), number(5), number(6));
        match fields[0] {
            "case" => {
                current = Some(Case {
                    name: fields[1].to_owned(),
                    truths: Vec::new(),
                    detections: Vec::new(),
                    summary: [0.0; 6],
                    classes: Vec::new(),
                });
            }
            "g" => current.as_mut().unwrap().truths.push(GroundTruth {
                image_id: fields[1].parse().unwrap(),
                class_id: fields[2].parse().unwrap(),
                bbox: edges(),
                crowd: fields[7] == "1",
            }),
            "d" => current.as_mut().unwrap().detections.push(detection(
                fields[1].parse().unwrap(),
                fields[2].parse().unwrap(),
                edges(),
                number(7),
            )),
            "e" => current.as_mut().unwrap().summary = std::array::from_fn(|i| number(i + 1)),
            "c" => current.as_mut().unwrap().classes.push((
                fields[1].parse().unwrap(),
                number(2),
                number(3),
            )),
            "end" => cases.push(current.take().unwrap()),
            other => panic!("unknown record {other}"),
        }
    }
    cases
}

/// The reference prints -1 for an undefined metric.
fn reference(value: f64) -> Option<f64> {
    (value >= 0.0).then_some(value)
}

fn agree(actual: Option<f64>, expected: f64, what: &str) {
    match (actual, reference(expected)) {
        (None, None) => {}
        (Some(a), Some(e)) => assert!((a - e).abs() <= 1e-9, "{what}: {a} != {e}"),
        (a, e) => panic!("{what}: {a:?} != {e:?}"),
    }
}

#[test]
fn matches_frozen_reference_evaluator_vectors() {
    let cases = cases();
    assert_eq!(cases.len(), 13);
    for case in cases {
        let Summary {
            map,
            map50,
            map75,
            map_small,
            map_medium,
            map_large,
            classes,
        } = evaluate(&case.truths, &case.detections, 100).unwrap();
        let name = &case.name;
        for (actual, expected, label) in [
            (map, case.summary[0], "map"),
            (map50, case.summary[1], "map50"),
            (map75, case.summary[2], "map75"),
            (map_small, case.summary[3], "small"),
            (map_medium, case.summary[4], "medium"),
            (map_large, case.summary[5], "large"),
        ] {
            agree(actual, expected, &format!("case {name} {label}"));
        }
        assert_eq!(classes.len(), case.classes.len(), "case {name} classes");
        for (result, (class_id, ap, ap50)) in classes.iter().zip(&case.classes) {
            assert_eq!(result.class_id, *class_id);
            agree(
                Some(result.ap),
                *ap,
                &format!("case {name} class {class_id} ap"),
            );
            agree(
                Some(result.ap50),
                *ap50,
                &format!("case {name} class {class_id} ap50"),
            );
        }
    }
}
