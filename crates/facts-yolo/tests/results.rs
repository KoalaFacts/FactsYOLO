// SPDX-License-Identifier: Apache-2.0
use facts_yolo::{
    ffi::types::*,
    result::{Candidate, ImageTransform, finalize},
};

fn identity() -> ImageTransform {
    ImageTransform {
        scale_x: 1.0,
        scale_y: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    }
}

fn candidate(index: usize) -> Candidate {
    Candidate {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        score: 0.9,
        class_id: 0,
        source_index: index,
    }
}

fn basic() -> [Candidate; 4] {
    [
        Candidate {
            class_id: 1,
            x_min: -1.0,
            y_min: -1.0,
            x_max: 3.0,
            y_max: 3.0,
            ..candidate(0)
        },
        Candidate {
            x_min: 1.0,
            y_min: 1.0,
            x_max: 5.0,
            y_max: 5.0,
            ..candidate(1)
        },
        Candidate {
            score: 0.5,
            x_min: 3.0,
            y_min: 3.0,
            x_max: 5.0,
            y_max: 5.0,
            ..candidate(2)
        },
        Candidate {
            score: 0.7,
            x_min: 3.0,
            y_min: 3.0,
            x_max: 3.0,
            y_max: 5.0,
            ..candidate(3)
        },
    ]
}

fn basic_transform() -> ImageTransform {
    ImageTransform {
        scale_x: 2.0,
        scale_y: 2.0,
        pad_x: 1.0,
        pad_y: 1.0,
    }
}

#[test]
fn canonical_original_image_coordinates_and_order() {
    let result = finalize(&basic(), basic_transform(), (2, 2), 2, 0.5, 16).unwrap();
    assert!(!result.truncated);
    assert_eq!(result.detections.len(), 3);
    assert_eq!(
        result.detections[0],
        FyDetection {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            score: 0.9,
            class_id: 0,
        }
    );
    assert_eq!(result.detections[1].class_id, 1);
    assert_eq!(result.detections[1].x_max, 1.0);
    assert_eq!(result.detections[2].score, 0.5);
    assert_eq!(result.detections[2].x_min, 1.0);
}

#[test]
fn cap_applies_after_filtering_and_marks_only_removed_boxes() {
    let result = finalize(&basic(), basic_transform(), (2, 2), 2, 0.5, 2).unwrap();
    assert_eq!(result.detections.len(), 2);
    assert!(result.truncated);
    let exact = finalize(&basic(), basic_transform(), (2, 2), 2, 0.5, 3).unwrap();
    assert!(!exact.truncated);
    let filtered = finalize(&basic(), basic_transform(), (2, 2), 2, 1.0, 1).unwrap();
    assert!(filtered.detections.is_empty());
    assert!(!filtered.truncated);
}

#[test]
fn threshold_is_inclusive_at_zero_and_one() {
    for score in [0.0, 0.5, 1.0] {
        let value = Candidate { score, ..candidate(0) };
        let result = finalize(&[value], identity(), (2, 2), 1, score, 1).unwrap();
        assert_eq!(result.detections.len(), 1);
    }
}

#[test]
fn clipping_precedes_zero_area_removal() {
    let outside = Candidate {
        x_min: -5.0,
        x_max: -2.0,
        ..candidate(0)
    };
    let partial = Candidate { x_min: -1.0, ..candidate(1) };
    let result = finalize(&[outside, partial], identity(), (2, 2), 1, 0.0, 1).unwrap();
    assert_eq!(result.detections.len(), 1);
    assert_eq!(result.detections[0].x_min, 0.0);
    assert!(!result.truncated);
}

#[test]
fn invalid_candidates_are_rejected_even_below_threshold_or_after_cap() {
    let mut cases = Vec::new();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        cases.push(Candidate { x_min: invalid, score: 0.1, ..candidate(1) });
        cases.push(Candidate { y_min: invalid, score: 0.1, ..candidate(1) });
        cases.push(Candidate { x_max: invalid, score: 0.1, ..candidate(1) });
        cases.push(Candidate { y_max: invalid, score: 0.1, ..candidate(1) });
        cases.push(Candidate { score: invalid, ..candidate(1) });
    }
    cases.push(Candidate { score: -0.1, ..candidate(1) });
    cases.push(Candidate { score: 1.1, ..candidate(1) });
    cases.push(Candidate { class_id: 1, score: 0.1, ..candidate(1) });
    cases.push(Candidate { x_min: 2.0, score: 0.1, ..candidate(1) });
    cases.push(Candidate { y_min: 2.0, score: 0.1, ..candidate(1) });
    for invalid in cases {
        let error = finalize(&[candidate(0), invalid], identity(), (2, 2), 1, 0.8, 1);
        assert_eq!(error.unwrap_err().status, FY_MODEL_OUTPUT_INVALID);
    }
}

#[test]
fn invalid_transforms_are_rejected_even_for_an_empty_result() {
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        for transform in [
            ImageTransform { scale_x: invalid, ..identity() },
            ImageTransform { scale_y: invalid, ..identity() },
        ] {
            let error = finalize(&[], transform, (2, 2), 1, 0.0, 1);
            assert_eq!(error.unwrap_err().status, FY_MODEL_OUTPUT_INVALID);
        }
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for transform in [
            ImageTransform { pad_x: invalid, ..identity() },
            ImageTransform { pad_y: invalid, ..identity() },
        ] {
            let error = finalize(&[], transform, (2, 2), 1, 0.0, 1);
            assert_eq!(error.unwrap_err().status, FY_MODEL_OUTPUT_INVALID);
        }
    }
}

#[test]
fn invalid_call_options_and_model_class_count_have_distinct_statuses() {
    for threshold in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        let error = finalize(&[], identity(), (2, 2), 1, threshold, 1);
        assert_eq!(error.unwrap_err().status, FY_INVALID_ARGUMENT);
    }
    for source in [(0, 2), (2, 0)] {
        let error = finalize(&[], identity(), source, 1, 0.0, 1);
        assert_eq!(error.unwrap_err().status, FY_INVALID_ARGUMENT);
    }
    let zero_cap = finalize(&[], identity(), (2, 2), 1, 0.0, 0);
    assert_eq!(zero_cap.unwrap_err().status, FY_INVALID_ARGUMENT);
    let zero_classes = finalize(&[], identity(), (2, 2), 0, 0.0, 1);
    assert_eq!(zero_classes.unwrap_err().status, FY_MODEL_OUTPUT_INVALID);
}

#[test]
fn signed_zero_is_normalized_in_public_coordinates_and_scores() {
    let value = Candidate {
        x_min: -0.0,
        y_min: -0.0,
        score: -0.0,
        ..candidate(0)
    };
    let result = finalize(&[value], identity(), (2, 2), 1, 0.0, 1).unwrap();
    for value in [result.detections[0].x_min, result.detections[0].y_min, result.detections[0].score] {
        assert_eq!(value.to_bits(), 0);
    }
}

#[test]
fn ordering_and_top_k_do_not_depend_on_input_enumeration() {
    let mut values: Vec<_> = (0..100)
        .map(|index| Candidate {
            x_min: index as f32 / 100.0,
            score: if index % 2 == 0 { 0.9 } else { 0.8 },
            class_id: (index % 3) as u32,
            ..candidate(index)
        })
        .collect();
    let expected = finalize(&values, identity(), (2, 2), 3, 0.0, 17).unwrap();
    values.reverse();
    let reversed = finalize(&values, identity(), (2, 2), 3, 0.0, 17).unwrap();
    assert_eq!(expected.detections, reversed.detections);
    assert_eq!(expected.detections.len(), 17);
    assert!(expected.truncated && reversed.truncated);
    assert!(expected.detections.iter().all(|d| d.score == 0.9 && d.class_id == 0));
}
