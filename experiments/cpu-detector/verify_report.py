# SPDX-License-Identifier: Apache-2.0
"""Independently recompute held-out detection metrics from JSON prediction rows.

This audits already-produced training evidence rather than treating the Rust
runner's `passed` flag as proof that its metric arithmetic is correct. The
checker has no Rust/PyPI dependencies and deliberately uses Python's distinct
geometry implementation, with a small tolerance for Rust F32 rounding.
"""
import math


_IMAGE_EDGE = 64.0
_ABS_TOL = 2e-5
_METRIC_KEYS = (
    "class_accuracy", "recall_iou50", "mean_iou", "false_positive_rate"
)


def _finite_number(value, what):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{what}: expected finite number")
    return float(value)


def _box(value, what):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{what}: expected four corners")
    x1, y1, x2, y2 = [_finite_number(v, what) for v in value]
    if not (0 <= x1 <= x2 <= _IMAGE_EDGE and
            0 <= y1 <= y2 <= _IMAGE_EDGE):
        raise ValueError(f"{what}: coordinates outside image or inverted")
    return x1, y1, x2, y2


def _iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _close(actual, expected):
    return abs(actual - expected) <= _ABS_TOL


def verify_test_predictions(report):
    """Return independently computed held-out metrics or raise ValueError.

    The reported metrics and per-row IoU must agree with independently
    recomputed IoU, class accuracy, recall and false-positive rate. The
    experiment separately checks model training, gradients, artifact integrity,
    test-set provenance, and native consumers; this function does not claim to
    replace any of those gates.
    """
    rows = report.get("predictions")
    declared_count = report.get("test_samples")
    if (not isinstance(rows, list) or type(declared_count) is not int or
            declared_count != len(rows) or not rows):
        raise ValueError("missing, extra, or incorrectly sized prediction rows")

    positives = negatives = class_hits = recall_hits = false_positives = 0
    iou_sum = 0.0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("index")) is not int or row["index"] != index:
            raise ValueError(f"prediction index mismatch at {index}")
        expected_class = row.get("expected_class")
        if expected_class is not None and (type(expected_class) is not int or expected_class not in (0, 1)):
            raise ValueError(f"invalid expected class at {index}")
        corners = row.get("expected_corners")
        if expected_class is None:
            if corners is not None:
                raise ValueError(f"background has target corners at {index}")
            negatives += 1
        else:
            corners = _box(corners, f"target corners {index}")
            positives += 1

        detection = row.get("prediction")
        predicted_class = None
        predicted_corners = None
        if detection is not None:
            if not isinstance(detection, dict) or type(detection.get("class")) is not int or detection["class"] not in (0, 1):
                raise ValueError(f"invalid detection class at {index}")
            predicted_class = detection["class"]
            predicted_corners = _box(detection.get("corners"), f"predicted corners {index}")
            score = _finite_number(detection.get("score"), f"score {index}")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"invalid detection score at {index}")

        overlap = _iou(predicted_corners, corners) if corners is not None and predicted_corners is not None else 0.0
        if not _close(overlap, _finite_number(row.get("iou"), f"reported IoU {index}")):
            raise ValueError(f"reported IoU mismatch at {index}")
        if expected_class is None:
            false_positives += predicted_class is not None
        else:
            iou_sum += overlap
            class_hits += predicted_class == expected_class
            recall_hits += predicted_class == expected_class and overlap >= 0.5

    if positives == 0 or negatives == 0:
        raise ValueError("held-out metrics require positive and background samples")
    actual = {
        "positives": positives,
        "negatives": negatives,
        "class_accuracy": class_hits / positives,
        "recall_iou50": recall_hits / positives,
        "mean_iou": iou_sum / positives,
        "false_positive_rate": false_positives / negatives,
    }
    reported = report.get("test")
    if not isinstance(reported, dict):
        raise ValueError("missing held-out metric summary")
    for key, value in actual.items():
        item = reported.get(key)
        if key in ("positives", "negatives"):
            if type(item) is not int or item != value:
                raise ValueError(f"held-out {key} mismatch")
        elif not _close(value, _finite_number(item, f"metric {key}")):
            raise ValueError(f"held-out {key} mismatch")
    return actual
