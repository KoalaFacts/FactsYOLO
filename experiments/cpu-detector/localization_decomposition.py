# SPDX-License-Identifier: Apache-2.0
"""Development-only decomposition of TD1 bounding-box localization error.

Only pre-registered development validation rows are passed here; final held-out
predictions are deliberately not analyzed or used for model selection. The
counterfactual boxes are geometric diagnostics, not model predictions or fixes.
"""
import math


def _box(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("invalid localization corners")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= 64 and 0 <= y1 < y2 <= 64):
        raise ValueError("invalid localization geometry")
    return tuple(value)


def _iou(a, b):
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def _mean(items):
    return sum(items) / len(items) if items else None


def decompose(rows, predictions):
    """Separate misses/class errors from center and size errors.

    Mean errors and geometric counterfactuals include only correct-class
    detections. Low-IoU cases retain sample indices, never pixels or weights.
    Counterfactual IoUs are unconstrained hypothetical boxes (no clipping),
    not estimates of improved detector performance.
    """
    if not rows or len(rows) != len(predictions):
        raise ValueError("localization diagnostic row count mismatch")
    groups = {name: {"positives": 0, "missing": 0, "wrong_class": 0,
                     "measurements": [], "low_iou_cases": []}
              for name in ("small", "larger")}
    for index, (row, pred) in enumerate(zip(rows, predictions)):
        cls = row["class"]
        if cls is None:
            continue
        if type(cls) is not int or cls not in (0, 1):
            raise ValueError("invalid target class")
        target = _box(row["corners"])
        area = (target[2] - target[0]) * (target[3] - target[1])
        if not math.isclose(area, row["area"], rel_tol=0, abs_tol=1e-9):
            raise ValueError("target area does not match geometry")
        group = groups["small" if area <= 256 else "larger"]
        group["positives"] += 1
        if pred is None:
            group["missing"] += 1
            continue
        if type(pred.get("class")) is not int or pred["class"] not in (0, 1):
            raise ValueError("invalid predicted class")
        box = _box(pred["corners"])
        if pred["class"] != cls:
            group["wrong_class"] += 1
            continue
        tx = (target[0] + target[2]) / 2
        ty = (target[1] + target[3]) / 2
        px = (box[0] + box[2]) / 2
        py = (box[1] + box[3]) / 2
        tw, th = target[2] - target[0], target[3] - target[1]
        pw, ph = box[2] - box[0], box[3] - box[1]
        actual = _iou(box, target)
        # Independently correct position or extent. These operations need not
        # have positive benefit; they are not additive causal attributions.
        center_corrected = _iou((tx-pw/2, ty-ph/2, tx+pw/2, ty+ph/2), target)
        size_corrected = _iou((px-tw/2, py-th/2, px+tw/2, py+th/2), target)
        values = (abs(px-tx), abs(py-ty), abs(pw-tw), abs(ph-th),
                  actual, center_corrected, size_corrected)
        group["measurements"].append(values)
        if actual < 0.5:
            group["low_iou_cases"].append({
                "sample_index": row.get("index", index),
                "iou": actual,
                "iou_if_center_aligned": center_corrected,
                "iou_if_size_aligned": size_corrected,
                "absolute_center_error_px": [values[0], values[1]],
                "absolute_size_error_px": [values[2], values[3]],
            })
    result = {}
    for name, group in groups.items():
        columns = list(zip(*group["measurements"])) if group["measurements"] else [[]] * 7
        result[name] = {
            "positives": group["positives"],
            "missing_detections": group["missing"],
            "wrong_class_detections": group["wrong_class"],
            "correct_class_detections": len(group["measurements"]),
            "mean_abs_center_error_px": [_mean(columns[0]), _mean(columns[1])],
            "mean_abs_size_error_px": [_mean(columns[2]), _mean(columns[3])],
            "mean_iou_correct_class_detected": _mean(columns[4]),
            "mean_iou_if_center_aligned": _mean(columns[5]),
            "mean_iou_if_size_aligned": _mean(columns[6]),
            "low_iou_correct_class_cases": group["low_iou_cases"],
        }
    return result
