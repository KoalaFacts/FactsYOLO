#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-registered TD1 edge-vs-interior development diagnostic.

A fixed geometric subsample is selected *before* invoking a trained model.
Existing final holdout samples, thresholds, optimizer and model are untouched.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import diagnostic_validation as d
from localization_decomposition import _box

SEED = 20261030
CANDIDATES = 4096
QUOTA = 48
EDGE_MARGIN_PX = 2
# Frozen on the untrained synthetic generator, before examining predictions.
SELECTED_FINGERPRINT = "04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4"
GROUPS = ("small_edge", "small_interior", "larger_edge", "larger_interior", "background")


def stratum(row):
    """Use *target* geometry alone, never a prediction, to define a group."""
    cls = row["class"]
    if cls is None:
        return "background"
    if type(cls) is not int or cls not in (0, 1):
        raise ValueError("invalid target class")
    x1, y1, x2, y2 = _box(row["corners"])
    area = (x2 - x1) * (y2 - y1)
    if not math.isclose(area, row["area"], rel_tol=0, abs_tol=1e-9):
        raise ValueError("target area mismatch")
    size = "small" if area <= d.SMALL_AREA_MAX else "larger"
    position = "edge" if min(x1, y1, 64 - x2, 64 - y2) <= EDGE_MARGIN_PX else "interior"
    return f"{size}_{position}"


def selected_rows():
    """First 48 rows of each stratum in a fixed 4,096-candidate generator stream."""
    by_group = {name: [] for name in GROUPS}
    for row in d.samples(SEED, CANDIDATES):
        group = by_group[stratum(row)]
        if len(group) < QUOTA:
            group.append(row)
        if all(len(group) == QUOTA for group in by_group.values()):
            break  # Remaining candidates cannot change the fixed first-quota selection.
    if any(len(rows) != QUOTA for rows in by_group.values()):
        raise ValueError("registered boundary split is incomplete")
    rows = sorted((row for group in by_group.values() for row in group), key=lambda r: r["index"])
    if d.fingerprint(rows) != SELECTED_FINGERPRINT:
        raise ValueError("registered boundary split fingerprint mismatch")
    if len({row["index"] for row in rows}) != len(rows):
        raise ValueError("repeated boundary sample index")
    return rows


def summarize(rows, predictions):
    if not rows or len(rows) != len(predictions):
        raise ValueError("boundary diagnostic prediction count mismatch")
    groups = {name: {"samples": 0, "missing": 0, "wrong_class": 0,
                     "correct_class_iou50": 0, "iou_sum": 0.0,
                     "abs_center_x": [], "abs_center_y": [],
                     "low_iou_sample_indices": [], "false_positives": 0}
              for name in GROUPS}
    for row, raw in zip(rows, predictions):
        name = stratum(row)
        group = groups[name]
        group["samples"] += 1
        pred = d._prediction(raw)
        if pred is not None:
            _box(pred["corners"])  # Fail closed on degenerate predicted boxes.
        if name == "background":
            group["false_positives"] += pred is not None
            continue
        if pred is None:
            group["missing"] += 1
            continue
        if pred["class"] != row["class"]:
            group["wrong_class"] += 1
            continue
        target = row["corners"]
        corners = pred["corners"]
        iou = d._iou(corners, target)
        group["iou_sum"] += iou  # Wrong-class and missing predictions contribute zero.
        group["correct_class_iou50"] += iou >= 0.5
        group["abs_center_x"].append(abs((corners[0] + corners[2] - target[0] - target[2]) / 2))
        group["abs_center_y"].append(abs((corners[1] + corners[3] - target[1] - target[3]) / 2))
        if iou < 0.5:
            group["low_iou_sample_indices"].append(row["index"])

    output = {}
    for name, group in groups.items():
        n = group["samples"]
        if name == "background":
            output[name] = {"samples": n, "false_positives": group["false_positives"]}
            continue
        center_count = len(group["abs_center_x"])
        output[name] = {
            "samples": n, "missing_detections": group["missing"],
            "wrong_class_detections": group["wrong_class"],
            "correct_class_detections": center_count,
            "correct_class_iou50": group["correct_class_iou50"],
            "recall_correct_class_iou50": group["correct_class_iou50"] / n if n else None,
            "mean_correct_class_iou_including_misses": group["iou_sum"] / n if n else None,
            "mean_abs_center_error_px_correct_class": [
                sum(group[key]) / center_count if center_count else None
                for key in ("abs_center_x", "abs_center_y")],
            "low_iou_correct_class_sample_indices": group["low_iou_sample_indices"],
        }
    return output


def evaluate(binary, artifact, training_metrics, *, invoke=None):
    """Run only after a passed frozen 2,000-step training gate."""
    training = json.loads(Path(training_metrics).read_text())
    if (training.get("passed") is not True or training.get("steps") != 2000 or
            training.get("training_seed") != 101 or training.get("test_seed") != 909 or
            training.get("test_fingerprint") != d.LEGACY_TEST_FINGERPRINT or
            training.get("source_commit") in (None, "unknown")):
        raise ValueError("incompatible training provenance")
    rows = selected_rows()
    original_hash = hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
    predictions = []
    with tempfile.TemporaryDirectory(prefix="factsyolo-edge-development-") as temporary:
        rgb = Path(temporary) / "sample.rgb"
        for row in rows:
            rgb.write_bytes(row["pixels"])
            if invoke is None:
                completed = subprocess.run([str(binary), "infer", str(artifact), str(rgb)],
                                           capture_output=True, text=True, timeout=15, check=True)
                prediction = json.loads(completed.stdout)
            else:
                prediction = invoke(row)
            predictions.append(prediction)
    if hashlib.sha256(Path(artifact).read_bytes()).hexdigest() != original_hash:
        raise ValueError("model artifact changed during boundary diagnostic")
    grouped = summarize(rows, predictions)
    comparisons = {
        size: grouped[f"{size}_edge"]["recall_correct_class_iou50"] -
              grouped[f"{size}_interior"]["recall_correct_class_iou50"]
        for size in ("small", "larger")
    }
    return {
        "status": "development-diagnostic-only", "source_commit": training["source_commit"],
        "backend": training["backend"], "trained_model_sha256": original_hash,
        "generator": d.GENERATOR, "seed": SEED, "candidate_count": CANDIDATES,
        "selected_count": len(rows), "selected_fingerprint": SELECTED_FINGERPRINT,
        "quota_per_stratum": QUOTA, "edge_margin_pixels": EDGE_MARGIN_PX,
        "small_target_area_max_pixels": d.SMALL_AREA_MAX,
        "groups": grouped, "edge_minus_interior_iou50_recall": comparisons,
    }


def main():
    if len(sys.argv) != 4:
        raise ValueError("usage: edge_validation.py <fy-cpu-evaluate> <model.fytd> <metrics.json>")
    print(json.dumps(evaluate(*sys.argv[1:]), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
