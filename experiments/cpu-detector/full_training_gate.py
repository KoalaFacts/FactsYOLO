# SPDX-License-Identifier: Apache-2.0
"""Reject unverifiable full-training benchmark output; never require a speedup."""
import math
import re

BACKEND_NAMES = {"burn-flex-0.22.0", "candle-cpu-0.11.0"}


def _int(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def _finite(value, minimum=0.0, maximum=None):
    return (type(value) in (int, float) and math.isfinite(value)
            and value >= minimum and (maximum is None or value <= maximum))


def _count_rate(value, count):
    # Rust counts detections before emitting a rounded f32 ratio.
    return (_finite(value, 0.0, 1.0)
            and abs(value * count - round(value * count)) <= 1e-4)


def _valid_validation(row):
    if not isinstance(row, dict) or set(row) != {
            "positives", "negatives", "class_accuracy", "recall_iou50",
            "mean_iou", "false_positive_rate"}:
        return False
    return (row["positives"] == 102 and type(row["positives"]) is int
            and row["negatives"] == 26 and type(row["negatives"]) is int
            and all(_finite(row[key], 0.0, 1.0) for key in (
                "class_accuracy", "recall_iou50", "mean_iou", "false_positive_rate"))
            and _count_rate(row["class_accuracy"], 102)
            and _count_rate(row["recall_iou50"], 102)
            and _count_rate(row["false_positive_rate"], 26)
            and row["recall_iou50"] <= row["class_accuracy"] + 1e-6
            and row["class_accuracy"] >= 0.95
            and row["recall_iou50"] >= 0.85
            and row["mean_iou"] >= 0.70
            and row["false_positive_rate"] <= 0.05)


def validate_full_training_benchmark(report, expected_sha, backend):
    """Validate provenance, actual run data, and bitwise-parity witnesses.

    Wall-time is a measurement, not a pass/fail speedup criterion. The 8 pairs (4 with each first arm)
    are sequential executions on one host, not independent hardware samples.
    """
    if (not isinstance(report, dict)
            or report.get("scope") != "full TD1 Burn/Candle training (2000 optimizer steps) in one process"
            or report.get("source_commit") != expected_sha
            or backend not in BACKEND_NAMES
            or report.get("backend") != backend
            or report.get("build") != "release"
            or report.get("os") not in ("linux", "windows")
            or report.get("architecture") != "x86_64"
            or report.get("rayon_threads") != "1"
            or any(type(report.get(key)) is not int or report[key] != value
                   for key, value in (
                       ("training_seed", 101),
                       ("initialization_seed", 404),
                       ("sampler_seed", 707),
                       ("steps_per_arm", 2000),
                       ("batch_size", 64),
                       ("training_samples", 4096)))
            or type(report.get("repetitions")) is not int
            or report.get("repetitions") != 8
            or report.get("data_preparation_excluded") is not True
            or report.get("evaluation_excluded_from_timing") is not True
            or report.get("exact_weight_rng_loss_validation_parity") is not True):
        raise ValueError("full training benchmark provenance/contract mismatch")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 8:
        raise ValueError("full training benchmark must contain eight balanced paired runs")
    canonical_witness = None
    for i, run in enumerate(runs):
        if (not isinstance(run, dict) or set(run) != {"first", "historical", "reused"}
                or run["first"] != ("historical" if i % 2 == 0 else "reused")):
            raise ValueError("full training benchmark order/arms mismatch")
        arms = [run.get("historical"), run.get("reused")]
        for arm in arms:
            if not isinstance(arm, dict):
                raise ValueError("missing full training arm")
            for field in ("training_ns", "batch_preparation_ns", "allocation_requests", "requested_bytes"):
                if not _int(arm.get(field)):
                    raise ValueError(f"invalid training resource metric: {field}")
            if arm["training_ns"] <= 0 or arm["batch_preparation_ns"] > arm["training_ns"]:
                raise ValueError("impossible training/preparation times")
            if not (isinstance(arm.get("model_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", arm["model_sha256"])):
                raise ValueError("invalid trained model digest")
            if not _int(arm.get("sampler_next_draw"), 0, 2**32 - 1):
                raise ValueError("invalid post-training RNG")
            if not _int(arm.get("last_batch_loss_bits"), 0, 2**32 - 1):
                raise ValueError("invalid batch loss witness")
            if not _valid_validation(arm.get("validation")):
                raise ValueError("training arm failed registered validation")
        left, right = arms
        for name in ("model_sha256", "sampler_next_draw", "last_batch_loss_bits", "validation"):
            if left[name] != right[name]:
                raise ValueError(f"non-equivalent complete training: {name}")
        # Every repeat uses identical model/seed/sampler inputs; final weight
        # parity alone does not certify RNG, loss or evaluation reproducibility.
        witness = (left["model_sha256"], left["sampler_next_draw"],
                   left["last_batch_loss_bits"], left["validation"])
        if canonical_witness is None:
            canonical_witness = witness
        elif witness != canonical_witness:
            raise ValueError("repeated training drifted despite identical seeds")
    return True
