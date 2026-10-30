# SPDX-License-Identifier: Apache-2.0
"""Reject unverifiable full-training benchmark output; never require a speedup."""
import json
import math
import re
import struct

# Metrics::passes compares f32 fields with f32 literals. Python parses JSON
# numbers as f64, so compare against the exact widened Rust f32 thresholds.
def _f32_literal(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


CLASS_ACCURACY_MIN = _f32_literal(0.95)
RECALL_IOU50_MIN = _f32_literal(0.85)
MEAN_IOU_MIN = _f32_literal(0.70)
FALSE_POSITIVE_RATE_MAX = _f32_literal(0.05)

BACKEND_NAMES = {"burn-flex-0.22.0", "candle-cpu-0.11.0"}
# The registered candidate budget is 600s, including all 2,000 optimizer steps.
MAX_TRAINING_NS = 600 * 1_000_000_000
# Each historical step allocates nonempty input and label Vec buffers; reuse
# reserves two such buffers exactly once. A successful allocator measurement
# cannot omit these requests or all requested bytes.
MIN_HISTORICAL_ALLOCATIONS = 2 * 2000
MIN_REUSED_ALLOCATIONS = 2


def _int(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def _valid_loss_bits(value):
    # calculate_loss rejects nonfinite loss and its objective is nonnegative.
    # A matching but impossible NaN/Inf/negative f32 witness is not evidence
    # of a successful optimizer step, even if model/RNG hashes still agree.
    if not _int(value, 0, 2**32 - 1):
        return False
    loss = struct.unpack("<f", struct.pack("<I", value))[0]
    return math.isfinite(loss) and loss >= 0.0


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
            and row["class_accuracy"] >= CLASS_ACCURACY_MIN
            and row["recall_iou50"] >= RECALL_IOU50_MIN
            and row["mean_iou"] >= MEAN_IOU_MIN
            and row["false_positive_rate"] <= FALSE_POSITIVE_RATE_MAX)


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
        for mode, arm in zip(("historical", "reused"), arms):
            if not isinstance(arm, dict):
                raise ValueError("missing full training arm")
            for field in ("training_ns", "batch_preparation_ns", "allocation_requests", "requested_bytes"):
                if not _int(arm.get(field)):
                    raise ValueError(f"invalid training resource metric: {field}")
            minimum_requests = (MIN_HISTORICAL_ALLOCATIONS if mode == "historical"
                                else MIN_REUSED_ALLOCATIONS)
            if arm["allocation_requests"] < minimum_requests or arm["requested_bytes"] == 0:
                raise ValueError("missing compulsory training batch buffer allocations")
            if (not 0 < arm["training_ns"] <= MAX_TRAINING_NS
                    or arm["batch_preparation_ns"] > arm["training_ns"]):
                raise ValueError("impossible training/preparation times")
            if not (isinstance(arm.get("model_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", arm["model_sha256"])):
                raise ValueError("invalid trained model digest")
            if not _int(arm.get("sampler_next_draw"), 0, 2**32 - 1):
                raise ValueError("invalid post-training RNG")
            if not _valid_loss_bits(arm.get("last_batch_loss_bits")):
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


# Deliberately error after a successful 2,000-step historical arm. This
# trace is from the real Release benchmark, not a synthetic completed report.
CONTROLLED_FAILURE = "controlled benchmark failure after one completed arm"


def validate_interrupted_full_training_diagnostics(raw_jsonl, source_commit, backend):
    """A failed Rust process must retain one genuine complete arm, not a success report."""
    if backend not in BACKEND_NAMES or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("invalid interrupted-training provenance")
    try:
        events = [json.loads(line) for line in raw_jsonl.splitlines()]
    except json.JSONDecodeError as error:
        raise ValueError("invalid interrupted-training JSONL") from error
    if len(events) != 5 or any(not isinstance(row, dict) for row in events):
        raise ValueError("missing interrupted-training events")
    for index, (kind, arm) in enumerate((
            ("started", "historical"), ("completed", "historical"),
            ("started", "reused"), ("failed", "reused"))):
        row = events[index]
        fields = {"schema", "source_commit", "backend", "pair", "arm", "event"}
        if kind == "completed":
            fields.add("measurement")
        if kind == "failed":
            fields.add("error")
        if (set(row) != fields
                or row["schema"] != "fy-full-training-progress-v1"
                or row["source_commit"] != source_commit
                or row["backend"] != backend or row["pair"] != 0
                or row["arm"] != arm or row["event"] != kind):
            raise ValueError("interrupted-training event provenance/order mismatch")
    if events[3]["error"] != CONTROLLED_FAILURE:
        raise ValueError("interrupted-training error identity mismatch")
    if events[4] != {"schema": "fy-full-training-progress-v1",
                     "event": "process_failed", "error": CONTROLLED_FAILURE}:
        raise ValueError("missing benchmark process failure event")
    arm = events[1]["measurement"]
    required = {"training_ns", "batch_preparation_ns", "allocation_requests",
                "requested_bytes", "model_sha256", "sampler_next_draw",
                "last_batch_loss_bits", "validation"}
    if (not isinstance(arm, dict) or set(arm) != required
            or not _int(arm["training_ns"], 1, MAX_TRAINING_NS)
            or not _int(arm["batch_preparation_ns"])
            or arm["batch_preparation_ns"] > arm["training_ns"]
            or not _int(arm["allocation_requests"], MIN_HISTORICAL_ALLOCATIONS)
            or not _int(arm["requested_bytes"], 1)
            or not isinstance(arm["model_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", arm["model_sha256"])
            or not _int(arm["sampler_next_draw"], 0, 2**32 - 1)
            or not _valid_loss_bits(arm["last_batch_loss_bits"])
            or not _valid_validation(arm["validation"])):
        raise ValueError("invalid retained real training measurement")
    return True

def validate_full_training_diagnostics(report, raw_jsonl):
    """A complete success report must agree with every flushed arm progress event.

    Progress is written to a separate stderr JSONL stream before the next arm
    starts. On failure the process still exits nonzero and emits no success JSON.
    """
    try:
        actual = [json.loads(line) for line in raw_jsonl.splitlines()]
    except json.JSONDecodeError as error:
        raise ValueError("invalid full training diagnostic JSONL") from error
    expected = []
    for pair, run in enumerate(report["runs"]):
        for arm in (("historical", "reused") if pair % 2 == 0 else ("reused", "historical")):
            shared = {
                "schema": "fy-full-training-progress-v1",
                "source_commit": report["source_commit"],
                "backend": report["backend"],
                "pair": pair,
                "arm": arm,
            }
            expected.append({**shared, "event": "started"})
            expected.append({**shared, "event": "completed", "measurement": run[arm]})
    if actual != expected:
        raise ValueError("incomplete, failed or mismatched full training diagnostics")
    return True
