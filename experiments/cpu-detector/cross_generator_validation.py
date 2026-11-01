#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exploratory TD1 evaluation on an independently written RGB generator.

No images or weights are published. Observed accuracy (including failures) is
recorded, not made a new acceptance threshold or used on final holdout 909.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from diagnostic_validation import _iou, _prediction, LEGACY_TEST_FINGERPRINT

GENERATOR = "TD1-cross-generator-rectangles-v1"
SEED = 20261101
SIDES = ("left", "top", "right", "bottom")
POS_PER_CLASS_SIDE = 12
BACKGROUNDS = 24
COUNT = 2 * len(SIDES) * POS_PER_CLASS_SIDE + BACKGROUNDS
# Frozen from the standalone generator before any model inference.
FINGERPRINT = "41a28eaf9eede38fc1439a7dff1a0b976dfac05bffda19c556a6c7f873ec9f5f"


class XorShift:
    def __init__(self, index):
        self.state = int.from_bytes(hashlib.sha256(
            f"{GENERATOR}:{SEED}:{index}".encode("ascii")).digest()[:4], "little") or 1

    def below(self, limit):
        if limit <= 0:
            raise ValueError("empty range")
        value = self.state
        value ^= (value << 13) & 0xffffffff
        value ^= value >> 17
        value ^= (value << 5) & 0xffffffff
        self.state = value & 0xffffffff
        return self.state % limit


def rows():
    """Independent xorshift32 sampling, patterned backgrounds and textured boxes.

    Unlike TD1-synthetic-v1's LCG, samples contain gradients/stripes,
    non-flat targets and independently sampled right/bottom positions.
    """
    for index in range(COUNT):
        rng = XorShift(index)
        positive = index < COUNT - BACKGROUNDS
        side = SIDES[index // (2 * POS_PER_CLASS_SIDE)] if positive else "background"
        cls = (index // POS_PER_CLASS_SIDE) % 2 if positive else None
        kind = index % 4
        pixels = bytearray(64 * 64 * 3)
        for y in range(64):
            for x in range(64):
                texture = (x // 8 + y // 8) % 2 if kind == 1 else 0
                wave = x // 4 if kind == 2 else y // 4 if kind == 3 else 0
                base = min(75, rng.below(26) + 14 * texture + wave)
                at = (y * 64 + x) * 3
                pixels[at:at + 3] = bytes((base, min(75, base + rng.below(6)),
                                             min(75, base + rng.below(6))))
        corners = None
        geometry = (0.0, 0.0, 0.0, 0.0)
        if positive:
            width, height = 11 + rng.below(13), 11 + rng.below(13)
            x = rng.below(3) if side == "left" else (
                64 - width - rng.below(3) if side == "right"
                else 4 + rng.below(64 - width - 7))
            y = rng.below(3) if side == "top" else (
                64 - height - rng.below(3) if side == "bottom"
                else 4 + rng.below(64 - height - 7))
            for yy in range(y, y + height):
                for xx in range(x, x + width):
                    at = (yy * 64 + xx) * 3
                    channels = [rng.below(46), rng.below(46), rng.below(46)]
                    channels[cls * 2] = 185 + rng.below(71)
                    pixels[at:at + 3] = bytes(channels)
            corners = (x, y, x + width, y + height)
            geometry = ((x + width / 2) / 64, (y + height / 2) / 64,
                        width / 64, height / 64)
        yield {"index": index, "side": side, "class": cls,
               "corners": corners, "geometry": geometry, "pixels": bytes(pixels)}


def fingerprint(items):
    digest = hashlib.sha256(GENERATOR.encode("ascii"))
    for item in items:
        digest.update(bytes([255 if item["class"] is None else item["class"]]))
        digest.update(struct.pack("<4f", *item["geometry"]))
        digest.update(item["pixels"])
    return digest.hexdigest()


def cohort():
    items = list(rows())
    if len(items) != COUNT or fingerprint(items) != FINGERPRINT:
        raise ValueError("independent generator fingerprint drift")
    if len({hashlib.sha256(item["pixels"]).digest() for item in items}) != COUNT:
        raise ValueError("repeated cross-generator image")
    for item in items:
        box = item["corners"]
        if item["class"] is None:
            if box is not None or item["side"] != "background":
                raise ValueError("background has target")
        elif box is None or not (0 <= box[0] < box[2] <= 64 and 0 <= box[1] < box[3] <= 64):
            raise ValueError("invalid target box")
    return items


def summarize(items, predictions):
    if len(items) != COUNT or len(predictions) != COUNT:
        raise ValueError("incomplete cross-generator inference")
    grouped = {f"{side}_class{cls}": {"samples": 0, "hits": 0, "missing": 0,
               "wrong_class": 0, "iou_sum": 0.0} for side in SIDES for cls in (0, 1)}
    backgrounds = {"samples": 0, "false_positives": 0}
    for item, raw in zip(items, predictions):
        detection = _prediction(raw)
        if item["class"] is None:
            backgrounds["samples"] += 1
            backgrounds["false_positives"] += detection is not None
            continue
        bucket = grouped[f'{item["side"]}_class{item["class"]}']
        bucket["samples"] += 1
        if detection is None:
            bucket["missing"] += 1
        elif detection["class"] != item["class"]:
            bucket["wrong_class"] += 1
        else:
            overlap = _iou(detection["corners"], item["corners"])
            bucket["iou_sum"] += overlap
            bucket["hits"] += overlap >= 0.5
    result = {}
    for name, bucket in grouped.items():
        if bucket["samples"] != POS_PER_CLASS_SIDE:
            raise ValueError("wrong class/side count")
        result[name] = {key: bucket[key] for key in ("samples", "hits", "missing", "wrong_class")}
        result[name]["recall_iou50"] = bucket["hits"] / bucket["samples"]
        result[name]["mean_iou_including_misses"] = bucket["iou_sum"] / bucket["samples"]
    if backgrounds["samples"] != BACKGROUNDS:
        raise ValueError("wrong background count")
    result["background"] = backgrounds
    return result


def check_report(report, expected_commit, expected_backend):
    if (report.get("status") != "exploratory-cross-generator-only"
            or report.get("source_commit") != expected_commit
            or report.get("backend") != expected_backend
            or report.get("generator") != GENERATOR
            or report.get("seed") != SEED or report.get("count") != COUNT
            or report.get("fingerprint") != FINGERPRINT
            or report.get("final_holdout_evaluated") is not False
            or not isinstance(report.get("model_sha256"), str)
            or len(report["model_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in report["model_sha256"])):
        raise ValueError("cross-generator provenance mismatch")
    groups = report.get("groups")
    if not isinstance(groups, dict) or set(groups) != {
            *(f"{side}_class{cls}" for side in SIDES for cls in (0, 1)), "background"}:
        raise ValueError("cross-generator group mismatch")
    bg = groups["background"]
    if (not isinstance(bg, dict) or set(bg) != {"samples", "false_positives"}
            or type(bg["samples"]) is not int or bg["samples"] != BACKGROUNDS
            or type(bg["false_positives"]) is not int
            or not 0 <= bg["false_positives"] <= BACKGROUNDS):
        raise ValueError("cross-generator background mismatch")
    for name, record in groups.items():
        if name == "background":
            continue
        if not isinstance(record, dict) or set(record) != {
                "samples", "hits", "missing", "wrong_class", "recall_iou50", "mean_iou_including_misses"}:
            raise ValueError("cross-generator missing metrics")
        samples, hits, missing, wrong = (record[k] for k in ("samples", "hits", "missing", "wrong_class"))
        if (any(type(v) is not int or v < 0 for v in (samples, hits, missing, wrong))
                or samples != POS_PER_CLASS_SIDE or hits + missing + wrong > samples):
            raise ValueError("cross-generator impossible counts")
        recall, mean = record["recall_iou50"], record["mean_iou_including_misses"]
        maximum = (hits + 0.5 * (samples - missing - wrong - hits)) / samples
        if (any(type(v) not in (float, int) or not math.isfinite(v) for v in (recall, mean))
                or abs(recall - hits / samples) > 1e-12
                or not 0.5 * hits / samples - 1e-6 <= mean <= maximum + 1e-6):
            raise ValueError("cross-generator metric arithmetic")
    raw = report.get("predictions")
    if (not isinstance(raw, list) or len(raw) != COUNT
            or any(not isinstance(entry, dict) or set(entry) != {"index", "prediction"}
                   or type(entry["index"]) is not int or entry["index"] != index
                   for index, entry in enumerate(raw))):
        raise ValueError("cross-generator inference rows missing or reordered")
    # Full per-image prediction metadata (but no pixels/weights) makes grouped
    # accuracy independently recomputable and rejects fabricated summaries.
    if summarize(cohort(), [entry["prediction"] for entry in raw]) != groups:
        raise ValueError("cross-generator aggregate disagrees with raw inference")
    return True


def evaluate(binary, artifact, metrics, *, invoke=None):
    record = json.loads(Path(metrics).read_text())
    if (record.get("passed") is not True or record.get("steps") != 2000
            or record.get("training_seed") != 101 or record.get("test_seed") != 909
            or record.get("test_fingerprint") != LEGACY_TEST_FINGERPRINT
            or record.get("backend") not in ("burn-flex-0.22.0", "candle-cpu-0.11.0")
            or not isinstance(record.get("source_commit"), str)
            or len(record["source_commit"]) != 40):
        raise ValueError("incompatible trained-model provenance")
    samples = cohort()
    artifact = Path(artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    detections = []
    with tempfile.TemporaryDirectory(prefix="factsyolo-cross-generator-") as tmp:
        rgb = Path(tmp) / "sample.rgb"
        for sample in samples:
            rgb.write_bytes(sample["pixels"])
            if invoke is None:
                result = subprocess.run([str(binary), "infer", str(artifact), str(rgb)],
                                        check=True, capture_output=True, text=True, timeout=15)
                detections.append(json.loads(result.stdout))
            else:
                detections.append(invoke(sample))
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
        raise ValueError("model was modified by inference")
    report = {"status": "exploratory-cross-generator-only", "source_commit": record["source_commit"],
              "backend": record["backend"], "model_sha256": digest,
              "generator": GENERATOR, "seed": SEED, "count": COUNT,
              "fingerprint": FINGERPRINT, "final_holdout_evaluated": False,
              "groups": summarize(samples, detections),
              "predictions": [{"index": sample["index"], "prediction": found}
                              for sample, found in zip(samples, detections)]}
    check_report(report, record["source_commit"], record["backend"])
    return report


def main():
    if len(sys.argv) != 4:
        raise ValueError("usage: cross_generator_validation.py <fy-cpu-evaluate> <model.fytd> <metrics.json>")
    print(json.dumps(evaluate(*sys.argv[1:]), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
