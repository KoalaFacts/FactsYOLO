#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-registered, development-only TD1 small-object validation.

Runs freshly trained artifacts through the existing portable infer executable.
All 256 samples are new synthetic validation samples, not the final test split.
No model selection, altered training budget, or new acceptance threshold follows
from this diagnostic. Only aggregate evidence is emitted; RGB bytes are temporary.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from localization_decomposition import decompose

# Registered before any trained-model evaluation; 909/128 remains the final test.
DIAGNOSTIC_SEED = 20261028
DIAGNOSTIC_COUNT = 256
SMALL_AREA_MAX = 256  # 16x16 pixels; fixed independently of prediction quality.
GENERATOR = "TD1-synthetic-v1"
# This fingerprint is computed solely from the Rust-compatible generator,
# independent of trained weights or predictions. Changes fail closed.
DIAGNOSTIC_FINGERPRINT = "3f57fc1b8f9dcad73d40de2bbdd6b253e363e7206ee7a06ea454b155b90880a4"
LEGACY_TEST_FINGERPRINT = "188366b67882eb3599300ac6b056c31b5b28e41733ab801fef1362bb05354c33"


class Rng:
    def __init__(self, seed):
        self.state = seed

    def next_u32(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.state >> 32


def samples(seed, count):
    """Independent stdlib reproduction of model/src/data.rs::dataset."""
    rng = Rng(seed)
    for index in range(count):
        cls = None if index % 5 == 0 else rng.next_u32() % 2
        x = 2 + rng.next_u32() % 40
        y = 2 + rng.next_u32() % 40
        width = 12 + rng.next_u32() % 11
        height = 12 + rng.next_u32() % 11
        pixels = bytearray(rng.next_u32() % 12 for _ in range(64 * 64 * 3))
        if cls is not None:
            color = [rng.next_u32() % 60 for _ in range(3)]
            color[cls * 2] = 180 + rng.next_u32() % 76
            for py in range(y, y + height):
                for px in range(x, x + width):
                    at = (py * 64 + px) * 3
                    pixels[at:at+3] = bytes(color)
        geometry = ((x + width / 2) / 64, (y + height / 2) / 64,
                    width / 64, height / 64)
        yield {"index": index, "class": cls, "geometry": geometry,
               "corners": (x, y, x + width, y + height),
               "area": width * height, "pixels": bytes(pixels)}


def fingerprint(rows):
    digest = hashlib.sha256(GENERATOR.encode("ascii"))
    for row in rows:
        digest.update(bytes([255 if row["class"] is None else row["class"]]))
        digest.update(struct.pack("<4f", *row["geometry"]))
        digest.update(row["pixels"])
    return digest.hexdigest()


def _prediction(value):
    if value is None:
        return None
    if not isinstance(value, dict) or type(value.get("class")) is not int or value["class"] not in (0, 1):
        raise ValueError("invalid detection class")
    score = value.get("score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0.5 <= score <= 1.0:
        raise ValueError("invalid detection score")
    corners = value.get("corners")
    if (not isinstance(corners, list) or len(corners) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in corners)):
        raise ValueError("invalid predicted corners")
    x1, y1, x2, y2 = corners
    if not (0 <= x1 <= x2 <= 64 and 0 <= y1 <= y2 <= 64):
        raise ValueError("invalid predicted box geometry")
    return value


def _iou(a, b):
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    overlap = max(0.0, min(x2, u2) - max(x1, u1)) * max(0.0, min(y2, v2) - max(y1, v1))
    area_a = (x2 - x1) * (y2 - y1)
    area_b = (u2 - u1) * (v2 - v1)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def summarize(rows, predictions):
    if len(rows) != len(predictions) or not rows:
        raise ValueError("missing or extra diagnostic predictions")
    groups = {"small": [], "larger": []}
    backgrounds = 0
    false_positives = 0
    for row, raw in zip(rows, predictions):
        pred = _prediction(raw)
        if row["class"] is None:
            backgrounds += 1
            false_positives += pred is not None
            continue
        target = row["corners"]
        name = "small" if row["area"] <= SMALL_AREA_MAX else "larger"
        correct = pred is not None and pred["class"] == row["class"]
        overlap = _iou(pred["corners"], target) if pred is not None else 0.0
        if pred is None:
            center_error = None
        else:
            x1, y1, x2, y2 = pred["corners"]
            tx1, ty1, tx2, ty2 = target
            center_error = (((x1 + x2) - (tx1 + tx2)) / 2.0,
                            ((y1 + y2) - (ty1 + ty2)) / 2.0)
        groups[name].append((correct, overlap, center_error))
    if not backgrounds or not groups["small"] or not groups["larger"]:
        raise ValueError("diagnostic needs backgrounds, small and larger targets")

    result = {"samples": len(rows), "backgrounds": backgrounds,
              "false_positives": false_positives,
              "background_false_positive_rate": false_positives / backgrounds}
    for name, group in groups.items():
        n = len(group)
        centers = [v[2] for v in group if v[2] is not None]
        result[name] = {
            "positives": n,
            "class_accuracy": sum(v[0] for v in group) / n,
            "recall_correct_class_iou50": sum(v[0] and v[1] >= 0.5 for v in group) / n,
            "mean_iou_including_misses": sum(v[1] for v in group) / n,
            "missing_detections": n - len(centers),
            "mean_signed_center_x_error_px_detected": sum(v[0] for v in centers) / len(centers) if centers else None,
            "mean_signed_center_y_error_px_detected": sum(v[1] for v in centers) / len(centers) if centers else None,
        }
    return result


def evaluate(binary, artifact, training_metrics, *, invoke=None):
    """Fail closed on generator drift, artifact mutation, or invalid infer output."""
    training = json.loads(Path(training_metrics).read_text())
    if (training.get("passed") is not True or training.get("steps") != 2000 or
            training.get("training_seed") != 101 or training.get("test_seed") != 909 or
            training.get("test_fingerprint") != LEGACY_TEST_FINGERPRINT or
            training.get("source_commit") in (None, "unknown")):
        raise ValueError("incompatible training provenance")
    rows = list(samples(DIAGNOSTIC_SEED, DIAGNOSTIC_COUNT))
    if fingerprint(rows) != DIAGNOSTIC_FINGERPRINT:
        raise ValueError("registered diagnostic generator fingerprint mismatch")
    model_hash = hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
    detections = []
    with tempfile.TemporaryDirectory(prefix="factsyolo-diagnostic-") as root:
        rgb_file = Path(root) / "sample.rgb"
        for row in rows:
            rgb_file.write_bytes(row["pixels"])
            if invoke is None:
                completed = subprocess.run([str(binary), "infer", str(artifact), str(rgb_file)],
                                           capture_output=True, text=True, timeout=15, check=True)
                detection = json.loads(completed.stdout)
            else:
                detection = invoke(row)
            detections.append(detection)
    if hashlib.sha256(Path(artifact).read_bytes()).hexdigest() != model_hash:
        raise ValueError("model artifact changed during diagnostic")
    metrics = summarize(rows, detections)
    localization = decompose(rows, detections)
    return {"status": "development-diagnostic-only", "source_commit": training["source_commit"],
            "backend": training["backend"], "trained_model_sha256": model_hash,
            "generator": GENERATOR, "seed": DIAGNOSTIC_SEED,
            "count": DIAGNOSTIC_COUNT, "fingerprint": DIAGNOSTIC_FINGERPRINT,
            "small_target_area_max_pixels": SMALL_AREA_MAX,
            "metrics": metrics, "localization_diagnostic": localization}


def main():
    if len(sys.argv) != 4:
        raise ValueError("usage: diagnostic_validation.py <fy-cpu-evaluate> <model.fytd> <metrics.json>")
    print(json.dumps(evaluate(*sys.argv[1:]), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
