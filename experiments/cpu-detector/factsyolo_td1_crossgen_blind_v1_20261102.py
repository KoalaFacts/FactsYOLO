#!/usr/bin/env python3
"""Pre-registered, *prediction-blind* TD1 synthetic cross-generator cohort.

Scope: exactly 256 64x64 interleaved RGB image samples. This program generates
labels and pixels independently of FactsYOLO's training data/generator. It never
loads models, prior predictions, weights or any GitHub evidence file.

This cohort is a research diagnostic, not natural-image or production evidence.
The frozen manifest has no weight/pixel uploads and the CLI doesn't materialize
raw image files on disk. To run future model inference, a separately reviewed
adapter may consume `samples()` and hand the bytes to the existing portable
inference process. Freeze before any model predictions, never regenerate based
on results, and never use #56/#58 previously observed images as new holdouts.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

VERSION = "facts-yolo-td1-crossgen-blind-neutral-v1"
SEED = 0x20261102B17D
SIZE = 64
POS_PER_SIDE = 48
NEG_PER_PATTERN = 16
SIDES = ("left", "top", "right", "bottom")
PATTERNS = ("uniform", "checker", "neutral_rectangle", "stripes")


class SplitMix64:
    """Local, version-stable generator; no host Python random() dependency."""
    def __init__(self, state: int) -> None:
        self.state = state & ((1 << 64) - 1)

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)

    def between(self, low: int, high: int) -> int:
        assert low <= high
        return low + self.next() % (high - low + 1)


@dataclass(frozen=True)
class Case:
    index: int
    cls: int | None
    side: str | None
    background: str
    box: tuple[int, int, int, int] | None  # pixel corners x0,y0,x1,y1, exclusive
    pixels: bytes


def _draw_background(rng: SplitMix64, pattern: str) -> bytearray:
    base = rng.between(34, 144)
    light = rng.between(11, 36)
    checker = rng.between(6, 19)
    rect_x = rng.between(5, 32)
    rect_y = rng.between(4, 34)
    rect_w = rng.between(8, 25)
    rect_h = rng.between(7, 22)
    stripe_width = rng.between(4, 11)
    out = bytearray(SIZE * SIZE * 3)
    for y in range(SIZE):
        for x in range(SIZE):
            jitter = rng.between(-4, 4)
            if pattern == "uniform":
                g = base + jitter
            elif pattern == "checker":
                g = base + (checker if (x // 8 + y // 8) % 2 else -checker) + jitter
            elif pattern == "neutral_rectangle":
                inside = rect_x <= x < rect_x + rect_w and rect_y <= y < rect_y + rect_h
                g = base + (light + 12 if inside else -7) + jitter
            elif pattern == "stripes":
                g = base + (light if (x // stripe_width) % 2 else -light) + jitter
            else:
                raise AssertionError(pattern)
            g = max(1, min(238, g))
            at = (y * SIZE + x) * 3
            out[at:at+3] = bytes([g, g, g])
    return out


def _paint_target(rgb: bytearray, box: tuple[int, int, int, int], cls: int, rng: SplitMix64) -> None:
    x0, y0, x1, y1 = box
    high = rng.between(211, 251)
    low_a = rng.between(11, 32)
    low_b = rng.between(13, 36)
    rgb_triplet = (high, low_a, low_b) if cls == 0 else (low_a, low_b, high)
    for y in range(y0, y1):
        for x in range(x0, x1):
            at = (y * SIZE + x) * 3
            rgb[at:at+3] = bytes(rgb_triplet)


def _box(rng: SplitMix64, side: str) -> tuple[int, int, int, int]:
    w = rng.between(7, 20)
    h = rng.between(7, 20)
    # Edges genuinely appear within 0..2 pixels, without corner confounds.
    edge = rng.between(0, 2)
    if side == "left":
        x0, y0 = edge, rng.between(10, SIZE - h - 10)
    elif side == "right":
        x0, y0 = SIZE - w - edge, rng.between(10, SIZE - h - 10)
    elif side == "top":
        x0, y0 = rng.between(10, SIZE - w - 10), edge
    elif side == "bottom":
        x0, y0 = rng.between(10, SIZE - w - 10), SIZE - h - edge
    else:
        raise AssertionError(side)
    return x0, y0, x0 + w, y0 + h


def samples():
    rng = SplitMix64(SEED)
    # Explicit balanced design; no prediction-dependent filtering or retries.
    index = 0
    for side in SIDES:
        for i in range(POS_PER_SIDE):
            cls = i % 2
            background = PATTERNS[i % len(PATTERNS)]
            pix = _draw_background(rng, background)
            box = _box(rng, side)
            _paint_target(pix, box, cls, rng)
            yield Case(index, cls, side, background, box, bytes(pix))
            index += 1
    for pattern in PATTERNS:
        for _ in range(NEG_PER_PATTERN):
            pix = _draw_background(rng, pattern)
            yield Case(index, None, None, pattern, None, bytes(pix))
            index += 1


def validate(cases: list[Case]) -> dict:
    assert len(cases) == 256
    assert [s.index for s in cases] == list(range(len(cases)))
    assert len({hashlib.sha256(s.pixels).digest() for s in cases}) == len(cases)
    assert all(len(c.pixels) == SIZE * SIZE * 3 for c in cases)
    assert Counter(c.side for c in cases if c.cls is not None) == {k: POS_PER_SIDE for k in SIDES}
    assert Counter(c.background for c in cases if c.cls is None) == {k: NEG_PER_PATTERN for k in PATTERNS}
    assert Counter((c.side, c.cls) for c in cases if c.cls is not None) == {(k, cls): 24 for k in SIDES for cls in (0, 1)}
    assert Counter((c.side, c.background) for c in cases if c.cls is not None) == {(k, p): 12 for k in SIDES for p in PATTERNS}
    assert sum(c.cls is None for c in cases) == 64
    chroma_object_pixels = 0
    for c in cases:
        if c.cls is None:
            assert c.side is None and c.box is None
            assert all(c.pixels[i] == c.pixels[i+1] == c.pixels[i+2]
                       for i in range(0, len(c.pixels), 3))
            continue
        x0, y0, x1, y1 = c.box
        assert 0 <= x0 < x1 <= SIZE and 0 <= y0 < y1 <= SIZE
        edge_distance = {"left": x0, "top": y0, "right": SIZE - x1, "bottom": SIZE - y1}[c.side]
        assert edge_distance in (0, 1, 2)
        assert 7 <= x1-x0 <= 20 and 7 <= y1-y0 <= 20
        for y in range(SIZE):
            for x in range(SIZE):
                at = (y * SIZE + x) * 3
                r, g, b = c.pixels[at:at+3]
                inside = x0 <= x < x1 and y0 <= y < y1
                if inside:
                    chroma_object_pixels += 1
                    assert (r > 200 and r > b + 150) if c.cls == 0 else (b > 200 and b > r + 150)
                else:
                    assert r == g == b  # no background class-colour cue
    return {
        "count": len(cases), "positives": 192, "negatives": 64,
        "by_side": dict(Counter(c.side for c in cases if c.cls is not None)),
        "by_negative_pattern": dict(Counter(c.background for c in cases if c.cls is None)),
        "red_positive": sum(c.cls == 0 for c in cases),
        "blue_positive": sum(c.cls == 1 for c in cases),
        "object_pixels_checked": chroma_object_pixels,
        "unique_rgb_frames": len(cases),
    }


def manifest() -> dict:
    cases = list(samples())
    summary = validate(cases)
    digest = hashlib.sha256()
    digest.update(VERSION.encode("ascii") + b"\0")
    image_digests = []
    for c in cases:
        # Unambiguous serialization of image + label + exact geometry.
        digest.update(struct.pack("<IB", c.index, 255 if c.cls is None else c.cls))
        digest.update(struct.pack("<4h", *(c.box if c.box is not None else (-1, -1, -1, -1))))
        digest.update(c.pixels)
        image_digests.append(hashlib.sha256(c.pixels).hexdigest())
    return {
        "schema": "facts-yolo-blind-cohort-pre-registration-1",
        "generator_version": VERSION,
        "seed_hex": f"0x{SEED:016x}",
        "rgb_format": "64x64 RGB interleaved uint8, row-major, no header",
        "boxes": "pixel x0,y0,x1,y1 exclusive; class=0 red, class=1 blue, null background",
        "purpose": "Exploratory prediction-blind cross-generator diagnostics; no claim of production accuracy or independent natural-image validation",
        "predeclared_comparison": {
            "arms": ["mirror", "neutral-background"],
            "pairing": "same images in exact index order, same trained model config for both arms, both Burn/Candle on Linux/Windows",
            "positive_measure": "correct-class IoU >=0.5, missing and wrong-class count as misses; mean IoU includes all misses",
            "background_measure": "false detections at existing unchanged score threshold, by neutral pattern and overall",
            "candidate_rule": "No default promotion. Further exploration only if false detections strictly decrease on both platforms and both backends, total positive IoU50 recall does not decrease, and each of four sides loses <=2/48 hits vs mirror. Report all negatives even if criteria fail",
            "forbidden": "No threshold tuning, no new RNG/model/optimizer settings based on this frozen cohort, no reusing seen #56/#58 images as holdout",
        },
        "validation": summary,
        "cohort_sha256": digest.hexdigest(),
        "image_sha256": image_digests,
        "model_predictions_observed": False,
        "model_weights_loaded": False,
        "holdout_status": "frozen-pre-inference",
    }


def test_negative_controls() -> None:
    # Fail tests on generator drift or annotation errors, not just statistics.
    rows = list(samples())
    good = validate(rows)
    assert good["count"] == 256
    bad = rows.copy()
    old = bad[0]
    bad[0] = Case(old.index, 1 - old.cls, old.side, old.background, old.box, old.pixels)
    try:
        validate(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("swapped class labels accepted")
    bad = rows.copy()
    old = bad[191]
    x0,y0,x1,y1 = old.box
    bad[191] = Case(old.index, old.cls, old.side, old.background,
                    (x0, y0, x1-1, y1), old.pixels)
    try:
        validate(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("drifted target rectangle accepted")
    bad = rows.copy()
    old = bad[200]
    pix = bytearray(old.pixels)
    pix[0] = 255
    bad[200] = Case(old.index, None, None, old.background, None, bytes(pix))
    try:
        validate(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("false-color background accepted")
    bad = rows.copy()
    bad[-1] = bad[-2]
    try:
        validate(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicated images accepted")


if __name__ == "__main__":
    test_negative_controls()
    output = Path(__file__).with_name("factsyolo_td1_crossgen_blind_v1_20261102_manifest.json")
    result = manifest()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "sha256": result["cohort_sha256"],
                      "validation": result["validation"], "negative_controls": 4,
                      "model_predictions_observed": False}, indent=2))
