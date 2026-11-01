#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only independent, leap-ahead LCG replay of the frozen TD1 disjoint-source test.

Never loads trained models or their predictions. Only selected source images are
rasterized; unselected deterministic RNG draws are skipped mathematically.
"""
import hashlib
import json
import struct

M = (1 << 64) - 1
A = 6364136223846793005
C = 1442695040888963407
SEED = 20261123
BUDGET = 8192
FINGERPRINT = "e156306394733b94c5d99daec4ae22647819cab6976bc3dd55938eb988500a56"
ORDER = ("left", "top", "right", "bottom", "background")


class Rng:
    def __init__(self, seed):
        self.state = seed

    def draw(self):
        self.state = (self.state * A + C) & M
        return self.state >> 32

    def skip(self, draws):
        # Fast affine exponentiation, modulo 2^64, independent of sample data.
        mult, add = 1, 0
        cmult, cadd = A, C
        while draws:
            if draws & 1:
                mult, add = mult * cmult & M, (add * cmult + cadd) & M
            cadd = (cmult + 1) * cadd & M
            cmult = cmult * cmult & M
            draws >>= 1
        self.state = (self.state * mult + add) & M


def mirror(pixels, horizontal, vertical):
    output = bytearray(len(pixels))
    for y in range(64):
        for x in range(64):
            sy = 63 - y if vertical else y
            sx = 63 - x if horizontal else x
            dst, src = (y * 64 + x) * 3, (sy * 64 + sx) * 3
            output[dst:dst + 3] = pixels[src:src + 3]
    return bytes(output)


def freeze():
    rng = Rng(SEED)
    groups = {key: [] for key in ORDER}
    left_sources = top_sources = 0
    last_source = None
    for index in range(BUDGET):
        cls = None if index % 5 == 0 else rng.draw() % 2
        x, y = 2 + rng.draw() % 40, 2 + rng.draw() % 40
        width, height = 12 + rng.draw() % 11, 12 + rng.draw() % 11
        background = cls is None and len(groups["background"]) < 48
        small = width * height <= 256
        left_only = x <= 2 and y > 2 and x + width < 62 and y + height < 62
        top_only = y <= 2 and x > 2 and x + width < 62 and y + height < 62
        take_left = cls is not None and small and left_only and left_sources < 48
        take_top = cls is not None and small and top_only and top_sources < 48
        if not (background or take_left or take_top):
            rng.skip(64 * 64 * 3 + (4 if cls is not None else 0))
            continue
        pixels = bytearray(rng.draw() % 12 for _ in range(64 * 64 * 3))
        if cls is not None:
            colour = [rng.draw() % 60 for _ in range(3)]
            colour[cls * 2] = 180 + rng.draw() % 76
            for yy in range(y, y + height):
                for xx in range(x, x + width):
                    at = (yy * 64 + xx) * 3
                    pixels[at:at + 3] = bytes(colour)
        if background:
            side = "background"
            horizontal = vertical = False
        elif take_left:
            horizontal = left_sources >= 24
            vertical = False
            side = "right" if horizontal else "left"
            left_sources += 1
        else:
            horizontal = False
            vertical = top_sources >= 24
            side = "bottom" if vertical else "top"
            top_sources += 1
        if horizontal:
            x = 64 - x - width
        if vertical:
            y = 64 - y - height
        groups[side].append((index, cls, x, y, width, height,
                             mirror(pixels, horizontal, vertical)))
        last_source = index
        if left_sources == 48 and top_sources == 48 and len(groups["background"]) == 48:
            break
    sizes = {name: len(rows) for name, rows in groups.items()}
    if sizes != {"left": 24, "top": 24, "right": 24,
                 "bottom": 24, "background": 48}:
        raise ValueError(f"incomplete preregistered source selection: {sizes}")
    digest = hashlib.sha256(b"TD1-four-side-disjoint-v1")
    source_ids = set()
    pixels_hashes = set()
    for name in ORDER:
        for index, cls, x, y, w, h, pixels in groups[name]:
            if index in source_ids:
                raise ValueError("reused source image")
            source_ids.add(index)
            pixels_hashes.add(hashlib.sha256(pixels).digest())
            digest.update(bytes((255 if cls is None else cls,)))
            digest.update(struct.pack("<4f", (x + w / 2) / 64,
                                      (y + h / 2) / 64, w / 64, h / 64))
            digest.update(pixels)
    if len(pixels_hashes) != 144 or last_source != 6876 or digest.hexdigest() != FINGERPRINT:
        raise ValueError("disjoint cohort identity/fingerprint drift")
    return {"seed": SEED, "candidates": BUDGET, "selected": sizes,
            "source_rows": len(source_ids), "unique_pixel_arrays": len(pixels_hashes),
            "last_selected_index": last_source, "fingerprint": digest.hexdigest()}


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True))
