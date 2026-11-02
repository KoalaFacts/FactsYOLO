#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exhaust the accepted TD1 rectangle geometry without training or holdout data.

Same foreground RGB and uniform background are required: tile foreground-pixel
counts then determine the complete 192-feature TD1 input. This enumeration does
NOT cover varying sample colors, backgrounds, multiple objects or natural images.
"""

import json
from itertools import combinations


def axis_overlap(start: int, size: int) -> tuple[int, ...]:
    """Number of pixels in each 8px block, clipped to the 64px image."""
    if not (2 <= start <= 41 and 12 <= size <= 22):
        raise ValueError('outside accepted TD1 rectangle geometry')
    return tuple(max(0, min(start + size, 8 * (i + 1)) - max(start, 8 * i))
                 for i in range(8))


def tile_counts(box: tuple[int, int, int, int]) -> bytes:
    """Exact foreground count of each 8x8 tile in row-major order (0..64)."""
    x1, y1, x2, y2 = box
    xs = axis_overlap(x1, x2 - x1)
    ys = axis_overlap(y1, y2 - y1)
    return bytes(y * x for y in ys for x in xs)


def box_iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def inrange_alias_report() -> dict:
    """Enumerate every 2..41 origin and 12..22 box dimension, 193600 total."""
    axes = {(start, width): axis_overlap(start, width)
            for start in range(2, 42) for width in range(12, 23)}
    first_by_signature: dict[bytes, tuple[int, int, int, int]] = {}
    duplicate_groups: dict[bytes, list[tuple[int, int, int, int]]] = {}
    checked = 0
    for y1 in range(2, 42):
        for height in range(12, 23):
            ys = axes[y1, height]
            for x1 in range(2, 42):
                for width in range(12, 23):
                    xs = axes[x1, width]
                    signature = bytes(y * x for y in ys for x in xs)
                    box = (x1, y1, x1 + width, y1 + height)
                    first = first_by_signature.get(signature)
                    if first is None:
                        first_by_signature[signature] = box
                    else:
                        duplicate_groups.setdefault(signature, [first]).append(box)
                    checked += 1

    pairs = ((box_iou(a, b), a, b)
             for boxes in duplicate_groups.values()
             for a, b in combinations(boxes, 2))
    # Tie-break by rectangle corners, independent of hash/dict iteration order.
    least_iou, a, b = min(pairs)
    count_pairs = sum(len(g) * (len(g) - 1) // 2 for g in duplicate_groups.values())
    return {
        'scope': 'filled axis-aligned rectangles, same foreground RGB and uniform background',
        'origin_range': [2, 41],
        'width_height_range': [12, 22],
        'total_rectangles': checked,
        'unique_tile_count_signatures': len(first_by_signature),
        'ambiguous_groups': len(duplicate_groups),
        'colliding_pairs': count_pairs,
        'max_group_size': max(map(len, duplicate_groups.values()), default=1),
        'minimum_pairwise_iou': least_iou,
        'minimum_pair': [list(a), list(b)],
        'every_colliding_pair_has_a_shared_iou50_box': least_iou >= 0.5,
        'training_executed': False,
        'final_holdout_evaluated': False,
        'detector_accuracy_claimed': False,
    }


if __name__ == '__main__':
    print(json.dumps(inrange_alias_report(), indent=2, sort_keys=True))
