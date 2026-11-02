#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A constructed, model-free negative control for TD1's spatial resolution.

This deliberately does not read model weights, an evaluation cohort, or the
held-out TD1 seed. It proves a representational limit rather than an observed
accuracy result. The 4x4 encoder is an unapproved candidate for a future model,
not an altered TD1 encoder or FYTDv001 format.
"""

import hashlib
import json
import math
import struct

SIDE = 64
CHANNELS = 3
PIXELS = SIDE * SIDE * CHANNELS


def block_features(pixels: bytes, cell_size: int) -> tuple[float, ...]:
    """F32 normalized RGB block means in TD1's row/column/channel order."""
    if cell_size not in (4, 8) or len(pixels) != PIXELS:
        raise ValueError('expected 64x64 RGB and one of 4x4 or 8x8 pooling')
    blocks = SIDE // cell_size
    out = []
    for by in range(blocks):
        for bx in range(blocks):
            for channel in range(CHANNELS):
                total = sum(
                    pixels[(y * SIDE + x) * CHANNELS + channel]
                    for y in range(by * cell_size, (by + 1) * cell_size)
                    for x in range(bx * cell_size, (bx + 1) * cell_size)
                )
                # Explicit F32 conversion, like TD1. All sums fit exactly
                # in both Python integers and Rust u32/f32.
                out.append(struct.unpack('<f', struct.pack('<f', total / (cell_size ** 2 * 255)))[0])
    return tuple(out)


def candidate_pair() -> tuple[tuple[bytes, tuple[int, ...]], ...]:
    """Same color/area in opposite 4x4 quadrants of one 8x8 tile."""
    samples = []
    for corner in ((0, 0), (4, 4)):
        pixels = bytearray([7] * PIXELS)
        for y in range(corner[1], corner[1] + 4):
            for x in range(corner[0], corner[0] + 4):
                at = (y * SIDE + x) * CHANNELS
                pixels[at:at + 3] = bytes((230, 20, 20))
        samples.append((bytes(pixels), (corner[0], corner[1], corner[0] + 4, corner[1] + 4)))
    return tuple(samples)


def _area(box):
    if (len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or not (0 <= box[0] < box[2] <= SIDE and 0 <= box[1] < box[3] <= SIDE)):
        raise ValueError('invalid pixel-edge rectangle')
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a, b):
    aa, ba = _area(a), _area(b)
    cross = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    return cross / (aa + ba - cross)


def spatial_alias_report():
    (a, abox), (b, bbox) = candidate_pair()
    coarse_a = block_features(a, 8)
    coarse_b = block_features(b, 8)
    fine_a = block_features(a, 4)
    fine_b = block_features(b, 4)
    if coarse_a != coarse_b or fine_a == fine_b:
        raise RuntimeError('unexpected feature-encoder witness')
    area = _area(abox)
    min_cover = (
        max(abox[2], bbox[2]) - min(abox[0], bbox[0])
    ) * (max(abox[3], bbox[3]) - min(abox[1], bbox[1]))
    if _area(bbox) != area or iou(abox, bbox) != 0 or min_cover <= 2 * area:
        raise RuntimeError('invalid geometric impossibility witness')
    # Proof: For disjoint equal-area targets A,B of area S, predicting P with
    # IoU(P,A)>=.5 and IoU(P,B)>=.5 requires |P|<=2S and
    # |P∩A|, |P∩B| >= (|P|+S)/3. As intersections are disjoint and both
    # <=S, these inequalities force |P|=2S and both intersections=S.
    # So P must exactly contain both targets and have area 2S. Their
    # minimum axis-aligned enclosing rectangle has area 4S: contradiction.
    original_params = 192 * 64 + 64 + 64 * 7 + 7
    proposed_params = 768 * 64 + 64 + 64 * 7 + 7
    return {
        'status': 'constructed-spatial-alias-negative-control',
        'model': 'TD1-v1-192-64-7',
        'source': 'two independent single-object 64x64 images; no external data',
        'training_executed': False,
        'final_holdout_evaluated': False,
        'detector_accuracy_claimed': False,
        'pixel_sha256': [hashlib.sha256(p).hexdigest() for p in (a, b)],
        'target_boxes': [list(abox), list(bbox)],
        'target_class': 0,
        'current_feature_count': len(coarse_a),
        'candidate_feature_count': len(fine_a),
        'coarse_feature_sha256': hashlib.sha256(struct.pack('<192f', *coarse_a)).hexdigest(),
        'coarse_features_identical': True,
        'fine_features_different': True,
        'fine_changed_feature_values': sum(x != y for x, y in zip(fine_a, fine_b)),
        'minimum_cover_area': min_cover,
        'sum_of_target_areas': 2 * area,
        'common_iou50_possible': False,
        'max_correct_class_iou50_recall_for_pair': 0.5,
        'current_parameter_count': original_params,
        'candidate_parameter_count': proposed_params,
        'additional_f32_parameter_bytes': (proposed_params - original_params) * 4,
        'candidate_status': 'unapproved feature/architecture change; research reference only',
    }


if __name__ == '__main__':
    print(json.dumps(spatial_alias_report(), indent=2, sort_keys=True))
