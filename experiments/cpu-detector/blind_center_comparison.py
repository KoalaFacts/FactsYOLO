#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preregistered paired TD1 mirror-vs-center evaluation, synthetic only.

Never select a treatment from these results. Only ephemeral RGB/model files are
read; JSON records exact model identities and per-image detections, not pixels.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

GENERATOR = 'TD1-independent-striped-boxes-v2'
SEED = 20261102
SIDES = ('left', 'top', 'right', 'bottom')
PER_CLASS_SIDE = 12
BACKGROUNDS = 24
COUNT = 2 * len(SIDES) * PER_CLASS_SIDE + BACKGROUNDS
# Registered from generator-only evaluation, before *any* trained model inference.
FINGERPRINT = 'aed8aaef623c07588159e3319ce59acdd02c0e41ad1622abbaf47a315c2f3ab2'
BACKENDS = ('burn-flex-0.22.0', 'candle-cpu-0.11.0')


class Rng:
    def __init__(self, index):
        self.value = int.from_bytes(hashlib.sha256(
            f'{GENERATOR}:{SEED}:{index}'.encode('ascii')).digest()[:4], 'little') or 1

    def pick(self, max_exclusive):
        if max_exclusive <= 0:
            raise ValueError('invalid generator range')
        v = self.value
        v ^= (v << 13) & 0xffffffff
        v ^= v >> 17
        v ^= (v << 5) & 0xffffffff
        self.value = v & 0xffffffff
        return self.value % max_exclusive


def rows():
    """Independent procedural stripes/noise, not TD1 synthetic-v1/old cohort."""
    for index in range(COUNT):
        rng = Rng(index)
        positive = index < COUNT - BACKGROUNDS
        side = SIDES[index // (2 * PER_CLASS_SIDE)] if positive else 'background'
        cls = (index // PER_CLASS_SIDE) % 2 if positive else None
        pixels = bytearray(64 * 64 * 3)
        for y in range(64):
            for x in range(64):
                # Spatially varying but class-neutral background, with local
                # channel differences. Distinct from old checker/wave pattern.
                stripe = 13 * ((x + 2 * y + index) // 7 % 3)
                band = 7 * ((x // 5 + index) % 3)
                base = 9 + stripe + band + rng.pick(17)
                at = 3 * (y * 64 + x)
                pixels[at:at + 3] = bytes((base + rng.pick(9),
                                             base + rng.pick(9),
                                             base + rng.pick(9)))
        box = None
        xywh = (0.0, 0.0, 0.0, 0.0)
        if positive:
            w, h = 11 + rng.pick(13), 11 + rng.pick(13)
            x = (rng.pick(4) if side == 'left' else
                 (64 - w - rng.pick(4) if side == 'right' else
                  5 + rng.pick(64 - w - 9)))
            y = (rng.pick(4) if side == 'top' else
                 (64 - h - rng.pick(4) if side == 'bottom' else
                  5 + rng.pick(64 - h - 9)))
            for py in range(y, y+h):
                for px in range(x, x+w):
                    at = 3 * (py * 64 + px)
                    # Non-flat coloured target with horizontal local bands;
                    # same red/blue class semantics as TD1, no label leakage.
                    channels = [rng.pick(35), rng.pick(35), rng.pick(35)]
                    channels[cls * 2] = 185 + (py // 3 % 3) * 9 + rng.pick(31)
                    pixels[at:at + 3] = bytes(channels)
            box = (x, y, x+w, y+h)
            xywh = ((x+w/2)/64, (y+h/2)/64, w/64, h/64)
        yield {'index': index, 'side': side, 'class': cls, 'pixels': bytes(pixels),
               'corners': box, 'geometry': xywh}


def fingerprint(items):
    digest = hashlib.sha256(GENERATOR.encode('ascii'))
    for item in items:
        digest.update(bytes([255 if item['class'] is None else item['class']]))
        digest.update(struct.pack('<4f', *item['geometry']))
        digest.update(item['pixels'])
    return digest.hexdigest()


def cohort():
    items = list(rows())
    if len(items) != COUNT or fingerprint(items) != FINGERPRINT:
        raise ValueError('frozen blind generator fingerprint mismatch')
    if len({hashlib.sha256(row['pixels']).digest() for row in items}) != COUNT:
        raise ValueError('duplicate frozen image')
    for item in items:
        if len(item['pixels']) != 12288:
            raise ValueError('invalid RGB size')
        if item['class'] is None:
            if item['side'] != 'background' or item['corners'] is not None:
                raise ValueError('background target present')
        else:
            a, b, c, d = item['corners']
            if not (0 <= a < c <= 64 and 0 <= b < d <= 64):
                raise ValueError('invalid true box')
    return items


def parse_prediction(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'class', 'score', 'corners'}:
        raise ValueError('unknown detection shape')
    if type(value['class']) is not int or value['class'] not in (0, 1):
        raise ValueError('invalid detection class')
    s = value['score']
    if type(s) not in (float, int) or not math.isfinite(s) or not 0.5 <= s <= 1:
        raise ValueError('invalid detection score')
    box = value['corners']
    if not isinstance(box, list) or len(box) != 4 or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in box):
        raise ValueError('invalid predicted corners')
    a, b, c, d = box
    if not (0 <= a <= c <= 64 and 0 <= b <= d <= 64):
        raise ValueError('invalid predicted geometry')
    return value


def iou(a, b):
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    overlap = max(0, min(x2, u2)-max(x1, u1))*max(0, min(y2, v2)-max(y1, v1))
    area_a = (x2-x1)*(y2-y1)
    area_b = (u2-u1)*(v2-v1)
    union = area_a+area_b-overlap
    return overlap/union if union > 0 else 0.0


def summarize(items, predictions):
    if len(items) != COUNT or len(predictions) != COUNT:
        raise ValueError('incomplete paired inference')
    groups = {f'{side}_class{cls}': {'samples': 0, 'hits': 0, 'missing': 0,
               'wrong_class': 0, 'iou_sum': 0.0, 'center_error_sum': 0.0,
               'correct_class_boxes': 0} for side in SIDES for cls in (0, 1)}
    background = {'samples': 0, 'false_positives': 0}
    for sample, raw in zip(items, predictions):
        prediction = parse_prediction(raw)
        if sample['class'] is None:
            background['samples'] += 1
            background['false_positives'] += prediction is not None
            continue
        bucket = groups[f"{sample['side']}_class{sample['class']}"]
        bucket['samples'] += 1
        if prediction is None:
            bucket['missing'] += 1
        elif prediction['class'] != sample['class']:
            bucket['wrong_class'] += 1
        else:
            a, b, c, d = prediction['corners']
            u, v, w, z = sample['corners']
            bucket['center_error_sum'] += math.hypot((a+c-u-w)/2, (b+d-v-z)/2)
            bucket['correct_class_boxes'] += 1
            overlap = iou(prediction['corners'], sample['corners'])
            bucket['iou_sum'] += overlap
            bucket['hits'] += overlap >= 0.5
    result = {}
    for name, bucket in groups.items():
        if bucket['samples'] != PER_CLASS_SIDE:
            raise ValueError('invalid side population')
        hits = bucket['hits']
        result[name] = {'samples': bucket['samples'], 'hits': hits,
                        'missing': bucket['missing'], 'wrong_class': bucket['wrong_class'],
                        'recall_iou50': hits/bucket['samples'],
                        'mean_iou_including_misses': bucket['iou_sum']/bucket['samples'],
                        'correct_class_boxes': bucket['correct_class_boxes'],
                        'mean_center_error_px': (
                            bucket['center_error_sum']/bucket['correct_class_boxes']
                            if bucket['correct_class_boxes'] else None)}
    if background['samples'] != BACKGROUNDS:
        raise ValueError('invalid background population')
    result['background'] = background
    return result


def make_report(items, mirror_predictions, center_predictions, commit, backend,
                mirror_hash, center_hash):
    mirror_groups = summarize(items, mirror_predictions)
    center_groups = summarize(items, center_predictions)
    hits = lambda groups: sum(groups[f'{side}_class{cls}']['hits']
                              for side in SIDES for cls in (0, 1))
    return {
        'status': 'synthetic-blind-paired-diagnostic-only',
        'source_commit': commit, 'backend': backend,
        'generator': GENERATOR, 'seed': SEED, 'fingerprint': FINGERPRINT,
        'count': COUNT, 'final_holdout_evaluated': False,
        'training_steps': 2000, 'sampler_seed': 707, 'initialization_seed': 404,
        'mirror': {'model_sha256': mirror_hash, 'gradient_multiplier': 1.0,
                   'groups': mirror_groups,
                   'predictions': [{'index': row['index'], 'prediction': value}
                                   for row, value in zip(items, mirror_predictions)]},
        'center': {'model_sha256': center_hash, 'gradient_multiplier': 2.0,
                   'groups': center_groups,
                   'predictions': [{'index': row['index'], 'prediction': value}
                                   for row, value in zip(items, center_predictions)]},
        'comparison': {'mirror_hits': hits(mirror_groups),
                       'center_hits': hits(center_groups),
                       'center_minus_mirror_hits': hits(center_groups)-hits(mirror_groups),
                       'mirror_background_fp': mirror_groups['background']['false_positives'],
                       'center_background_fp': center_groups['background']['false_positives']}
    }


def check_report(report, commit, backend, mirror_hash, center_hash):
    if (not isinstance(report, dict) or
            report.get('source_commit') != commit or report.get('backend') != backend or
            report.get('status') != 'synthetic-blind-paired-diagnostic-only' or
            report.get('generator') != GENERATOR or report.get('seed') != SEED or
            report.get('fingerprint') != FINGERPRINT or report.get('count') != COUNT or
            report.get('final_holdout_evaluated') is not False or
            report.get('training_steps') != 2000 or report.get('sampler_seed') != 707 or
            report.get('initialization_seed') != 404):
        raise ValueError('blind comparison provenance mismatch')
    items = cohort()
    arrays = []
    for arm, expected_sha, multiplier in (
            ('mirror', mirror_hash, 1.0), ('center', center_hash, 2.0)):
        section = report.get(arm)
        if (not isinstance(section, dict) or section.get('model_sha256') != expected_sha
                or section.get('gradient_multiplier') != multiplier or
                not isinstance(section.get('predictions'), list) or
                len(section['predictions']) != COUNT):
            raise ValueError('blind arm provenance mismatch')
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64 or
                any(c not in '0123456789abcdef' for c in expected_sha)):
            raise ValueError('invalid artifact digest')
        predictions = []
        for index, record in enumerate(section['predictions']):
            if (not isinstance(record, dict) or set(record) != {'index', 'prediction'}
                    or type(record['index']) is not int or record['index'] != index):
                raise ValueError('missing or reordered raw predictions')
            predictions.append(record['prediction'])
        if summarize(items, predictions) != section.get('groups'):
            raise ValueError('blind arm aggregate mismatch')
        arrays.append(predictions)
    expected = make_report(items, *arrays, commit, backend, mirror_hash, center_hash)
    if report != expected:
        raise ValueError('blind comparison does not replay from raw detections')
    return True


def check_training(record, *, center, commit, backend, digest):
    if (not isinstance(record, dict) or
            record.get('status') != ('center-weighted-mirror-training-experiment-only' if center
                                     else 'mirrored-edge-training-experiment-only') or
            record.get('source_commit') != commit or record.get('backend') != backend or
            record.get('training_seed') != 101 or record.get('validation_seed') != 202 or
            record.get('initialization_seed') != 404 or record.get('sampler_seed') != 707 or
            record.get('steps_per_arm') != 2000 or record.get('batch_size') != 64 or
            record.get('train_samples') != 4096 or
            record.get('feature_illumination') is not False or
            record.get('center_gradient_multiplier') != (2.0 if center else 1.0) or
            record.get('validation_passes_original_thresholds') is not True or
            record.get('final_holdout_evaluated') is not False or
            record.get('artifact_sha256') != digest):
        raise ValueError('training provenance mismatch')


def evaluate(binary, mirror_path, center_path, mirror_report, center_report, *, invoke=None):
    mirror_path, center_path = Path(mirror_path), Path(center_path)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (mirror_path, center_path)]
    if hashes[0] == hashes[1]:
        raise ValueError('both training arms used identical model bytes')
    commit, backend = mirror_report.get('source_commit'), mirror_report.get('backend')
    if not isinstance(commit, str) or len(commit) != 40 or backend not in BACKENDS:
        raise ValueError('training provenance mismatch')
    check_training(mirror_report, center=False, commit=commit, backend=backend, digest=hashes[0])
    check_training(center_report, center=True, commit=commit, backend=backend, digest=hashes[1])
    samples = cohort()  # frozen and verified before any prediction is read
    predictions = [[], []]
    with tempfile.TemporaryDirectory(prefix='facts-yolo-blind-') as tmp:
        rgb = Path(tmp) / 'input.rgb'
        for sample in samples:
            rgb.write_bytes(sample['pixels'])
            for arm, model, output in zip(('mirror', 'center'),
                                          (mirror_path, center_path), predictions):
                if invoke is None:
                    proc = subprocess.run([str(binary), 'infer', str(model), str(rgb)],
                                          capture_output=True, text=True, check=True, timeout=15)
                    output.append(json.loads(proc.stdout))
                else:
                    output.append(invoke(sample, arm))
    if [hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (mirror_path, center_path)] != hashes:
        raise ValueError('trained model modified during inference')
    result = make_report(samples, *predictions, commit, backend, *hashes)
    check_report(result, commit, backend, *hashes)
    return result


def main():
    if len(sys.argv) != 6:
        raise ValueError('usage: blind_center_comparison.py <fy-cpu-evaluate> <mirror.fytd> '
                         '<center.fytd> <mirror-training.json> <center-training.json>')
    mr, cr = (json.loads(Path(p).read_text()) for p in sys.argv[4:])
    print(json.dumps(evaluate(*sys.argv[1:4], mr, cr), sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as err:
        print(f'FAIL: {err}', file=sys.stderr)
        sys.exit(1)
