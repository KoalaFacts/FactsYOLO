#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Frozen independent-mosaic TD1 mirror-vs-background gradient experiment.

This cohort is defined by its code and SHA-256 before any model predictions.
It is synthetic, is never a training/selection signal, and does not load seed909.
Models and RGB bytes remain ephemeral; only raw predictions/metrics are reported.
"""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import blind_center_comparison as bc

GENERATOR = 'TD1-independent-mosaic-background-v1'
SEED = 2026110203
SIDES = ('left', 'top', 'right', 'bottom')
PER_CLASS_SIDE = 12
BACKGROUNDS = 24
COUNT = 2 * len(SIDES) * PER_CLASS_SIDE + BACKGROUNDS
# Frozen from the untrained generator on 2026-11-02, before any predictions.
FINGERPRINT = '9366f137ba573d6a63f34bce0649b033bdd4e0bf5f3fdb9cb8edfc6fdcbc8edf'
STATUS = 'synthetic-background-paired-diagnostic-only'


class Rng:
    def __init__(self, index):
        self.state = int.from_bytes(hashlib.sha256(
            f'{GENERATOR}:{SEED}:{index}'.encode('ascii')).digest()[:8], 'little') or 1

    def pick(self, n):
        if n <= 0:
            raise ValueError('invalid random range')
        self.state ^= (self.state << 13) & 0xffffffffffffffff
        self.state ^= self.state >> 7
        self.state ^= (self.state << 17) & 0xffffffffffffffff
        self.state &= 0xffffffffffffffff
        return self.state % n


def rows():
    """Independent mosaics plus neutral rectangular background distractors."""
    for index in range(COUNT):
        rng = Rng(index)
        positive = index < COUNT - BACKGROUNDS
        side = SIDES[index // (2 * PER_CLASS_SIDE)] if positive else 'background'
        cls = (index // PER_CLASS_SIDE) % 2 if positive else None
        pixels = bytearray(64 * 64 * 3)
        for y in range(64):
            for x in range(64):
                mosaic = 5 * ((x // 4 + 2 * (y // 6) + index) % 5)
                ramp = (3 * x + 2 * y + index) % 19
                base = 10 + mosaic + ramp + rng.pick(13)
                at = 3 * (y * 64 + x)
                pixels[at:at + 3] = bytes((base + rng.pick(7),
                                           base + rng.pick(7),
                                           base + rng.pick(7)))
        corners = None
        xywh = (0.0, 0.0, 0.0, 0.0)
        if positive:
            w, h = 11 + rng.pick(12), 11 + rng.pick(12)
            x = (rng.pick(4) if side == 'left' else
                 (64 - w - rng.pick(4) if side == 'right' else
                  7 + rng.pick(64 - w - 14)))
            y = (rng.pick(4) if side == 'top' else
                 (64 - h - rng.pick(4) if side == 'bottom' else
                  7 + rng.pick(64 - h - 14)))
            for py in range(y, y + h):
                for px in range(x, x + w):
                    at = 3 * (py * 64 + px)
                    channels = [rng.pick(45), rng.pick(45), rng.pick(45)]
                    channels[cls * 2] = 190 + ((px // 5 + py // 4) % 5) * 9 + rng.pick(15)
                    pixels[at:at + 3] = bytes(channels)
            corners = (x, y, x + w, y + h)
            xywh = ((x + w / 2) / 64, (y + h / 2) / 64, w / 64, h / 64)
        else:
            # Deliberately rectangular but chromatically neutral: a hard negative.
            w, h = 11 + rng.pick(14), 11 + rng.pick(14)
            x, y = 4 + rng.pick(64 - w - 8), 4 + rng.pick(64 - h - 8)
            for py in range(y, y + h):
                for px in range(x, x + w):
                    at = 3 * (py * 64 + px)
                    gray = 75 + rng.pick(38)
                    pixels[at:at + 3] = bytes((gray, gray + rng.pick(7), gray + rng.pick(7)))
        yield {'index': index, 'side': side, 'class': cls, 'pixels': bytes(pixels),
               'corners': corners, 'geometry': xywh}


def fingerprint(items):
    digest = hashlib.sha256(GENERATOR.encode('ascii'))
    for sample in items:
        digest.update(bytes([255 if sample['class'] is None else sample['class']]))
        digest.update(struct.pack('<4f', *sample['geometry']))
        digest.update(sample['pixels'])
    return digest.hexdigest()


def cohort():
    samples = list(rows())
    if len(samples) != COUNT or fingerprint(samples) != FINGERPRINT:
        raise ValueError('frozen independent background cohort fingerprint drift')
    digests = {hashlib.sha256(row['pixels']).digest() for row in samples}
    if len(digests) != COUNT:
        raise ValueError('duplicate independent background image')
    # The prior blind-center cohort is already observed; assert pixel-level disjointness.
    if digests.intersection(hashlib.sha256(row['pixels']).digest() for row in bc.cohort()):
        raise ValueError('observed blind-center pixel reused')
    for sample in samples:
        if len(sample['pixels']) != 64 * 64 * 3:
            raise ValueError('invalid independent RGB size')
        if sample['class'] is None:
            if sample['side'] != 'background' or sample['corners'] is not None:
                raise ValueError('invalid independent background label')
        else:
            a, b, c, d = sample['corners']
            if not (0 <= a < c <= 64 and 0 <= b < d <= 64):
                raise ValueError('invalid independent target box')
            margin = {'left': a, 'top': b, 'right': 64 - c, 'bottom': 64 - d}[sample['side']]
            if margin > 3:
                raise ValueError('independent target not at registered edge')
    return samples


def make_report(samples, mirror_preds, background_preds, commit, backend, mirror_sha, background_sha):
    original = bc.summarize(samples, mirror_preds)
    experimental = bc.summarize(samples, background_preds)
    def hits(groups):
        return sum(groups[f'{side}_class{cls}']['hits'] for side in SIDES for cls in (0, 1))
    def arm(sha, multiplier, groups, predictions):
        return {'model_sha256': sha, 'background_class_gradient_multiplier': multiplier,
                'groups': groups,
                'predictions': [{'index': row['index'], 'prediction': pred}
                                for row, pred in zip(samples, predictions)]}
    mirror_hits, background_hits = hits(original), hits(experimental)
    mirror_fp = original['background']['false_positives']
    background_fp = experimental['background']['false_positives']
    return {'status': STATUS, 'source_commit': commit, 'backend': backend,
            'generator': GENERATOR, 'seed': SEED, 'fingerprint': FINGERPRINT,
            'count': COUNT, 'final_holdout_evaluated': False,
            'training_steps': 2000, 'sampler_seed': 707, 'initialization_seed': 404,
            'mirror': arm(mirror_sha, 1.0, original, mirror_preds),
            'background': arm(background_sha, 2.0, experimental, background_preds),
            'comparison': {'mirror_hits': mirror_hits, 'background_hits': background_hits,
                           'background_minus_mirror_hits': background_hits - mirror_hits,
                           'mirror_background_fp': mirror_fp,
                           'background_background_fp': background_fp,
                           'background_fp_reduction': mirror_fp - background_fp}}


def check_report(report, commit, backend, mirror_sha, background_sha):
    if (not isinstance(report, dict) or report.get('status') != STATUS
            or report.get('source_commit') != commit or report.get('backend') != backend
            or report.get('generator') != GENERATOR or report.get('seed') != SEED
            or report.get('fingerprint') != FINGERPRINT or report.get('count') != COUNT
            or report.get('training_steps') != 2000 or report.get('sampler_seed') != 707
            or report.get('initialization_seed') != 404
            or report.get('final_holdout_evaluated') is not False):
        raise ValueError('independent background comparison provenance mismatch')
    samples = cohort()
    pair = []
    for name, digest, scale in [('mirror', mirror_sha, 1.0),
                                ('background', background_sha, 2.0)]:
        section = report.get(name)
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest) or
                not isinstance(section, dict) or section.get('model_sha256') != digest or
                section.get('background_class_gradient_multiplier') != scale or
                not isinstance(section.get('predictions'), list) or
                len(section['predictions']) != COUNT):
            raise ValueError('independent background arm provenance mismatch')
        predictions = []
        for idx, record in enumerate(section['predictions']):
            if (not isinstance(record, dict) or set(record) != {'index', 'prediction'}
                    or type(record['index']) is not int or record['index'] != idx):
                raise ValueError('missing or reordered background raw predictions')
            predictions.append(record['prediction'])
        if bc.summarize(samples, predictions) != section.get('groups'):
            raise ValueError('independent background aggregate mismatch')
        pair.append(predictions)
    if make_report(samples, *pair, commit, backend, mirror_sha, background_sha) != report:
        raise ValueError('independent background comparison not replayable')
    return True


def check_training(record, *, background, commit, backend, digest):
    status = ('background-weighted-mirror-training-experiment-only' if background
              else 'mirrored-edge-training-experiment-only')
    if (not isinstance(record, dict) or record.get('status') != status
            or record.get('source_commit') != commit or record.get('backend') != backend
            or record.get('training_seed') != 101 or record.get('validation_seed') != 202
            or record.get('initialization_seed') != 404 or record.get('sampler_seed') != 707
            or record.get('steps_per_arm') != 2000 or record.get('batch_size') != 64
            or record.get('train_samples') != 4096 or record.get('feature_illumination') is not False
            or record.get('center_gradient_multiplier') != 1.0
            or record.get('background_class_gradient_multiplier') != (2.0 if background else 1.0)
            or record.get('validation_passes_original_thresholds') is not True
            or record.get('final_holdout_evaluated') is not False
            or record.get('artifact_sha256') != digest):
        raise ValueError('independent background training provenance mismatch')


def evaluate(binary, mirror_path, background_path, mirror_report, background_report, *, invoke=None):
    paths = [Path(mirror_path), Path(background_path)]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if hashes[0] == hashes[1]:
        raise ValueError('mirror/background used identical model bytes')
    commit, backend = mirror_report.get('source_commit'), mirror_report.get('backend')
    if (not isinstance(commit, str) or len(commit) != 40 or
            any(c not in '0123456789abcdef' for c in commit) or backend not in bc.BACKENDS):
        raise ValueError('independent background training provenance mismatch')
    check_training(mirror_report, background=False, commit=commit, backend=backend, digest=hashes[0])
    check_training(background_report, background=True, commit=commit, backend=backend, digest=hashes[1])
    samples = cohort()
    predictions = [[], []]
    with tempfile.TemporaryDirectory(prefix='factsyolo-background-blind-') as temporary:
        rgb = Path(temporary) / 'sample.rgb'
        for sample in samples:
            rgb.write_bytes(sample['pixels'])
            for name, path, output in zip(('mirror', 'background'), paths, predictions):
                if invoke is None:
                    process = subprocess.run([str(binary), 'infer', str(path), str(rgb)],
                                             capture_output=True, text=True, timeout=15, check=True)
                    output.append(json.loads(process.stdout))
                else:
                    output.append(invoke(sample, name))
    if [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] != hashes:
        raise ValueError('trained background or mirror model modified during inference')
    result = make_report(samples, *predictions, commit, backend, *hashes)
    check_report(result, commit, backend, *hashes)
    return result


def main():
    if len(sys.argv) != 6:
        raise ValueError('usage: blind_background_comparison.py <fy-cpu-evaluate> <mirror.fytd> '
                         '<background.fytd> <mirror-training.json> <background-training.json>')
    mr, br = (json.loads(Path(p).read_text()) for p in sys.argv[4:])
    print(json.dumps(evaluate(*sys.argv[1:4], mr, br), sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
