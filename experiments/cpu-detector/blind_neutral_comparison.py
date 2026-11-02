#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prediction-blind paired cross-generator evaluation for opt-in TD1 neutral negatives.

The cohort was frozen independently before any model prediction. This is a
synthetic research diagnostic, never a production/holdout acceptance gate.
Public CI artifacts contain per-image predictions and digests, not RGB/weights.
"""
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import blind_center_comparison as existing
import factsyolo_td1_crossgen_blind_v1_20261102 as frozen

SOURCE_SHA256 = '1e37e6c910504cbbee55c124cfa2ff7cae11c8618211e790e1de8c76f6ff79c2'
COHORT_SHA256 = 'accace0d0731b1262ef258b4ec62fcb6efa3911a1faae5c5bdbeb3aaa990e8df'
STATUS = 'synthetic-prediction-blind-neutral-comparison-only'
PHASES = ['identity', 'uniform-neutral', 'checkerboard-neutral', 'rectangle-neutral']
BACKENDS = {'burn-flex-0.22.0', 'candle-cpu-0.11.0'}
PAIRED_FIELDS = ('backend', 'source_commit', 'training_seed', 'validation_seed',
                 'initialization_seed', 'sampler_seed', 'steps_per_arm', 'batch_size',
                 'train_samples', 'mirror_slots', 'edge_override_slots_per_batch',
                 'development_fingerprint', 'prospective_fingerprint',
                 'four_side_fingerprint', 'disjoint_fingerprint')


def cohort():
    if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('preregistered generator source bytes changed')
    return _verified_cases()


@lru_cache(maxsize=1)
def _verified_cases():
    cases = list(frozen.samples())
    frozen.validate(cases)
    proof = frozen.manifest()
    if proof['cohort_sha256'] != COHORT_SHA256 or proof['model_predictions_observed'] is not False:
        raise ValueError('preregistered cohort identity changed')
    if [hashlib.sha256(c.pixels).hexdigest() for c in cases] != proof['image_sha256']:
        raise ValueError('preregistered per-image bytes changed')
    return tuple(cases)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_training(mirror, neutral, mirror_model, neutral_model):
    if not isinstance(mirror, dict) or not isinstance(neutral, dict):
        raise ValueError('missing training report')
    if any(k not in mirror or k not in neutral for k in PAIRED_FIELDS):
        raise ValueError('training-arm report fields missing')
    if any(mirror[k] != neutral[k] for k in PAIRED_FIELDS):
        raise ValueError('training-arm provenance differs')
    commit = mirror.get('source_commit')
    if not (isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit)
            and mirror.get('backend') in BACKENDS):
        raise ValueError('missing exact source or backend identity')
    if (mirror.get('status') != 'mirrored-edge-training-experiment-only'
            or neutral.get('status') != 'neutral-background-mirror-training-experiment-only'
            or mirror.get('neutral_negative_features') is not False
            or neutral.get('neutral_negative_features') is not True
            or neutral.get('neutral_background_phases') != PHASES):
        raise ValueError('wrong mirror/neutral treatment')
    for arm, path in ((mirror, mirror_model), (neutral, neutral_model)):
        if (arm.get('training_seed') != 101 or arm.get('validation_seed') != 202
                or arm.get('initialization_seed') != 404 or arm.get('sampler_seed') != 707
                or arm.get('train_samples') != 4096 or arm.get('batch_size') != 64
                or arm.get('steps_per_arm') != 2000
                or arm.get('mirror_slots') != [0, 16, 32, 48]
                or arm.get('edge_override_slots_per_batch') != 4
                or arm.get('center_gradient_multiplier') != 1.0
                or arm.get('background_class_gradient_multiplier') != 1.0
                or arm.get('feature_illumination') is not False
                or arm.get('validation_passes_original_thresholds') is not True
                or arm.get('final_holdout_evaluated') is not False
                or Path(path).stat().st_size != 51284
                or arm.get('artifact_sha256') != digest(path)):
            raise ValueError('training report/model does not meet unchanged provenance gates')
    if digest(mirror_model) == digest(neutral_model):
        raise ValueError('same model bytes used in both arms')
    return commit, mirror['backend']


def summarize(cases, predictions):
    if len(cases) != 256 or len(predictions) != len(cases):
        raise ValueError('missing paired inference results')
    sides = {side: {'samples': 0, 'hits': 0, 'missing': 0,
                    'wrong_class': 0, 'iou_sum': 0.0} for side in frozen.SIDES}
    negatives = {p: {'samples': 0, 'false_positives': 0} for p in frozen.PATTERNS}
    for case, raw in zip(cases, predictions):
        pred = existing.parse_prediction(raw)
        if case.cls is None:
            group = negatives[case.background]
            group['samples'] += 1
            group['false_positives'] += pred is not None
        else:
            group = sides[case.side]
            group['samples'] += 1
            if pred is None:
                group['missing'] += 1
            elif pred['class'] != case.cls:
                group['wrong_class'] += 1
            else:
                overlap = existing.iou(pred['corners'], case.box)
                group['hits'] += overlap >= 0.5
                group['iou_sum'] += overlap
    for side, group in sides.items():
        if group['samples'] != frozen.POS_PER_SIDE:
            raise ValueError(f'invalid side quota {side}')
        group['mean_iou_including_misses'] = group.pop('iou_sum') / group['samples']
        group['recall_iou50'] = group['hits'] / group['samples']
    if any(group['samples'] != frozen.NEG_PER_PATTERN for group in negatives.values()):
        raise ValueError('invalid negative-pattern quota')
    return {'sides': sides, 'negatives': negatives,
            'positive_hits': sum(g['hits'] for g in sides.values()),
            'background_false_positives': sum(g['false_positives'] for g in negatives.values())}


def make_report(cases, mirror_predictions, neutral_predictions, commit, backend, mirror_sha, neutral_sha):
    a, b = summarize(cases, mirror_predictions), summarize(cases, neutral_predictions)
    delta = {side: b['sides'][side]['hits'] - a['sides'][side]['hits'] for side in frozen.SIDES}
    fp_improved = b['background_false_positives'] < a['background_false_positives']
    nonregressed = b['positive_hits'] >= a['positive_hits']
    # IoU50-only nonregression is insufficient when the reference predicts
    # badly localized boxes: 0 -> 1 IoU50 hits can mask 191 new misses.
    # Preserve correct-class detection coverage separately from box quality.
    correct_class_a = sum(s['samples'] - s['missing'] - s['wrong_class']
                          for s in a['sides'].values())
    correct_class_b = sum(s['samples'] - s['missing'] - s['wrong_class']
                          for s in b['sides'].values())
    positive_detection_nonregressed = correct_class_b >= correct_class_a
    return {'status': STATUS, 'source_commit': commit, 'backend': backend,
            'generator': frozen.VERSION, 'generator_source_sha256': SOURCE_SHA256,
            'cohort_sha256': COHORT_SHA256, 'count': len(cases),
            'final_holdout_evaluated': False, 'research_only': True,
            'mirror': {'artifact_sha256': mirror_sha, 'summary': a,
                       'predictions': [{'index': c.index, 'prediction': p}
                                       for c, p in zip(cases, mirror_predictions)]},
            'neutral': {'artifact_sha256': neutral_sha, 'summary': b,
                        'predictions': [{'index': c.index, 'prediction': p}
                                        for c, p in zip(cases, neutral_predictions)]},
            'comparison': {'side_hit_delta_neutral_minus_mirror': delta,
                           'positive_hits_delta': b['positive_hits']-a['positive_hits'],
                           'positive_correct_class_detections_delta': correct_class_b - correct_class_a,
                           'background_fp_reduction': a['background_false_positives']-b['background_false_positives'],
                           'passes_this_backend_as_one_required_component_only': fp_improved and nonregressed
                           and positive_detection_nonregressed
                           and all(v >= -2 for v in delta.values())}}


def check_report(report, commit, backend, mirror_sha, neutral_sha):
    if (not isinstance(report, dict) or report.get('status') != STATUS
            or report.get('source_commit') != commit or report.get('backend') != backend
            or report.get('generator') != frozen.VERSION
            or report.get('generator_source_sha256') != SOURCE_SHA256
            or report.get('cohort_sha256') != COHORT_SHA256
            or report.get('count') != 256
            or report.get('final_holdout_evaluated') is not False
            or report.get('research_only') is not True):
        raise ValueError('blinded comparison identity or scope changed')
    cases = cohort()
    arms = []
    for name, sha in (('mirror', mirror_sha), ('neutral', neutral_sha)):
        arm = report.get(name)
        if not isinstance(arm, dict) or arm.get('artifact_sha256') != sha:
            raise ValueError('paired artifact SHA mismatch')
        rows = arm.get('predictions')
        if not isinstance(rows, list) or len(rows) != 256:
            raise ValueError('paired predictions missing')
        preds = []
        for i, item in enumerate(rows):
            if (not isinstance(item, dict) or set(item) != {'index', 'prediction'}
                    or type(item['index']) is not int or item['index'] != i):
                raise ValueError('paired prediction order mutated')
            preds.append(item['prediction'])
        arms.append(preds)
    if report != make_report(cases, *arms, commit, backend, mirror_sha, neutral_sha):
        raise ValueError('summary/prediction comparison cannot be independently replayed')
    return True


def evaluate(binary, mirror_model, neutral_model, mirror_report, neutral_report, *, invoke=None):
    commit, backend = check_training(mirror_report, neutral_report, mirror_model, neutral_model)
    sha_a, sha_b = digest(mirror_model), digest(neutral_model)
    cases = cohort()  # Verified before opening either trained model for inference.
    predictions = ([], [])
    with tempfile.TemporaryDirectory(prefix='factsyolo-blind-neutral-') as temp:
        rgb = Path(temp) / 'current.rgb'
        for case in cases:
            rgb.write_bytes(case.pixels)
            for name, artifact, out in zip(('mirror', 'neutral'), (mirror_model, neutral_model), predictions):
                if invoke is not None:
                    raw = invoke(case, name)
                else:
                    proc = subprocess.run([str(binary), 'infer', str(artifact), str(rgb)],
                                          capture_output=True, text=True, check=True, timeout=15)
                    raw = json.loads(proc.stdout)
                existing.parse_prediction(raw)
                out.append(raw)
    if digest(mirror_model) != sha_a or digest(neutral_model) != sha_b:
        raise ValueError('trained model bytes mutated during inference')
    result = make_report(cases, *predictions, commit, backend, sha_a, sha_b)
    check_report(result, commit, backend, sha_a, sha_b)
    return result


def main():
    if len(sys.argv) != 6:
        raise ValueError('usage: blind_neutral_comparison.py <fy-cpu-evaluate> <mirror.fytd> '
                         '<neutral.fytd> <mirror.json> <neutral.json>')
    mr, nr = (json.loads(Path(f).read_text()) for f in sys.argv[4:])
    print(json.dumps(evaluate(*sys.argv[1:4], mr, nr), sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as err:
        print(f'FAIL: {err}', file=sys.stderr)
        sys.exit(1)
