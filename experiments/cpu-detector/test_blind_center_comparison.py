# SPDX-License-Identifier: Apache-2.0
"""Negative controls for a frozen, paired, never-tuned TD1 cross-generator cohort."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import blind_center_comparison as bc


def perfect(item):
    return None if item['class'] is None else {
        'class': item['class'], 'score': 0.99, 'corners': list(item['corners'])}


class BlindCenterComparisonTests(unittest.TestCase):
    def make_report(self, mirror=None, center=None):
        items = bc.cohort()
        mirror = [perfect(row) for row in items] if mirror is None else mirror
        center = [perfect(row) for row in items] if center is None else center
        return bc.make_report(items, mirror, center, 'a'*40,
                              'burn-flex-0.22.0', '1'*64, '2'*64)

    def test_frozen_cohort_is_balanced_and_nonduplicated(self):
        rows = bc.cohort()
        self.assertEqual(len(rows), 120)
        self.assertEqual(bc.fingerprint(rows), bc.FINGERPRINT)
        self.assertEqual(len({hashlib.sha256(x['pixels']).digest() for x in rows}), 120)
        for side in bc.SIDES:
            for cls in (0, 1):
                selected = [r for r in rows if r['side'] == side and r['class'] == cls]
                self.assertEqual(len(selected), 12)
                for row in selected:
                    a, b, c, d = row['corners']
                    self.assertEqual(len(row['pixels']), 64*64*3)
                    self.assertTrue({'left': a<=3, 'top': b<=3,
                                     'right': 64-c<=3, 'bottom': 64-d<=3}[side])
        self.assertEqual(sum(x['class'] is None for x in rows), 24)

    def test_report_is_recomputed_from_per_sample_predictions(self):
        report = self.make_report()
        self.assertTrue(bc.check_report(report, 'a'*40, 'burn-flex-0.22.0',
                                        '1'*64, '2'*64))
        self.assertEqual(report['comparison']['mirror_hits'], 96)
        self.assertEqual(report['comparison']['center_hits'], 96)
        self.assertEqual(report['comparison']['center_minus_mirror_hits'], 0)
        self.assertEqual(report['mirror']['groups']['left_class0']['mean_center_error_px'], 0)
        for change in (
            lambda r: r.update(generator='different'),
            lambda r: r.update(fingerprint='0'*64),
            lambda r: r.update(final_holdout_evaluated=True),
            lambda r: r['center'].update(model_sha256='1'*64),
            lambda r: r['mirror']['predictions'].pop(),
            lambda r: r['mirror']['predictions'][0].update(index=1),
            lambda r: r['mirror']['predictions'][0].update(prediction=None),
            lambda r: r['comparison'].update(center_minus_mirror_hits=20),
            lambda r: r['center']['groups']['right_class1'].update(hits=9),
            lambda r: r['mirror']['groups']['background'].update(false_positives=24),
        ):
            with self.subTest(change=change):
                bad = copy.deepcopy(report)
                change(bad)
                with self.assertRaises(ValueError):
                    bc.check_report(bad, 'a'*40, 'burn-flex-0.22.0', '1'*64, '2'*64)

    def test_failing_predictions_and_localization_are_visible(self):
        items = bc.cohort()
        mirror = [perfect(row) for row in items]
        center = copy.deepcopy(mirror)
        center[0] = None
        center[12] = {**center[12], 'class': 0}
        center[24]['corners'] = [0, 0, 1, 1]
        report = self.make_report(mirror, center)
        self.assertEqual(report['comparison']['center_minus_mirror_hits'], -3)
        self.assertEqual(report['center']['groups']['left_class0']['missing'], 1)
        self.assertEqual(report['center']['groups']['left_class1']['wrong_class'], 1)
        self.assertEqual(report['center']['groups']['top_class0']['hits'], 11)
        self.assertTrue(bc.check_report(report, 'a'*40, 'burn-flex-0.22.0', '1'*64, '2'*64))
        corrupted = copy.deepcopy(center)
        corrupted[3]['score'] = float('nan')
        with self.assertRaises(ValueError):
            bc.make_report(items, mirror, corrupted, 'a'*40, 'burn-flex-0.22.0', '1'*64, '2'*64)

    def test_model_provenance_and_immutable_weight_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            mirror, center = directory/'mirror.fytd', directory/'center.fytd'
            mirror.write_bytes(b'mirror model')
            center.write_bytes(b'center model')
            a, b = (hashlib.sha256(p.read_bytes()).hexdigest() for p in (mirror, center))
            record = lambda h: {'status': h, 'source_commit': 'a'*40,
                                'backend': 'burn-flex-0.22.0',
                                'training_seed': 101, 'validation_seed': 202,
                                'initialization_seed': 404, 'sampler_seed': 707,
                                'steps_per_arm': 2000, 'batch_size': 64,
                                'train_samples': 4096,
                                'center_gradient_multiplier': 2.0 if h.startswith('center') else 1.0,
                                'feature_illumination': False,
                                'validation_passes_original_thresholds': True,
                                'final_holdout_evaluated': False}
            mr = record('mirrored-edge-training-experiment-only')
            cr = record('center-weighted-mirror-training-experiment-only')
            mr['artifact_sha256'] = a
            cr['artifact_sha256'] = b
            result = bc.evaluate('not-invoked', mirror, center, mr, cr,
                                 invoke=lambda row, arm: perfect(row))
            self.assertTrue(bc.check_report(result, 'a'*40,
                                            'burn-flex-0.22.0', a, b))
            cr['sampler_seed'] = 909
            with self.assertRaisesRegex(ValueError, 'training provenance'):
                bc.evaluate('not-invoked', mirror, center, mr, cr,
                            invoke=lambda row, arm: perfect(row))
            cr['sampler_seed'] = 707
            def corrupt(row, arm):
                if arm == 'center' and row['index'] == 0:
                    center.write_bytes(b'changed')
                return perfect(row)
            with self.assertRaisesRegex(ValueError, 'modified'):
                bc.evaluate('not-invoked', mirror, center, mr, cr, invoke=corrupt)


if __name__ == '__main__':
    unittest.main()
