# SPDX-License-Identifier: Apache-2.0
"""Frozen background-gradient comparison: independent data and negative controls."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import blind_background_comparison as bg


def oracle(sample):
    return None if sample['class'] is None else {
        'class': sample['class'], 'score': 0.99,
        'corners': list(sample['corners'])}


class BackgroundBlindTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = bg.cohort()

    def report(self, mirror=None, background=None):
        base = [oracle(item) for item in self.samples]
        return bg.make_report(
            self.samples, base if mirror is None else mirror,
            base if background is None else background,
            'a' * 40, 'burn-flex-0.22.0', '1' * 64, '2' * 64)

    def test_frozen_generator_and_pixel_label_oracle(self):
        self.assertEqual(len(self.samples), 120)
        self.assertEqual(bg.fingerprint(self.samples), bg.FINGERPRINT)
        self.assertEqual(len({hashlib.sha256(x['pixels']).digest()
                              for x in self.samples}), 120)
        for side in bg.SIDES:
            for cls in (0, 1):
                matches = [x for x in self.samples if x['side'] == side and x['class'] == cls]
                self.assertEqual(len(matches), 12)
        background = [r for r in self.samples if r['class'] is None]
        self.assertEqual(len(background), 24)
        for row in self.samples:
            active = []
            data = row['pixels']
            for offset in range(0, len(data), 3):
                red, green, blue = data[offset:offset + 3]
                if max(red, blue) >= 150:
                    self.assertIsNotNone(row['class'])
                    self.assertEqual(red > blue, row['class'] == 0)
                    active.append(((offset // 3) % 64, (offset // 3) // 64))
                else:
                    self.assertLess(max(red, green, blue), 150)
            if row['class'] is None:
                self.assertEqual(active, [])
            else:
                self.assertEqual((min(x for x, y in active), min(y for x, y in active),
                                  max(x for x, y in active) + 1,
                                  max(y for x, y in active) + 1), row['corners'])

    def test_aggregates_replay_from_all_raw_predictions(self):
        report = self.report()
        self.assertTrue(bg.check_report(report, 'a' * 40, 'burn-flex-0.22.0',
                                        '1' * 64, '2' * 64))
        self.assertEqual(report['comparison']['mirror_hits'], 96)
        self.assertEqual(report['comparison']['background_hits'], 96)
        self.assertEqual(report['comparison']['background_fp_reduction'], 0)
        changes = (
            lambda r: r.update(generator='old-cohort'),
            lambda r: r.update(fingerprint='0' * 64),
            lambda r: r.update(final_holdout_evaluated=True),
            lambda r: r['background'].update(background_class_gradient_multiplier=1),
            lambda r: r['mirror']['predictions'].pop(),
            lambda r: r['background']['predictions'][0].update(index=1),
            lambda r: r['comparison'].update(background_fp_reduction=24),
            lambda r: r['background']['groups']['background'].update(false_positives=20),
            lambda r: r['background']['predictions'][0].update(prediction=None),
        )
        for change in changes:
            with self.subTest(change=change):
                damaged = copy.deepcopy(report)
                change(damaged)
                with self.assertRaises(ValueError):
                    bg.check_report(damaged, 'a' * 40, 'burn-flex-0.22.0',
                                    '1' * 64, '2' * 64)

    def test_false_positives_and_localization_tradeoff_are_not_hidden(self):
        mirror = [oracle(item) for item in self.samples]
        background = copy.deepcopy(mirror)
        for i in range(96, 108):
            mirror[i] = {'class': 0, 'score': .6, 'corners': [10, 10, 30, 30]}
        background[0] = None
        report = self.report(mirror, background)
        self.assertEqual(report['comparison']['mirror_background_fp'], 12)
        self.assertEqual(report['comparison']['background_background_fp'], 0)
        self.assertEqual(report['comparison']['background_fp_reduction'], 12)
        self.assertEqual(report['comparison']['background_minus_mirror_hits'], -1)
        self.assertTrue(bg.check_report(report, 'a' * 40, 'burn-flex-0.22.0',
                                        '1' * 64, '2' * 64))
        invalid = copy.deepcopy(background)
        invalid[0] = {'class': 0, 'score': float('nan'), 'corners': [0, 0, 12, 12]}
        with self.assertRaises(ValueError):
            self.report(mirror, invalid)

    def test_pair_requires_real_distinct_unchanged_weight_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            mirror_path, background_path = folder/'mirror.fytd', folder/'background.fytd'
            mirror_path.write_bytes(b'validated mock mirror')
            background_path.write_bytes(b'validated mock background')
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (mirror_path, background_path)]
            def training(background, digest):
                return {'status': ('background-weighted-mirror-training-experiment-only'
                                   if background else 'mirrored-edge-training-experiment-only'),
                        'source_commit': 'a' * 40, 'backend': 'burn-flex-0.22.0',
                        'training_seed': 101, 'validation_seed': 202,
                        'initialization_seed': 404, 'sampler_seed': 707,
                        'steps_per_arm': 2000, 'batch_size': 64, 'train_samples': 4096,
                        'center_gradient_multiplier': 1.0,
                        'background_class_gradient_multiplier': 2.0 if background else 1.0,
                        'feature_illumination': False,
                        'validation_passes_original_thresholds': True,
                        'final_holdout_evaluated': False, 'artifact_sha256': digest}
            mirror = training(False, hashes[0])
            background = training(True, hashes[1])
            tested = bg.evaluate('not-invoked', mirror_path, background_path,
                                 mirror, background, invoke=lambda sample, arm: oracle(sample))
            self.assertTrue(bg.check_report(tested, 'a' * 40, 'burn-flex-0.22.0', *hashes))
            background['sampler_seed'] = 909
            with self.assertRaisesRegex(ValueError, 'training provenance'):
                bg.evaluate('not-invoked', mirror_path, background_path,
                            mirror, background, invoke=lambda sample, arm: oracle(sample))
            background['sampler_seed'] = 707
            def corrupt(sample, arm):
                if arm == 'background' and sample['index'] == 0:
                    background_path.write_bytes(b'changed')
                return oracle(sample)
            with self.assertRaisesRegex(ValueError, 'modified'):
                bg.evaluate('not-invoked', mirror_path, background_path,
                            mirror, background, invoke=corrupt)


if __name__ == '__main__':
    unittest.main()
