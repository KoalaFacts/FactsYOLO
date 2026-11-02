# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the prediction-blind TD1 neutral-background comparison."""
import copy
import tempfile
import unittest
from pathlib import Path

import blind_neutral_comparison as cmp
import factsyolo_td1_crossgen_blind_v1_20261102 as frozen


class BlindNeutralComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = cmp.cohort()
        cls.commit = 'a' * 40
        cls.backend = 'burn-flex-0.22.0'
        cls.mirror = [None] * 256
        cls.neutral = [None] * 256
        cls.report = cmp.make_report(cls.cases, cls.mirror, cls.neutral,
                                     cls.commit, cls.backend, 'b' * 64, 'c' * 64)

    def test_pre_registered_source_pixels_and_corruption_controls(self):
        self.assertEqual(len(self.cases), 256)
        self.assertEqual(frozen.manifest()['cohort_sha256'], cmp.COHORT_SHA256)
        frozen.test_negative_controls()

    def test_unmodified_predictions_replay_even_when_both_models_fail(self):
        self.assertTrue(cmp.check_report(self.report, self.commit, self.backend,
                                         'b' * 64, 'c' * 64))
        self.assertEqual(self.report['mirror']['summary']['background_false_positives'], 0)
        self.assertFalse(self.report['comparison']['passes_this_backend_as_one_required_component_only'])

    def test_falsified_metrics_reordered_rows_and_bad_detection_fail_closed(self):
        for mutation in ('summary', 'index', 'score', 'fingerprint', 'rule'):
            with self.subTest(mutation=mutation):
                report = copy.deepcopy(self.report)
                if mutation == 'summary':
                    report['neutral']['summary']['background_false_positives'] = 12
                elif mutation == 'index':
                    report['mirror']['predictions'][1]['index'] = 0
                elif mutation == 'score':
                    report['neutral']['predictions'][192]['prediction'] = {
                        'class': 0, 'score': 0.4, 'corners': [0, 0, 10, 10]}
                elif mutation == 'fingerprint':
                    report['cohort_sha256'] = 'f' * 64
                else:
                    report['comparison']['passes_this_backend_as_one_required_component_only'] = True
                with self.assertRaises(ValueError):
                    cmp.check_report(report, self.commit, self.backend, 'b' * 64, 'c' * 64)

    def test_true_positive_and_false_positive_are_counted_from_raw_predictions(self):
        positives = [None] * 256
        positives[0] = {'class': self.cases[0].cls, 'score': 0.95,
                        'corners': list(self.cases[0].box)}
        positives[192] = {'class': 1, 'score': 0.7, 'corners': [1, 1, 11, 11]}
        report = cmp.make_report(self.cases, self.mirror, positives,
                                 self.commit, self.backend, 'b' * 64, 'c' * 64)
        self.assertEqual(report['neutral']['summary']['positive_hits'], 1)
        self.assertEqual(report['neutral']['summary']['background_false_positives'], 1)
        self.assertEqual(report['neutral']['summary']['sides']['left']['hits'], 1)
        self.assertEqual(report['neutral']['summary']['negatives']['uniform']['false_positives'], 1)
        self.assertTrue(cmp.check_report(report, self.commit, self.backend, 'b' * 64, 'c' * 64))

    def test_training_reports_require_distinct_weights_and_exact_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / 'a.fytd', Path(temp) / 'b.fytd'
            first.write_bytes(b'\x00' * 51284)
            second.write_bytes(b'\x01' * 51284)
            common = {'source_commit': self.commit, 'backend': self.backend,
                      'training_seed': 101, 'validation_seed': 202,
                      'initialization_seed': 404, 'sampler_seed': 707,
                      'steps_per_arm': 2000, 'batch_size': 64, 'train_samples': 4096,
                      'mirror_slots': [0,16,32,48], 'edge_override_slots_per_batch': 4,
                      'development_fingerprint': 'a'*64, 'prospective_fingerprint': 'b'*64,
                      'four_side_fingerprint': 'c'*64, 'disjoint_fingerprint': 'd'*64,
                      'feature_illumination': False, 'center_gradient_multiplier': 1.0,
                      'background_class_gradient_multiplier': 1.0,
                      'validation_passes_original_thresholds': True,
                      'final_holdout_evaluated': False}
            mirror = dict(common, status='mirrored-edge-training-experiment-only',
                          neutral_negative_features=False, artifact_sha256=cmp.digest(first))
            neutral = dict(common, status='neutral-background-mirror-training-experiment-only',
                           neutral_negative_features=True,
                           neutral_background_phases=cmp.PHASES,
                           artifact_sha256=cmp.digest(second))
            self.assertEqual(cmp.check_training(mirror, neutral, first, second),
                             (self.commit, self.backend))
            corrupt = dict(neutral, sampler_seed=909)
            with self.assertRaises(ValueError):
                cmp.check_training(mirror, corrupt, first, second)
            # Both arms can be consistently mislabeled; pairwise equality alone
            # must not allow a different edge-slot schedule or slot budget.
            for spoof in ({'mirror_slots': [1, 17, 33, 49]},
                          {'edge_override_slots_per_batch': 5}):
                with self.subTest(spoof=spoof):
                    with self.assertRaises(ValueError):
                        cmp.check_training(dict(mirror, **spoof),
                                           dict(neutral, **spoof), first, second)
            with self.assertRaises(ValueError):
                cmp.check_training(mirror, neutral, first, first)
            result = cmp.evaluate(None, first, second, mirror, neutral,
                                  invoke=lambda case, name: None)
            self.assertTrue(cmp.check_report(result, self.commit, self.backend,
                                             cmp.digest(first), cmp.digest(second)))


if __name__ == '__main__':
    unittest.main()
