# SPDX-License-Identifier: Apache-2.0
"""Regression: an IoU-only relative win must not hide missing positives."""
import unittest

import blind_neutral_comparison as comparison


class NeutralDetectionGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = comparison.cohort()

    def test_background_improvement_cannot_mask_positive_detection_collapse(self):
        # A correctly classified but grossly mislocalized reference can have
        # zero IoU50 hits while still recognizing all 192 positive inputs.
        # One well-localized neutral prediction must not offset 191 misses.
        mirror = []
        neutral = []
        for case in self.cases:
            if case.cls is None:
                mirror.append({'class': 0, 'score': 0.95, 'corners': [0, 0, 1, 1]}
                              if case.index == 192 else None)
            else:
                mirror.append({'class': case.cls, 'score': 0.95,
                               'corners': [0, 0, 1, 1]})
            neutral.append(None)
        target = self.cases[0]
        neutral[0] = {'class': target.cls, 'score': 0.95,
                      'corners': list(target.box)}
        report = comparison.make_report(self.cases, mirror, neutral, 'a' * 40,
                                        'burn-flex-0.22.0', 'b' * 64, 'c' * 64)
        self.assertEqual(report['mirror']['summary']['positive_hits'], 0)
        self.assertEqual(report['neutral']['summary']['positive_hits'], 1)
        self.assertEqual(report['mirror']['summary']['background_false_positives'], 1)
        self.assertEqual(report['neutral']['summary']['background_false_positives'], 0)
        self.assertEqual(sum(s['missing'] for s in report['mirror']['summary']['sides'].values()), 0)
        self.assertEqual(sum(s['missing'] for s in report['neutral']['summary']['sides'].values()), 191)
        # Fails on c9e83aee: the original pass expression incorrectly returns True.
        self.assertFalse(report['comparison']['passes_this_backend_as_one_required_component_only'])
        self.assertTrue(comparison.check_report(report, 'a' * 40,
                                               'burn-flex-0.22.0', 'b' * 64, 'c' * 64))


if __name__ == '__main__':
    unittest.main()
