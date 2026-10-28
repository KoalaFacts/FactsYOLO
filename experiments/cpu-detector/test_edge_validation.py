# SPDX-License-Identifier: Apache-2.0
"""Prospective geometric sampling and negative controls; no trained-model tuning."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import diagnostic_validation as d
import edge_validation as e


class EdgeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = e.selected_rows()

    def test_fixed_pre_prediction_sample_ids_and_no_historical_overlap(self):
        self.assertEqual(len(self.rows), 240)
        self.assertEqual(d.fingerprint(self.rows), e.SELECTED_FINGERPRINT)
        self.assertEqual({name: sum(e.stratum(r) == name for r in self.rows)
                          for name in e.GROUPS}, dict.fromkeys(e.GROUPS, 48))
        selected_hashes = {hashlib.sha256(r['pixels']).digest() for r in self.rows}
        self.assertEqual(len(selected_hashes), len(self.rows))
        for seed, count in ((101, 4096), (202, 128), (909, 128),
                            (d.DIAGNOSTIC_SEED, d.DIAGNOSTIC_COUNT)):
            self.assertTrue(selected_hashes.isdisjoint(
                hashlib.sha256(r['pixels']).digest() for r in d.samples(seed, count)))

    def test_edge_definition_and_small_boundary_are_fixed(self):
        def target(box):
            x1, y1, x2, y2 = box
            return {'class': 0, 'corners': box, 'area': (x2-x1)*(y2-y1)}
        self.assertEqual(e.stratum(target((2, 10, 14, 22))), 'small_edge')
        self.assertEqual(e.stratum(target((3, 10, 15, 22))), 'small_interior')
        self.assertEqual(e.stratum(target((30, 3, 46, 19))), 'small_interior')
        self.assertEqual(e.stratum(target((30, 2, 46, 18))), 'small_edge')
        self.assertEqual(e.stratum(target((3, 3, 20, 20))), 'larger_interior')
        self.assertEqual(e.stratum(target((42, 3, 62, 23))), 'larger_edge')
        with self.assertRaises(ValueError):
            e.stratum({**target((2, 10, 14, 22)), 'area': 0})

    def test_misses_wrong_class_and_background_do_not_inflate_iou(self):
        rows = [
            {'index': 1, 'class': 0, 'corners': (2, 4, 14, 16), 'area': 144},
            {'index': 2, 'class': 1, 'corners': (4, 4, 16, 16), 'area': 144},
            {'index': 3, 'class': 0, 'corners': (4, 4, 16, 16), 'area': 144},
            {'index': 4, 'class': 0, 'corners': (3, 3, 23, 23), 'area': 400},
            {'index': 5, 'class': None, 'corners': (2, 2, 14, 14), 'area': 144},
        ]
        predictions = [
            {'class': 0, 'score': .8, 'corners': [2, 4, 14, 16]},
            {'class': 0, 'score': .8, 'corners': [4, 4, 16, 16]},
            None,
            {'class': 0, 'score': .8, 'corners': [3, 3, 23, 23]},
            {'class': 1, 'score': .8, 'corners': [3, 3, 16, 16]},
        ]
        result = e.summarize(rows, predictions)
        self.assertEqual(result['small_edge']['correct_class_iou50'], 1)
        self.assertEqual(result['small_edge']['mean_correct_class_iou_including_misses'], 1)
        self.assertEqual(result['small_interior']['samples'], 2)
        self.assertEqual(result['small_interior']['missing_detections'], 1)
        self.assertEqual(result['small_interior']['wrong_class_detections'], 1)
        self.assertEqual(result['small_interior']['recall_correct_class_iou50'], 0)
        self.assertEqual(result['small_interior']['mean_correct_class_iou_including_misses'], 0)
        self.assertEqual(result['larger_interior']['recall_correct_class_iou50'], 1)
        self.assertEqual(result['background']['false_positives'], 1)

    def test_corrupted_predictions_fail_closed(self):
        row = {'index': 1, 'class': 0, 'corners': (2, 4, 14, 16), 'area': 144}
        for prediction in [
            {'class': 0, 'score': .8, 'corners': [2, 4, 2, 16]},
            {'class': True, 'score': .8, 'corners': [2, 4, 14, 16]},
            {'class': 0, 'score': .4, 'corners': [2, 4, 14, 16]},
            {'class': 0, 'score': .8, 'corners': [2, 4, float('nan'), 16]},
        ]:
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                e.summarize([row], [prediction])
        with self.assertRaises(ValueError):
            e.summarize([row], [])

    def test_evaluate_preserves_model_and_training_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / 'model.fytd'
            model.write_bytes(b'frozen-mock-trained-model')
            metrics = root / 'metrics.json'
            training = {'passed': True, 'steps': 2000, 'training_seed': 101,
                        'test_seed': 909, 'test_fingerprint': d.LEGACY_TEST_FINGERPRINT,
                        'backend': 'burn-flex', 'source_commit': 'tested-commit'}
            metrics.write_text(json.dumps(training))
            def perfect(row):
                if row['class'] is None:
                    return None
                return {'class': row['class'], 'score': .9,
                        'corners': list(row['corners'])}
            # Reuse frozen class fixture to avoid re-running 4,096-row sampling here.
            with patch.object(e, 'selected_rows', return_value=self.rows):
                report = e.evaluate('unused', model, metrics, invoke=perfect)
                self.assertEqual(report['selected_fingerprint'], e.SELECTED_FINGERPRINT)
                self.assertEqual(report['groups']['small_edge']['correct_class_iou50'], 48)
                self.assertEqual(report['groups']['small_interior']['correct_class_iou50'], 48)
                self.assertEqual(report['groups']['background']['false_positives'], 0)
                self.assertEqual(report['edge_minus_interior_iou50_recall']['small'], 0)
                self.assertEqual(report['edge_minus_interior_iou50_recall']['larger'], 0)
                self.assertNotIn('predictions', report)
                training['test_seed'] = 303
                metrics.write_text(json.dumps(training))
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    e.evaluate('unused', model, metrics, invoke=perfect)
                training['test_seed'] = 909
                metrics.write_text(json.dumps(training))
                calls = 0
                def mutated(row):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        model.write_bytes(b'changed-mock-model')
                    return perfect(row)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    e.evaluate('unused', model, metrics, invoke=mutated)

    def test_generator_or_selection_fingerprint_change_rejected(self):
        with patch.object(e, 'SELECTED_FINGERPRINT', '0' * 64), \
             patch.object(d, 'samples', return_value=iter(self.rows)):
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                e.selected_rows()


if __name__ == '__main__':
    unittest.main()
