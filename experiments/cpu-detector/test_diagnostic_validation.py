# SPDX-License-Identifier: Apache-2.0
"""Independent deterministic and negative diagnostic validation checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import diagnostic_validation as d


class RegisteredDiagnosticTests(unittest.TestCase):
    def test_legacy_generator_identity_is_exact(self):
        # Confirms the independent Python generator matches the Rust test data
        # rather than diagnosing a different synthetic distribution.
        self.assertEqual(d.fingerprint(d.samples(909, 128)), d.LEGACY_TEST_FINGERPRINT)

    def test_new_split_registered_before_model_evaluation(self):
        rows = list(d.samples(d.DIAGNOSTIC_SEED, d.DIAGNOSTIC_COUNT))
        self.assertEqual(len(rows), 256)
        self.assertEqual(d.fingerprint(rows), d.DIAGNOSTIC_FINGERPRINT)
        self.assertEqual(len({hash(row["pixels"]) for row in rows}), 256)
        self.assertTrue(any(row["class"] is None for row in rows))
        self.assertTrue(any(row["class"] is not None and row["area"] <= 256 for row in rows))
        self.assertTrue(any(row["class"] is not None and row["area"] > 256 for row in rows))

    def test_missing_and_wrong_class_penalized_without_excluding_samples(self):
        rows = [
            {"class": 0, "corners": (4, 4, 16, 16), "area": 144},
            {"class": 1, "corners": (8, 8, 28, 28), "area": 400},
            {"class": 1, "corners": (10, 10, 22, 22), "area": 144},
            {"class": None, "corners": (2, 2, 14, 14), "area": 144},
        ]
        predictions = [
            {"class": 0, "score": 0.7, "corners": [4, 4, 16, 16]},
            {"class": 0, "score": 0.8, "corners": [8, 8, 28, 28]},
            None,
            {"class": 1, "score": 0.9, "corners": [10, 10, 22, 22]},
        ]
        report = d.summarize(rows, predictions)
        self.assertEqual(report["small"]["positives"], 2)
        self.assertEqual(report["small"]["missing_detections"], 1)
        self.assertEqual(report["small"]["mean_iou_including_misses"], 0.5)
        self.assertEqual(report["small"]["recall_correct_class_iou50"], 0.5)
        self.assertEqual(report["larger"]["class_accuracy"], 0.0)
        self.assertEqual(report["larger"]["mean_iou_including_misses"], 1.0)
        self.assertEqual(report["false_positives"], 1)

    def test_rejects_untrusted_inference_output(self):
        rows = [
            {"class": 0, "corners": (4, 4, 16, 16), "area": 144},
            {"class": 1, "corners": (8, 8, 28, 28), "area": 400},
            {"class": None, "corners": (0, 0, 0, 0), "area": 0},
        ]
        for bad in [
            {"class": True, "score": 0.8, "corners": [4, 4, 16, 16]},
            {"class": 0, "score": 0.49, "corners": [4, 4, 16, 16]},
            {"class": 0, "score": float("nan"), "corners": [4, 4, 16, 16]},
            {"class": 0, "score": 0.8, "corners": [4, 4, 65, 16]},
            {"class": 0, "score": 0.8, "corners": [16, 4, 4, 16]},
            {"class": 0, "score": 0.8, "corners": [4, 4, 16, float("inf")]},
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                d.summarize(rows, [bad, None, None])
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            d.summarize(rows, [None, None])

    def test_diagnostic_rejects_legacy_or_mutated_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.fytd"
            model.write_bytes(b"frozen-test-model")
            metrics = root / "metrics.json"
            training = {"passed": True, "steps": 2000, "training_seed": 101,
                        "test_seed": 909, "test_fingerprint": d.LEGACY_TEST_FINGERPRINT,
                        "backend": "burn-flex", "source_commit": "verified-revision"}
            metrics.write_text(json.dumps(training))
            perfect = lambda row: ({"class": row["class"], "score": 0.9,
                                    "corners": list(row["corners"])}
                                   if row["class"] is not None else None)
            report = d.evaluate("unused", model, metrics, invoke=perfect)
            self.assertEqual(report["metrics"]["small"]["class_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["larger"]["mean_iou_including_misses"], 1.0)
            self.assertEqual(report["metrics"]["background_false_positive_rate"], 0.0)
            self.assertNotIn("predictions", report)
            self.assertEqual(report["localization_diagnostic"]["small"]["correct_class_detections"], 78)
            self.assertEqual(report["localization_diagnostic"]["larger"]["correct_class_detections"], 126)
            self.assertEqual(report["localization_diagnostic"]["small"]["low_iou_correct_class_cases"], [])
            self.assertEqual(report["localization_diagnostic"]["larger"]["mean_abs_size_error_px"], [0.0, 0.0])
            with patch.object(d, "DIAGNOSTIC_SEED", 909):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    d.evaluate("unused", model, metrics, invoke=perfect)
            training["test_seed"] = 303
            metrics.write_text(json.dumps(training))
            with self.assertRaisesRegex(ValueError, "provenance"):
                d.evaluate("unused", model, metrics, invoke=perfect)
            training["test_seed"] = 909
            metrics.write_text(json.dumps(training))
            calls = 0
            def changed_model(row):
                nonlocal calls
                calls += 1
                if calls == 2:
                    model.write_bytes(b"modified-model")
                return perfect(row)
            with self.assertRaisesRegex(ValueError, "changed"):
                d.evaluate("unused", model, metrics, invoke=changed_model)


if __name__ == "__main__":
    unittest.main()
