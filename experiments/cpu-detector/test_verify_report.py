# SPDX-License-Identifier: Apache-2.0
"""Independent held-out metric report checks; only Python stdlib is required."""
import copy
import unittest

from verify_report import verify_test_predictions


class VerifyTestPredictionsTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "test_samples": 3,
            "predictions": [
                {"index": 0, "expected_class": 0,
                 "expected_corners": [16.0, 16.0, 48.0, 48.0],
                 "prediction": {"class": 0, "score": 0.9,
                                "corners": [16.0, 16.0, 48.0, 48.0]},
                 "iou": 1.0},
                {"index": 1, "expected_class": 1,
                 "expected_corners": [20.0, 20.0, 40.0, 40.0],
                 "prediction": None, "iou": 0.0},
                {"index": 2, "expected_class": None,
                 "expected_corners": None,
                 "prediction": {"class": 1, "score": 0.7,
                                "corners": [10.0, 10.0, 30.0, 30.0]},
                 "iou": 0.0},
            ],
            "test": {"positives": 2, "negatives": 1,
                     "class_accuracy": 0.5, "recall_iou50": 0.5,
                     "mean_iou": 0.5, "false_positive_rate": 1.0},
        }

    def test_recomputes_full_metrics(self):
        actual = verify_test_predictions(self.report)
        self.assertEqual(actual, self.report["test"])

    def test_rejects_falsified_metric(self):
        self.report["test"]["mean_iou"] = 0.75
        with self.assertRaises(ValueError):
            verify_test_predictions(self.report)

    def test_rejects_malformed_predicted_box(self):
        self.report["predictions"][0]["prediction"]["corners"][0] = 90.0
        with self.assertRaises(ValueError):
            verify_test_predictions(self.report)

    def test_rejects_wrong_per_prediction_iou(self):
        self.report["predictions"][0]["iou"] = 0.0
        with self.assertRaises(ValueError):
            verify_test_predictions(self.report)

    def test_rejects_missing_or_duplicate_index(self):
        self.report["predictions"][1]["index"] = 0
        with self.assertRaises(ValueError):
            verify_test_predictions(self.report)

    def test_rejects_invalid_label(self):
        self.report["predictions"][0]["expected_class"] = 2
        with self.assertRaises(ValueError):
            verify_test_predictions(self.report)

    def test_rejects_nan_score(self):
        self.report["predictions"][0]["prediction"]["score"] = float("nan")
        with self.assertRaises(ValueError):
            verify_test_predictions(self.report)

    def test_rejects_background_with_expected_box(self):
        self.report["predictions"][2]["expected_corners"] = [0.0, 0.0, 1.0, 1.0]
        with self.assertRaises(ValueError):
            verify_test_predictions(self.report)

    def test_rejects_unreported_extra_row(self):
        self.report["predictions"].append(copy.deepcopy(self.report["predictions"][1]))
        with self.assertRaises(ValueError):
            verify_test_predictions(self.report)


if __name__ == "__main__":
    unittest.main()
