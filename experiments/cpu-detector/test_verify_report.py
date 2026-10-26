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

class F32RecallThresholdRegression(unittest.TestCase):
    # Pixel-edge corners below are exact f32 values. The independent f64 IoU
    # audit differs from the f32 Rust evaluator by only a few 1e-8, yet the
    # discontinuous IoU50 recall gate must agree with the evaluator exactly.
    def _report(self, target, prediction, reported_iou, recall):
        return {
            "test_samples": 3,
            "predictions": [
                {"index": 0, "expected_class": 0, "expected_corners": target,
                 "prediction": {"class": 0, "score": 0.9, "corners": prediction},
                 "iou": reported_iou},
                {"index": 1, "expected_class": 1,
                 "expected_corners": [16.0, 16.0, 48.0, 48.0],
                 "prediction": {"class": 1, "score": 0.9,
                                "corners": [16.0, 16.0, 48.0, 48.0]},
                 "iou": 1.0},
                {"index": 2, "expected_class": None,
                 "expected_corners": None, "prediction": None, "iou": 0.0},
            ],
            "test": {"positives": 2, "negatives": 1,
                     "class_accuracy": 1.0, "recall_iou50": recall,
                     "mean_iou": 0.75, "false_positive_rate": 0.0},
        }

    def test_f32_rounds_up_to_half(self):
        target = [0.8196417093276978, 8.558881759643555,
                  13.149816513061523, 27.078170776367188]
        prediction = [0.8196417093276978, 8.558881759643555,
                      6.984728813171387, 27.078170776367188]
        report = self._report(target, prediction, 0.5, 1.0)
        self.assertEqual(verify_test_predictions(report)["recall_iou50"], 1.0)

    def test_f32_rounds_below_half(self):
        target = [16.950172424316406, 1.0512468814849854,
                  29.976699829101562, 17.1527156829834]
        prediction = [16.950172424316406, 1.0512468814849854,
                      23.463436126708984, 17.1527156829834]
        report = self._report(target, prediction, 0.4999999701976776, 0.5)
        self.assertEqual(verify_test_predictions(report)["recall_iou50"], 0.5)


class IndependentHeldoutAcceptanceTests(unittest.TestCase):
    def setUp(self):
        VerifyTestPredictionsTests.setUp(self)
    # Report arithmetic can be internally consistent while an incorrectly
    # reported `passed=true` hides sub-threshold model quality.
    def test_forged_success_cannot_override_failed_quality_gates(self):
        from verify_report import enforce_heldout_thresholds
        self.report["passed"] = True
        actual = verify_test_predictions(self.report)
        self.assertEqual(actual["class_accuracy"], 0.5)
        with self.assertRaisesRegex(ValueError, "acceptance"):
            enforce_heldout_thresholds(actual, self.report["test"])

    def test_all_four_gates_fail_closed(self):
        from verify_report import enforce_heldout_thresholds
        passed = {"class_accuracy": 1.0, "recall_iou50": 1.0,
                  "mean_iou": 0.8, "false_positive_rate": 0.0}
        enforce_heldout_thresholds(passed, passed)
        for key, bad in (("class_accuracy", 0.9), ("recall_iou50", 0.8),
                         ("mean_iou", 0.6), ("false_positive_rate", 0.1)):
            fail = dict(passed, **{key: bad})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                enforce_heldout_thresholds(fail, fail)

    def test_f32_rounded_equality_is_not_false_rejection(self):
        from verify_report import enforce_heldout_thresholds
        import struct
        f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
        reported = {"class_accuracy": f32(0.95), "recall_iou50": f32(0.85),
                    "mean_iou": f32(0.70), "false_positive_rate": f32(0.05)}
        # Correct raw f64 aggregate is permitted to differ by floating-point
        # noise, but Rust's reported f32 must meet *exact* f32 gates.
        recomputed = {"class_accuracy": 0.95, "recall_iou50": 0.85,
                      "mean_iou": 0.70, "false_positive_rate": 0.05}
        enforce_heldout_thresholds(recomputed, reported)
    def test_pipeline_rejects_consistent_but_below_budget_success_flag(self):
        from verify import checked_training_heldout
        self.report.update({"passed": True, "source_commit": "same-rev"})
        with self.assertRaisesRegex(ValueError, "acceptance"):
            checked_training_heldout(self.report, "same-rev")


class RegisteredHoldoutContractTests(unittest.TestCase):
    """A self-consistent, easy subset must not masquerade as the 128-image holdout."""

    @staticmethod
    def passing_report():
        rows = []
        for index in range(128):
            positive = index < 102
            cls = index % 2 if positive else None
            box = [16.0, 16.0, 48.0, 48.0] if positive else None
            rows.append({
                "index": index,
                "expected_class": cls,
                "expected_corners": box,
                "prediction": ({"class": cls, "score": 0.9, "corners": box[:]}
                               if positive else None),
                "iou": 1.0 if positive else 0.0,
            })
        return {
            "source_commit": "same-rev",
            "passed": True,
            "test_seed": 909,
            "test_samples": 128,
            "test_fingerprint": "188366b67882eb3599300ac6b056c31b5b28e41733ab801fef1362bb05354c33",
            "predictions": rows,
            "test": {
                "positives": 102, "negatives": 26,
                "class_accuracy": 1.0, "recall_iou50": 1.0,
                "mean_iou": 1.0, "false_positive_rate": 0.0,
            },
        }

    def test_registered_full_holdout_is_accepted(self):
        from verify import checked_training_heldout
        report = self.passing_report()
        self.assertEqual(checked_training_heldout(report, "same-rev"),
                         report["test"])

    def test_below_decoder_score_cannot_pass_perfect_holdout(self):
        from verify import checked_training_heldout
        # Removing the decoder's >=0.5 gate must not let reported perfect
        # class/box metrics certify detections the real decoder suppresses.
        for score in (0.0, 0.49, 0.4999999701976776):
            with self.subTest(score=score):
                report = self.passing_report()
                for row in report["predictions"]:
                    if row["prediction"] is not None:
                        row["prediction"]["score"] = score
                with self.assertRaisesRegex(ValueError, "detection score"):
                    checked_training_heldout(report, "same-rev")

    def test_decoder_score_boundary_is_inclusive(self):
        from verify import checked_training_heldout
        for score in (0.5, 0.5000000596046448, 1.0):
            with self.subTest(score=score):
                report = self.passing_report()
                for row in report["predictions"]:
                    if row["prediction"] is not None:
                        row["prediction"]["score"] = score
                self.assertEqual(checked_training_heldout(report, "same-rev"),
                                 report["test"])

    def test_truncated_perfect_report_is_rejected(self):
        from verify import checked_training_heldout
        report = self.passing_report()
        report["predictions"] = [report["predictions"][0],
                                 report["predictions"][-1]]
        report["predictions"][1]["index"] = 1
        report["test_samples"] = 2
        report["test"]["positives"] = report["test"]["negatives"] = 1
        with self.assertRaisesRegex(ValueError, "unregistered TD1 held-out split"):
            checked_training_heldout(report, "same-rev")

    def test_wrong_final_seed_is_rejected(self):
        from verify import checked_training_heldout
        report = self.passing_report()
        report["test_seed"] = 303
        with self.assertRaisesRegex(ValueError, "unregistered TD1 held-out split"):
            checked_training_heldout(report, "same-rev")

    def test_wrong_final_fingerprint_is_rejected(self):
        from verify import checked_training_heldout
        report = self.passing_report()
        report["test_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "unregistered TD1 held-out split"):
            checked_training_heldout(report, "same-rev")
