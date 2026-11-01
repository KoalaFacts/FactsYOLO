# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the independent-source TD1 inference evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cross_generator_validation as cv
from diagnostic_validation import LEGACY_TEST_FINGERPRINT


def perfect(item):
    return None if item["class"] is None else {
        "class": item["class"], "score": 0.99,
        "corners": list(item["corners"])}


class CrossGeneratorTests(unittest.TestCase):
    def test_first_frozen_generator_passes_and_covers_both_classes_all_sides(self):
        rows = cv.cohort()
        self.assertEqual(len(rows), 120)
        self.assertEqual(cv.fingerprint(rows), cv.FINGERPRINT)
        for side in cv.SIDES:
            for cls in (0, 1):
                group = [r for r in rows if r["side"] == side and r["class"] == cls]
                self.assertEqual(len(group), 12)
                for row in group:
                    a, b, c, d = row["corners"]
                    self.assertTrue({"left": a <= 2, "right": 64-c <= 2,
                                     "top": b <= 2, "bottom": 64-d <= 2}[side])
                    self.assertEqual(len(row["pixels"]), 12288)
        self.assertEqual(sum(r["class"] is None for r in rows), 24)

    def test_correct_inference_summary_and_checked_report(self):
        rows = cv.cohort()
        groups = cv.summarize(rows, [perfect(x) for x in rows])
        report = {"status": "exploratory-cross-generator-only", "source_commit": "1"*40,
                  "backend": "burn-flex-0.22.0", "model_sha256": "a"*64,
                  "generator": cv.GENERATOR, "seed": cv.SEED, "count": cv.COUNT,
                  "fingerprint": cv.FINGERPRINT, "final_holdout_evaluated": False,
                  "groups": groups,
                  "predictions": [{"index": item["index"], "prediction": perfect(item)}
                                  for item in rows]}
        self.assertTrue(cv.check_report(report, "1"*40, "burn-flex-0.22.0"))
        for name in ("left_class0", "right_class1"):
            self.assertEqual(groups[name]["hits"], 12)
            self.assertEqual(groups[name]["mean_iou_including_misses"], 1.0)
        for target, value in (
            ("recall_iou50", 0.1), ("hits", 13),
            ("mean_iou_including_misses", 0.49), ("missing", 1),
        ):
            with self.subTest(target=target):
                bad = copy.deepcopy(report)
                bad["groups"]["left_class0"][target] = value
                with self.assertRaises(ValueError):
                    cv.check_report(bad, "1"*40, "burn-flex-0.22.0")
        for bad_change in (lambda r: r.update(source_commit="0"*40),
                           lambda r: r.update(final_holdout_evaluated=True),
                           lambda r: r["groups"].pop("background"),
                           lambda r: r["groups"]["background"].update(false_positives=25),
                           lambda r: r["predictions"][0].update(index=1),
                           lambda r: r["predictions"][0].update(prediction=None),
                           lambda r: r["predictions"].pop()):
            bad = copy.deepcopy(report)
            bad_change(bad)
            with self.assertRaises(ValueError):
                cv.check_report(bad, "1"*40, "burn-flex-0.22.0")

    def test_missing_wrong_class_and_incomplete_predictions(self):
        rows = cv.cohort()
        predictions = [perfect(x) for x in rows]
        predictions[0] = None
        predictions[1] = {**predictions[1], "class": 1}
        groups = cv.summarize(rows, predictions)
        self.assertEqual(groups["left_class0"]["missing"], 1)
        self.assertEqual(groups["left_class0"]["wrong_class"], 1)
        self.assertEqual(groups["left_class0"]["hits"], 10)
        with self.assertRaises(ValueError):
            cv.summarize(rows, predictions[:-1])
        predictions[2] = {"class": 0, "score": float("nan"), "corners": [1, 1, 2, 2]}
        with self.assertRaises(ValueError):
            cv.summarize(rows, predictions)

    def test_inference_requires_correct_training_provenance_and_immutable_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model = path / "model.fytd"
            model.write_bytes(b"fake-regression-model")
            metrics = path / "metrics.json"
            record = {"passed": True, "steps": 2000, "training_seed": 101,
                      "test_seed": 909, "test_fingerprint": LEGACY_TEST_FINGERPRINT,
                      "backend": "candle-cpu-0.11.0", "source_commit": "b"*40}
            metrics.write_text(json.dumps(record))
            report = cv.evaluate("not-invoked", model, metrics, invoke=perfect)
            self.assertTrue(cv.check_report(report, "b"*40, "candle-cpu-0.11.0"))
            record["test_seed"] = 808
            metrics.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "provenance"):
                cv.evaluate("not-invoked", model, metrics, invoke=perfect)
            record["test_seed"] = 909
            metrics.write_text(json.dumps(record))
            def corrupt(sample):
                if sample["index"] == 1:
                    model.write_bytes(b"changed")
                return perfect(sample)
            with self.assertRaisesRegex(ValueError, "modified"):
                cv.evaluate("not-invoked", model, metrics, invoke=corrupt)


if __name__ == "__main__":
    unittest.main()
