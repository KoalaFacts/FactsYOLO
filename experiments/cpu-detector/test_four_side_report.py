# SPDX-License-Identifier: Apache-2.0
"""Fail-closed controls for a newly frozen four-sided TD1 paired diagnostic."""
import copy
import unittest

from verify import checked_four_side_report, FOUR_SIDE_FINGERPRINT


def group(hits=42):
    return {
        "samples": 48,
        "correct_class_iou50": hits,
        "recall_correct_class_iou50": hits / 48,
        "missing_detections": 0,
        "wrong_class_detections": 0,
        "mean_iou_including_misses": 0.75,
    }


def report():
    sides = {name: group() for name in ("left", "top", "right", "bottom")}
    return {
        "source_commit": "candidate-head",
        "four_side_generator": "TD1-four-side-paired-v1",
        "four_side_seed": 20261113,
        "four_side_candidate_budget": 4096,
        "four_side_source_rows": 48,
        "four_side_transformed_samples": 192,
        "four_side_fingerprint": FOUR_SIDE_FINGERPRINT,
        "final_holdout_evaluated": False,
        "baseline_four_side": copy.deepcopy(sides),
        "edge_sampling_four_side": copy.deepcopy(sides),
        "mirrored_four_side": copy.deepcopy(sides),
    }


class FrozenFourSideReportTests(unittest.TestCase):
    def test_accepts_consistent_paired_reports_without_selecting_winner(self):
        data = report()
        data["mirrored_four_side"]["bottom"] = group(1)
        self.assertTrue(checked_four_side_report(
            data, "candidate-head", ("mirrored_four_side",)))
        self.assertTrue(checked_four_side_report(
            data, "candidate-head", ("baseline_four_side", "edge_sampling_four_side")))

    def test_rejects_mutated_seed_hash_counts_and_final_holdout_access(self):
        for key, value in (
            ("source_commit", "stale"),
            ("four_side_generator", "TD1-synthetic-v1"),
            ("four_side_seed", 909),
            ("four_side_candidate_budget", 8192),
            ("four_side_source_rows", 47),
            ("four_side_transformed_samples", 48),
            ("four_side_fingerprint", "0" * 64),
            ("final_holdout_evaluated", True),
        ):
            with self.subTest(key=key):
                data = report()
                data[key] = value
                with self.assertRaisesRegex(ValueError, "four-side provenance"):
                    checked_four_side_report(data, "candidate-head", ("mirrored_four_side",))

    def test_rejects_missing_group_missing_arm_and_forged_recall(self):
        data = report()
        del data["mirrored_four_side"]["bottom"]
        with self.assertRaisesRegex(ValueError, "four-side groups"):
            checked_four_side_report(data, "candidate-head", ("mirrored_four_side",))
        data = report()
        del data["baseline_four_side"]
        with self.assertRaisesRegex(ValueError, "four-side groups"):
            checked_four_side_report(data, "candidate-head", ("baseline_four_side",))
        data = report()
        data["mirrored_four_side"]["right"]["recall_correct_class_iou50"] = 0.9
        with self.assertRaisesRegex(ValueError, "four-side recall"):
            checked_four_side_report(data, "candidate-head", ("mirrored_four_side",))

    def test_rejects_infeasible_iou_and_inconsistent_detections(self):
        for hits, missing, wrong, mean in (
            (48, 0, 0, 0.49),
            (0, 48, 0, 0.1),
            (0, 0, 48, 0.1),
            (46, 0, 0, 0.99),
        ):
            with self.subTest(hits=hits, missing=missing, wrong=wrong, mean=mean):
                data = report()
                value = data["mirrored_four_side"]["left"]
                value.update(correct_class_iou50=hits,
                             recall_correct_class_iou50=hits / 48,
                             missing_detections=missing,
                             wrong_class_detections=wrong,
                             mean_iou_including_misses=mean)
                with self.assertRaisesRegex(ValueError, "four-side IoU"):
                    checked_four_side_report(data, "candidate-head", ("mirrored_four_side",))

    def test_rejects_nonfinite_values_and_boolean_counts(self):
        data = report()
        data["mirrored_four_side"]["left"]["mean_iou_including_misses"] = float("nan")
        with self.assertRaises(ValueError):
            checked_four_side_report(data, "candidate-head", ("mirrored_four_side",))
        data = report()
        data["mirrored_four_side"]["left"]["correct_class_iou50"] = True
        with self.assertRaises(ValueError):
            checked_four_side_report(data, "candidate-head", ("mirrored_four_side",))


if __name__ == "__main__":
    unittest.main()
