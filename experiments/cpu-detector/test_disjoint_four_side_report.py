# SPDX-License-Identifier: Apache-2.0
"""Pre-result fail-closed checks for the disjoint-source TD1 accuracy comparison."""
import copy
import unittest

from verify import (DISJOINT_FINGERPRINT,
                    assess_disjoint_mirror_candidate,
                    checked_disjoint_four_side_report)


def side(hits=12):
    # All detections correct-class; misses at IoU 0.25 are still detected.
    return {
        "samples": 24,
        "correct_class_iou50": hits,
        "recall_correct_class_iou50": hits / 24,
        "mean_iou_including_misses": (hits + 0.25 * (24 - hits)) / 24,
        "missing_detections": 0,
        "wrong_class_detections": 0,
    }


def groups():
    return {**{name: side() for name in ("left", "top", "right", "bottom")},
            "background": {"samples": 48, "false_positives": 0}}


def report():
    return {
        "status": "controlled-training-experiment-only",
        "source_commit": "head-for-test",
        "backend": "burn-flex-0.22.0",
        "disjoint_generator": "TD1-four-side-disjoint-v1",
        "disjoint_seed": 20261123,
        "disjoint_candidate_budget": 8192,
        "disjoint_source_rows": 144,
        "disjoint_last_source_index": 6876,
        "disjoint_fingerprint": DISJOINT_FINGERPRINT,
        "final_holdout_evaluated": False,
        "baseline": {"validation_passes_original_thresholds": True},
        "validation_passes_original_thresholds": True,
        "baseline_disjoint_four_side": groups(),
        "edge_sampling_disjoint_four_side": groups(),
        "mirrored_disjoint_four_side": groups(),
    }


class DisjointSourceReportTests(unittest.TestCase):
    def test_registered_split_and_arithmetic_can_record_losses(self):
        original = report()
        mirrored = report()
        mirrored["mirrored_disjoint_four_side"]["right"] = side(0)
        self.assertTrue(checked_disjoint_four_side_report(
            original, "head-for-test",
            ("baseline_disjoint_four_side", "edge_sampling_disjoint_four_side")))
        self.assertTrue(checked_disjoint_four_side_report(
            mirrored, "head-for-test", ("mirrored_disjoint_four_side",)))
        self.assertFalse(assess_disjoint_mirror_candidate(
            original, mirrored)["qualified_for_further_synthetic_research"])

    def test_frozen_rule_only_qualifies_both_sides_without_regressions(self):
        original = report()
        mirrored = report()
        b = original["baseline_disjoint_four_side"]
        m = mirrored["mirrored_disjoint_four_side"]
        for name, n in (("left", 20), ("top", 19), ("right", 2), ("bottom", 3)):
            b[name] = side(n)
        for name, n in (("left", 19), ("top", 19), ("right", 19), ("bottom", 16)):
            m[name] = side(n)
        assessment = assess_disjoint_mirror_candidate(original, mirrored)
        self.assertTrue(assessment["qualified_for_further_synthetic_research"])
        self.assertEqual(assessment["correct_class_iou50_hit_delta_mirror_minus_baseline"],
                         {"left": -1, "top": 0, "right": 17, "bottom": 13})
        m["bottom"] = side(10)
        self.assertFalse(assess_disjoint_mirror_candidate(
            original, mirrored)["qualified_for_further_synthetic_research"])
        m["bottom"] = side(16)
        m["left"] = side(17)
        self.assertFalse(assess_disjoint_mirror_candidate(
            original, mirrored)["qualified_for_further_synthetic_research"])
        m["left"] = side(19)
        m["background"]["false_positives"] = 1
        self.assertFalse(assess_disjoint_mirror_candidate(
            original, mirrored)["qualified_for_further_synthetic_research"])
        m["background"]["false_positives"] = 0
        mirrored["validation_passes_original_thresholds"] = False
        self.assertFalse(assess_disjoint_mirror_candidate(
            original, mirrored)["qualified_for_further_synthetic_research"])

    def test_seed_hash_budget_and_final_holdout_mutations_rejected(self):
        for key, mutation in (
            ("source_commit", "wrong-head"),
            ("disjoint_generator", "TD1-four-side-paired-v1"),
            ("disjoint_seed", 909),
            ("disjoint_candidate_budget", 4096),
            ("disjoint_source_rows", 96),
            ("disjoint_last_source_index", 0),
            ("disjoint_fingerprint", "0" * 64),
            ("final_holdout_evaluated", True),
        ):
            with self.subTest(key=key):
                changed = report()
                changed[key] = mutation
                with self.assertRaisesRegex(ValueError, "disjoint four-side provenance"):
                    checked_disjoint_four_side_report(
                        changed, "head-for-test", ("mirrored_disjoint_four_side",))

    def test_corrupt_counts_recall_iou_and_groups_rejected(self):
        for section, key, value in (
            ("right", "samples", True),
            ("right", "correct_class_iou50", True),
            ("right", "correct_class_iou50", 25),
            ("right", "missing_detections", 25),
            ("right", "recall_correct_class_iou50", 0.99),
            ("right", "mean_iou_including_misses", float("nan")),
            ("right", "mean_iou_including_misses", 0.0),
            ("background", "samples", 24),
            ("background", "false_positives", True),
            ("background", "false_positives", 49),
        ):
            with self.subTest(section=section, key=key):
                changed = report()
                changed["mirrored_disjoint_four_side"][section][key] = value
                with self.assertRaises(ValueError):
                    checked_disjoint_four_side_report(
                        changed, "head-for-test", ("mirrored_disjoint_four_side",))
        changed = report()
        changed["mirrored_disjoint_four_side"].pop("top")
        with self.assertRaises(ValueError):
            checked_disjoint_four_side_report(
                changed, "head-for-test", ("mirrored_disjoint_four_side",))

    def test_cannot_compare_incompatible_backend_or_source(self):
        original = report()
        mirrored = copy.deepcopy(original)
        mirrored["backend"] = "candle-cpu-0.11.0"
        with self.assertRaisesRegex(ValueError, "incompatible provenance"):
            assess_disjoint_mirror_candidate(original, mirrored)
        mirrored["backend"] = original["backend"]
        mirrored["source_commit"] = "stale"
        with self.assertRaisesRegex(ValueError, "incompatible provenance"):
            assess_disjoint_mirror_candidate(original, mirrored)


if __name__ == "__main__":
    unittest.main()
