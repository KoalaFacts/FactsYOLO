# SPDX-License-Identifier: Apache-2.0
"""Negative controls for pre-registered prospective TD1 edge evaluation."""
import unittest
import hashlib
import copy

import diagnostic_validation as d
import edge_validation as e
from verify import (checked_prospective_edge_report, checked_mirrored_edge_report,
                    PROSPECTIVE_EDGE_FINGERPRINT)


def sample_group(correct=46):
    return {
        "samples": 48,
        "correct_class_iou50": correct,
        "recall_correct_class_iou50": correct / 48,
        "missing_detections": 0,
        "wrong_class_detections": 0,
        "mean_iou_including_misses": 0.7,
    }


def report():
    def groups(edge):
        return {
            "small_edge": sample_group(edge),
            "small_interior": sample_group(47),
            "larger_edge": sample_group(47),
            "larger_interior": sample_group(48),
            "background": {"samples": 48, "false_positives": 0},
        }
    return {
        "source_commit": "tested-commit",
        "prospective_seed": 20261101,
        "prospective_candidate_budget": 4096,
        "prospective_samples": 240,
        "prospective_fingerprint": PROSPECTIVE_EDGE_FINGERPRINT,
        "final_holdout_evaluated": False,
        "baseline_prospective": groups(42),
        "edge_sampling_prospective": groups(44),
        "prospective_edge_minus_baseline_small_edge_iou50_recall": 2 / 48,
        "prospective_edge_minus_baseline_small_interior_iou50_recall": 0.0,
    }


class ProspectiveEdgeReportTests(unittest.TestCase):
    def test_prospective_generator_frozen_before_model_predictions(self):
        counts = dict.fromkeys(e.GROUPS, 0)
        selected = []
        for row in d.samples(20261101, 4096):
            name = e.stratum(row)
            if counts[name] < 48:
                counts[name] += 1
                selected.append(row)
            if all(n == 48 for n in counts.values()):
                break
        self.assertEqual(counts, dict.fromkeys(e.GROUPS, 48))
        self.assertEqual(d.fingerprint(selected), PROSPECTIVE_EDGE_FINGERPRINT)
        image_hashes = {hashlib.sha256(row["pixels"]).digest() for row in selected}
        self.assertEqual(len(image_hashes), 240)
        for seed, count in ((101, 4096), (202, 128), (909, 128),
                            (20261028, 256), (20261030, 4096)):
            with self.subTest(seed=seed):
                self.assertTrue(image_hashes.isdisjoint(
                    hashlib.sha256(row["pixels"]).digest()
                    for row in d.samples(seed, count)))

    def test_accepts_frozen_split_with_consistent_counts(self):
        self.assertTrue(checked_prospective_edge_report(report(), "tested-commit"))

    def test_negative_accuracy_result_is_still_valid_evidence(self):
        data = report()
        data["edge_sampling_prospective"]["small_edge"] = sample_group(40)
        data["prospective_edge_minus_baseline_small_edge_iou50_recall"] = -2 / 48
        self.assertTrue(checked_prospective_edge_report(data, "tested-commit"))

    def test_rejects_fingerprint_source_and_holdout_drift(self):
        for key, value in (
            ("prospective_fingerprint", "0" * 64),
            ("source_commit", "stale-commit"),
            ("prospective_seed", 909),
            ("final_holdout_evaluated", True),
        ):
            with self.subTest(key=key):
                data = report()
                data[key] = value
                with self.assertRaisesRegex(ValueError, "provenance"):
                    checked_prospective_edge_report(data, "tested-commit")

    def test_rejects_wrong_counts_and_forged_recall(self):
        for key, value in (
            ("samples", 47),
            ("correct_class_iou50", 49),
            ("recall_correct_class_iou50", 1.0),
            ("missing_detections", 48),
        ):
            with self.subTest(key=key):
                data = report()
                data["baseline_prospective"]["small_edge"][key] = value
                with self.assertRaises(ValueError):
                    checked_prospective_edge_report(data, "tested-commit")

    def test_rejects_mean_iou_inconsistent_with_iou50_hits_and_detection_counts(self):
        for hits, missing, wrong, mean_iou in (
            (48, 0, 0, 0.49),   # 48 hits require mean IoU >= 0.5
            (0, 48, 0, 0.1),    # all missing detections must contribute zero
            (0, 0, 48, 0.1),    # all wrong-class detections must contribute zero
            (46, 0, 0, 0.99),  # the two IoU50 misses are strictly below 0.5
        ):
            with self.subTest(hits=hits, missing=missing, wrong=wrong, mean_iou=mean_iou):
                data = report()
                row = data["baseline_prospective"]["small_edge"]
                row["correct_class_iou50"] = hits
                row["recall_correct_class_iou50"] = hits / 48
                row["missing_detections"] = missing
                row["wrong_class_detections"] = wrong
                row["mean_iou_including_misses"] = mean_iou
                with self.assertRaisesRegex(ValueError, "prospective IoU"):
                    checked_prospective_edge_report(data, "tested-commit")

    def test_accepts_consistent_mean_iou_boundary(self):
        data = report()
        row = data["baseline_prospective"]["small_edge"]
        row["correct_class_iou50"] = 48
        row["recall_correct_class_iou50"] = 1.0
        row["mean_iou_including_misses"] = 0.5
        data["prospective_edge_minus_baseline_small_edge_iou50_recall"] = -4 / 48
        self.assertTrue(checked_prospective_edge_report(data, "tested-commit"))

    def test_rejects_nonfinite_metrics(self):
        for key in ("recall_correct_class_iou50", "mean_iou_including_misses"):
            data = report()
            data["baseline_prospective"]["small_edge"][key] = float("nan")
            with self.assertRaises(ValueError):
                checked_prospective_edge_report(data, "tested-commit")
        data = report()
        data["prospective_edge_minus_baseline_small_edge_iou50_recall"] = float("nan")
        with self.assertRaises(ValueError):
            checked_prospective_edge_report(data, "tested-commit")

    def test_rejects_missing_group_and_forged_paired_gain(self):
        data = report()
        del data["baseline_prospective"]["larger_interior"]
        with self.assertRaisesRegex(ValueError, "groups"):
            checked_prospective_edge_report(data, "tested-commit")
        data = report()
        data["prospective_edge_minus_baseline_small_edge_iou50_recall"] = 1.0
        with self.assertRaisesRegex(ValueError, "paired"):
            checked_prospective_edge_report(data, "tested-commit")


def mirrored_report():
    # The experimental third-arm output contains no released model or holdout.
    groups = copy.deepcopy(report()["edge_sampling_prospective"])
    return {
        "status": "mirrored-edge-training-experiment-only",
        "source_commit": "tested-commit",
        "backend": "burn",
        "model": "TD1-v1-192-64-7",
        "training_seed": 101,
        "validation_seed": 202,
        "initialization_seed": 404,
        "sampler_seed": 707,
        "steps_per_arm": 2000,
        "batch_size": 64,
        "train_samples": 4096,
        "edge_pool_count": 48,
        "edge_override_slots_per_batch": 4,
        "mirror_slots": [0, 16, 32, 48],
        "mirror_orientations": ["identity", "horizontal", "vertical", "both"],
        "development_seed": 20261030,
        "development_samples": 240,
        "development_fingerprint":
            "04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4",
        "prospective_seed": 20261101,
        "prospective_candidate_budget": 4096,
        "prospective_samples": 240,
        "prospective_fingerprint": PROSPECTIVE_EDGE_FINGERPRINT,
        "final_holdout_evaluated": False,
        "training_seconds": 1.0,
        "development": copy.deepcopy(groups),
        "prospective": copy.deepcopy(groups),
        "validation": {
            "positives": 102, "negatives": 26,
            "class_accuracy": 1.0, "recall_iou50": 1.0,
            "mean_iou": 0.78, "false_positive_rate": 0.0,
        },
        "validation_passes_original_thresholds": True,
    }


class MirroredEdgeReportTests(unittest.TestCase):
    def test_accepts_exact_third_arm_provenance_and_valid_outcome(self):
        self.assertTrue(checked_mirrored_edge_report(mirrored_report(), "tested-commit"))

    def test_rejects_mirror_slot_or_provenance_drift(self):
        for key, value in (
            ("source_commit", "stale"),
            ("mirror_slots", [0, 15, 32, 48]),
            ("mirror_orientations", ["identity", "horizontal", "both", "vertical"]),
            ("sampler_seed", 404),
            ("final_holdout_evaluated", True),
            ("training_seconds", float("nan")),
        ):
            with self.subTest(key=key):
                data = mirrored_report()
                data[key] = value
                with self.assertRaises(ValueError):
                    checked_mirrored_edge_report(data, "tested-commit")

    def test_rejects_impossible_mirrored_iou_and_validation_regression(self):
        data = mirrored_report()
        data["prospective"]["small_edge"]["mean_iou_including_misses"] = 0.0
        with self.assertRaisesRegex(ValueError, "prospective IoU"):
            checked_mirrored_edge_report(data, "tested-commit")
        data = mirrored_report()
        data["validation"]["recall_iou50"] = 0.84
        with self.assertRaisesRegex(ValueError, "validation"):
            checked_mirrored_edge_report(data, "tested-commit")
        data = mirrored_report()
        data["validation_passes_original_thresholds"] = False
        with self.assertRaisesRegex(ValueError, "validation"):
            checked_mirrored_edge_report(data, "tested-commit")


if __name__ == "__main__":
    unittest.main()
