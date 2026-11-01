# SPDX-License-Identifier: Apache-2.0
"""Fail-closed provenance controls for the experimental photometric TD1 arm."""
import copy
import unittest
from unittest.mock import patch

from verify import checked_mirrored_edge_report


def report(textured):
    return {
        "status": ("textured-mirror-training-experiment-only" if textured
                   else "mirrored-edge-training-experiment-only"),
        "feature_illumination": textured,
        "illumination_phases": [
            "identity", "checkerboard", "horizontal-gradient", "vertical-gradient"],
        "source_commit": "candidate-head",
        "backend": "burn-flex-0.22.0",
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
        "mirror_orientations": [
            "identity", "horizontal", "vertical", "both"],
        "development_seed": 20261030,
        "development_samples": 240,
        "development_fingerprint":
            "04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4",
        "prospective_seed": 20261101,
        "prospective_candidate_budget": 4096,
        "prospective_samples": 240,
        "prospective_fingerprint":
            "0239bac3429d484db479974c10e5a038cef8829762a7d5533f8cd915e3a6ede1",
        "final_holdout_evaluated": False,
        "training_seconds": 1.0,
        "validation_passes_original_thresholds": True,
        "validation": {
            "positives": 102, "negatives": 26, "class_accuracy": 0.99,
            "recall_iou50": 0.91, "mean_iou": 0.75,
            "false_positive_rate": 0.0},
        "development": {},
        "prospective": {},
    }


class TextureTrainingReportTests(unittest.TestCase):
    def check_metadata(self, record, textured):
        # The per-stratum arithmetic checker has its own negative controls.
        # Isolate the new report-identity contract here.
        with patch("verify.checked_prospective_edge_report", return_value=True):
            return checked_mirrored_edge_report(
                record, "candidate-head", textured=textured)

    def test_new_texture_and_unchanged_mirror_modes_are_distinct(self):
        self.assertTrue(self.check_metadata(report(False), False))
        self.assertTrue(self.check_metadata(report(True), True))
        with self.assertRaises(ValueError):
            self.check_metadata(report(False), True)
        with self.assertRaises(ValueError):
            self.check_metadata(report(True), False)

    def test_wrong_phase_schedule_and_holdout_claim_fail_closed(self):
        for field, replacement in (
            ("source_commit", "stale-head"),
            ("illumination_phases", ["identity"]),
            ("feature_illumination", False),
            ("status", "production-ready"),
            ("steps_per_arm", 1999),
            ("sampler_seed", 909),
            ("final_holdout_evaluated", True),
            ("validation_passes_original_thresholds", False),
        ):
            with self.subTest(field=field):
                altered = copy.deepcopy(report(True))
                altered[field] = replacement
                with self.assertRaises(ValueError):
                    self.check_metadata(altered, True)


if __name__ == "__main__":
    unittest.main()
