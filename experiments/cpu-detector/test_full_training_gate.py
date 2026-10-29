# SPDX-License-Identifier: Apache-2.0
"""RED-before-GREEN tests for the opt-in full-training comparison verifier."""
import copy
import unittest
from full_training_gate import validate_full_training_benchmark

SHA = "a" * 40
BACKEND = "burn-flex-0.22.0"


def fixture():
    row = {
        "training_ns": 100000000,
        "batch_preparation_ns": 3000000,
        "allocation_requests": 30000,
        "requested_bytes": 90000000,
        "model_sha256": "f" * 64,
        "sampler_next_draw": 42,
        "last_batch_loss_bits": 1000,
        "validation": {
            "positives": 102, "negatives": 26, "class_accuracy": 1.0,
            "recall_iou50": 1.0, "mean_iou": 0.76, "false_positive_rate": 0.0,
        },
    }
    runs = []
    for i in range(3):
        h = copy.deepcopy(row)
        r = copy.deepcopy(row)
        h["allocation_requests"] += 4000
        h["requested_bytes"] += 105000000
        runs.append({
            "first": "historical" if i % 2 == 0 else "reused",
            "historical": h, "reused": r,
        })
    return {
        "scope": "full TD1 Burn/Candle training (2000 optimizer steps) in one process",
        "source_commit": SHA, "backend": BACKEND, "build": "release",
        "training_seed": 101, "initialization_seed": 404, "sampler_seed": 707,
        "steps_per_arm": 2000, "batch_size": 64,
        "training_samples": 4096, "repetitions": 3,
        "os": "linux", "architecture": "x86_64", "rayon_threads": "1",
        "data_preparation_excluded": True,
        "evaluation_excluded_from_timing": True,
        "exact_weight_rng_loss_validation_parity": True, "runs": runs,
    }


class FullTrainingGateTests(unittest.TestCase):
    def test_rejects_unpinned_or_unsupported_benchmark_configuration(self):
        for field, invalid in (
            ("rayon_threads", ""),
            ("rayon_threads", "8"),
            ("architecture", "aarch64"),
            ("os", "unknown"),
            ("steps_per_arm", 2000.0),
            ("training_seed", True),
        ):
            report = fixture()
            report[field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                validate_full_training_benchmark(report, SHA, BACKEND)

    def test_rejects_impossible_aggregate_metrics_without_changing_thresholds(self):
        def set_all(report, **metrics):
            for pair in report["runs"]:
                for mode in ("historical", "reused"):
                    pair[mode]["validation"].update(metrics)

        plausible = fixture()
        set_all(plausible, class_accuracy=100 / 102, recall_iou50=99 / 102,
                false_positive_rate=1 / 26)
        self.assertTrue(validate_full_training_benchmark(plausible, SHA, BACKEND))

        for changes in (
            {"class_accuracy": 101 / 102, "recall_iou50": 1.0},
            {"class_accuracy": 0.975, "recall_iou50": 0.95},
            {"false_positive_rate": 0.04},
        ):
            report = fixture()
            set_all(report, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_full_training_benchmark(report, SHA, BACKEND)

    def test_accepts_valid_equal_models_without_assuming_speedup(self):
        report = fixture()
        report["runs"][0]["reused"]["training_ns"] = 120000000
        self.assertTrue(validate_full_training_benchmark(report, SHA, BACKEND))

    def test_rejects_mismatched_weights_despite_self_reported_parity(self):
        report = fixture()
        report["runs"][2]["reused"]["model_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_full_training_benchmark(report, SHA, BACKEND)

    def test_rejects_mismatched_validation_or_rng(self):
        for field, value in [("sampler_next_draw", 3), ("last_batch_loss_bits", 2)]:
            report = fixture()
            report["runs"][1]["reused"][field] = value
            with self.assertRaises(ValueError):
                validate_full_training_benchmark(report, SHA, BACKEND)
        report = fixture()
        report["runs"][2]["historical"]["validation"]["mean_iou"] = 0.5
        with self.assertRaises(ValueError):
            validate_full_training_benchmark(report, SHA, BACKEND)

    def test_rejects_wrong_provenance_and_nonfinite_counters(self):
        report = fixture()
        report["source_commit"] = "b" * 40
        with self.assertRaises(ValueError):
            validate_full_training_benchmark(report, SHA, BACKEND)
        report = fixture()
        report["runs"][0]["reused"]["training_ns"] = float("inf")
        with self.assertRaises(ValueError):
            validate_full_training_benchmark(report, SHA, BACKEND)

    def test_rejects_bad_order_or_count_and_impossible_prep(self):
        report = fixture()
        report["runs"][1]["first"] = "historical"
        with self.assertRaises(ValueError):
            validate_full_training_benchmark(report, SHA, BACKEND)
        report = fixture()
        report["runs"][0]["historical"]["batch_preparation_ns"] = 110000000
        with self.assertRaises(ValueError):
            validate_full_training_benchmark(report, SHA, BACKEND)
        report = fixture()
        report["runs"].pop()
        with self.assertRaises(ValueError):
            validate_full_training_benchmark(report, SHA, BACKEND)


if __name__ == "__main__":
    unittest.main()
