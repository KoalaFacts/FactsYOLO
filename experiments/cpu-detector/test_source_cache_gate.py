# SPDX-License-Identifier: Apache-2.0
"""Reject ungrounded TD1 source-cache allocation claims."""
import copy
import unittest
from source_cache_gate import validate_cache_report

SHA = "a" * 40
CAPACITY = 4096 * 64 * 64 * 3


def fixture():
    runs = []
    for index in range(3):
        retained = {
            "elapsed_ns": 1000000,
            "source_capacity_bytes": CAPACITY,
            "live_after_cache_bytes": 54000000,
            "peak_live_during_preparation_bytes": 55000000,
            "feature_label_sha256": "b" * 64,
        }
        released = {
            **retained,
            "live_after_cache_bytes": 54000000 - CAPACITY,
            "peak_live_during_preparation_bytes": 55000000,
        }
        runs.append({
            "first": "retained" if index % 2 == 0 else "released",
            "retained": retained,
            "released": released,
        })
    return {
        "schema": "fy-td1-source-cache-allocation-v1",
        "source_commit": SHA,
        "build": "release",
        "scope": "feature caching and source buffer residency only; excludes training",
        "allocator_metric": "System allocator live requested payload bytes, not RSS",
        "model_training_executed": False,
        "training_seed": 101,
        "training_samples": 4096,
        "features_per_sample": 192,
        "repetitions": 3,
        "runs": runs,
    }


class SourceCacheGateTests(unittest.TestCase):
    def test_paired_real_allocation_contract(self):
        self.assertTrue(validate_cache_report(fixture(), SHA))

    def test_rejects_missing_or_wrong_provenance(self):
        for key, replacement in [
            ("source_commit", "c" * 40),
            ("training_seed", 102),
            ("model_training_executed", True),
            ("allocator_metric", "peak resident memory"),
            ("scope", "complete Burn training"),
            ("build", "debug"),
        ]:
            with self.subTest(key=key):
                report = fixture()
                report[key] = replacement
                with self.assertRaises(ValueError):
                    validate_cache_report(report, SHA)

    def test_rejects_claimed_savings_without_allocator_evidence(self):
        for key, value in [
            ("source_capacity_bytes", CAPACITY - 1),
            ("live_after_cache_bytes", 54000000 - 1),
            ("peak_live_during_preparation_bytes", 0),
            ("elapsed_ns", 0),
            ("feature_label_sha256", "0" * 64),
        ]:
            with self.subTest(key=key):
                report = fixture()
                report["runs"][1]["retained"][key] = value
                with self.assertRaises(ValueError):
                    validate_cache_report(report, SHA)

    def test_rejects_mismatched_source_or_feature_identity(self):
        report = fixture()
        report["runs"][2]["released"]["feature_label_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_cache_report(report, SHA)

    def test_peak_during_feature_build_need_not_improve(self):
        report = fixture()
        for run in report["runs"]:
            run["released"]["peak_live_during_preparation_bytes"] = (
                run["retained"]["peak_live_during_preparation_bytes"]
            )
        self.assertTrue(validate_cache_report(report, SHA))

    def test_rejects_unbalanced_orders_and_partial_repetitions(self):
        report = fixture()
        report["runs"][1]["first"] = "retained"
        with self.assertRaises(ValueError):
            validate_cache_report(report, SHA)
        report = fixture()
        report["runs"].pop()
        with self.assertRaises(ValueError):
            validate_cache_report(report, SHA)


if __name__ == "__main__":
    unittest.main()
