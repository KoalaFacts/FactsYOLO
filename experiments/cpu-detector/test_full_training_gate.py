# SPDX-License-Identifier: Apache-2.0
"""RED-before-GREEN tests for the opt-in full-training comparison verifier."""
import copy
import json
import struct
import unittest
from full_training_gate import (validate_full_training_benchmark,
                                validate_full_training_diagnostics,
                                validate_interrupted_full_training_diagnostics,
                                CONTROLLED_FAILURE)

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
    for i in range(8):
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
        "training_samples": 4096, "repetitions": 8,
        "os": "linux", "architecture": "x86_64", "rayon_threads": "1",
        "data_preparation_excluded": True,
        "evaluation_excluded_from_timing": True,
        "exact_weight_rng_loss_validation_parity": True, "runs": runs,
    }


def diagnostics_fixture(report):
    records = []
    for pair, run in enumerate(report["runs"]):
        for arm in (("historical", "reused") if pair % 2 == 0 else ("reused", "historical")):
            shared = {"schema": "fy-full-training-progress-v1",
                      "source_commit": report["source_commit"],
                      "backend": report["backend"], "pair": pair, "arm": arm}
            records.append({**shared, "event": "started"})
            records.append({**shared, "event": "completed", "measurement": run[arm]})
    return "\n".join(json.dumps(row) for row in records) + "\n"


class FullTrainingGateTests(unittest.TestCase):
    def test_requires_eight_balanced_training_pairs(self):
        report = fixture()
        starts = [run["first"] for run in report["runs"]]
        self.assertEqual(len(starts), 8)
        self.assertEqual(starts.count("historical"), 4)
        self.assertEqual(starts.count("reused"), 4)
        self.assertTrue(validate_full_training_benchmark(report, SHA, BACKEND))
        report["runs"] = report["runs"][:-1]
        with self.assertRaises(ValueError):
            validate_full_training_benchmark(report, SHA, BACKEND)

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


    def test_rejects_cross_repeat_rng_loss_or_metric_drift(self):
        # A shared final model digest is not proof of identical RNG/loss/metrics.
        for kind in ("rng", "loss", "metrics"):
            report = fixture()
            for mode in ("historical", "reused"):
                arm = report["runs"][3][mode]
                if kind == "rng":
                    arm["sampler_next_draw"] += 1
                elif kind == "loss":
                    arm["last_batch_loss_bits"] += 1
                else:
                    arm["validation"]["mean_iou"] = 0.75
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_full_training_benchmark(report, SHA, BACKEND)

    def test_accepts_exact_rust_f32_iou_boundary_but_not_lower_or_nan(self):
        report = fixture()
        boundary_bits = struct.unpack("<I", struct.pack("<f", 0.70))[0]
        boundary = struct.unpack("<f", struct.pack("<I", boundary_bits))[0]
        below = struct.unpack("<f", struct.pack("<I", boundary_bits - 1))[0]
        self.assertLess(boundary, 0.70)  # Previous Python gate incorrectly rejected this.
        for value, accepted in ((boundary, True), (below, False), (float("nan"), False)):
            sample = copy.deepcopy(report)
            for pair in sample["runs"]:
                for arm in ("historical", "reused"):
                    pair[arm]["validation"]["mean_iou"] = value
            with self.subTest(value=value):
                if accepted:
                    self.assertTrue(validate_full_training_benchmark(sample, SHA, BACKEND))
                else:
                    with self.assertRaises(ValueError):
                        validate_full_training_benchmark(sample, SHA, BACKEND)

    def test_requires_complete_flushed_progress_matching_the_final_report(self):
        report = fixture()
        trace = diagnostics_fixture(report)
        self.assertTrue(validate_full_training_diagnostics(report, trace))
        lines = trace.splitlines()
        self.assertEqual(len(lines), 32)
        mutated = json.loads(lines[15])
        mutated["measurement"]["training_ns"] += 1
        changed_lines = lines.copy()
        changed_lines[15] = json.dumps(mutated)
        failure = json.loads(lines[-1])
        failure["event"] = "failed"
        failure.pop("measurement")
        failure["error"] = "training stopped"
        for invalid in ("\n".join(lines[:-1]), "\n".join(changed_lines),
                        "\n".join(lines[:-1] + [json.dumps(failure)]),
                        trace + json.dumps({
                            "schema": "fy-full-training-progress-v1",
                            "event": "process_failed", "error": "stopped"}) + "\n",
                        trace + "not-json\n"):
            with self.subTest(invalid=invalid[-120:]), self.assertRaises(ValueError):
                validate_full_training_diagnostics(report, invalid)


    def test_interrupted_rust_arm_must_preserve_completed_measurement(self):
        report = fixture()
        common = {"schema": "fy-full-training-progress-v1",
                  "source_commit": SHA, "backend": BACKEND, "pair": 0}
        events = [
            {**common, "arm": "historical", "event": "started"},
            {**common, "arm": "historical", "event": "completed",
             "measurement": report["runs"][0]["historical"]},
            {**common, "arm": "reused", "event": "started"},
            {**common, "arm": "reused", "event": "failed",
             "error": CONTROLLED_FAILURE},
            {"schema": "fy-full-training-progress-v1",
             "event": "process_failed", "error": CONTROLLED_FAILURE},
        ]
        encode = lambda rows: "".join(json.dumps(row) + "\\n" for row in rows)
        self.assertTrue(validate_interrupted_full_training_diagnostics(
            encode(events), SHA, BACKEND))
        for mutant in (
            events[:-1],
            events[1:],
            events[:1] + events[2:],
            events[:2] + events[3:],
            events + [events[-1]],
        ):
            with self.subTest(count=len(mutant)), self.assertRaises(ValueError):
                validate_interrupted_full_training_diagnostics(encode(mutant), SHA, BACKEND)
        for idx, key, value in (
            (1, "source_commit", "b" * 40),
            (1, "pair", 1),
            (1, "event", "started"),
            (3, "error", "other failure"),
            (4, "error", "other failure"),
        ):
            mutant = copy.deepcopy(events)
            mutant[idx][key] = value
            with self.subTest(idx=idx, key=key), self.assertRaises(ValueError):
                validate_interrupted_full_training_diagnostics(encode(mutant), SHA, BACKEND)
        mutant = copy.deepcopy(events)
        mutant[1]["measurement"]["training_ns"] = 0
        with self.assertRaises(ValueError):
            validate_interrupted_full_training_diagnostics(encode(mutant), SHA, BACKEND)


if __name__ == "__main__":
    unittest.main()
