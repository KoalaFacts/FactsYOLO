# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the fixed TD1 inference benchmark evidence gate."""
import unittest

from inference_gate import verify_inference_benchmark


class InferenceBenchmarkGateTests(unittest.TestCase):
    @staticmethod
    def valid():
        return {
            "source_commit": "test-revision", "build": "release", "images": 8,
            "iterations_per_repeat": 256, "repetitions": 7,
            "exact_positive_and_background_parity": True,
            "inputs_sha256": "3fc69efa48a1b4d9a7f596476c0c405295dc96355c9c587b4058bce20aebcca4",
            "model_sha256": "4d3ce2b2a284dd2644f91ae4b73a12c486d6a12d9fd1cecfe4c9ce8fd5b230bd",
            "historical": [{"elapsed_ns": 19_000_000,
                            "allocation_requests": 2048,
                            "requested_bytes": 1_572_864} for _ in range(7)],
            "current": [{"elapsed_ns": 22_000_000,
                         "allocation_requests": 0,
                         "requested_bytes": 0} for _ in range(7)],
        }

    def test_accepts_registered_complete_measurements(self):
        valid = self.valid()
        self.assertIs(verify_inference_benchmark(valid, "test-revision"), valid)

    def test_rejects_mutated_input_and_model_fingerprints(self):
        for key in ("inputs_sha256", "model_sha256"):
            with self.subTest(key=key):
                sample = self.valid()
                sample[key] = "f" * 64
                with self.assertRaises(ValueError):
                    verify_inference_benchmark(sample, "test-revision")

    def test_rejects_unreported_or_nonzero_allocated_bytes(self):
        for mode in ("historical", "current"):
            for value in (None, 768):
                with self.subTest(mode=mode, value=value):
                    sample = self.valid()
                    if value is None:
                        del sample[mode][0]["requested_bytes"]
                    else:
                        sample[mode][0]["requested_bytes"] = value
                    with self.assertRaises(ValueError):
                        verify_inference_benchmark(sample, "test-revision")

    def test_rejects_zero_negative_or_missing_timing(self):
        for value in (0, -1, None, False, "2"):
            with self.subTest(value=value):
                sample = self.valid()
                sample["current"][0]["elapsed_ns"] = value
                with self.assertRaises(ValueError):
                    verify_inference_benchmark(sample, "test-revision")

    def test_rejects_stale_sha_incomplete_or_false_parity(self):
        sample = self.valid()
        with self.assertRaises(ValueError):
            verify_inference_benchmark(sample, "different-revision")
        sample = self.valid()
        sample["historical"].pop()
        with self.assertRaises(ValueError):
            verify_inference_benchmark(sample, "test-revision")
        sample = self.valid()
        sample["exact_positive_and_background_parity"] = False
        with self.assertRaises(ValueError):
            verify_inference_benchmark(sample, "test-revision")

    def test_rejects_missing_repetition_fields_and_wrong_counts(self):
        sample = self.valid()
        del sample["historical"][0]["allocation_requests"]
        with self.assertRaises(ValueError):
            verify_inference_benchmark(sample, "test-revision")
        sample = self.valid()
        sample["historical"][0]["allocation_requests"] = 2047
        with self.assertRaises(ValueError):
            verify_inference_benchmark(sample, "test-revision")


if __name__ == "__main__":
    unittest.main()
