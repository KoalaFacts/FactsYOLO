# SPDX-License-Identifier: Apache-2.0
"""Do not mistake a crashed detector for successful bad-artifact rejection."""
import unittest

from verify import require_expected_inference_rejection


class NegativeInferenceGateTests(unittest.TestCase):
    def test_accepts_expected_cli_rejection(self):
        require_expected_inference_rejection(
            1, "error: artifact checksum mismatch\n", "artifact checksum mismatch"
        )
        require_expected_inference_rejection(
            1, "error: input exceeds bounded file size\n", "input exceeds bounded file size"
        )

    def test_rejects_signal_or_other_exit_even_if_diagnostic_matches(self):
        for exit_code in (0, -11, -6, 127, 134, 255):
            with self.subTest(exit_code=exit_code):
                with self.assertRaises(RuntimeError):
                    require_expected_inference_rejection(
                        exit_code, "error: artifact checksum mismatch\n", "artifact checksum mismatch"
                    )

    def test_rejects_unrelated_cli_failure(self):
        for output in (
            "error: permission denied\n",
            "error: invalid input image\n",
            "error: artifact checksum mismatch spoof\n",
            "",
        ):
            with self.subTest(output=output):
                with self.assertRaises(RuntimeError):
                    require_expected_inference_rejection(
                        1, output, "artifact checksum mismatch"
                    )
