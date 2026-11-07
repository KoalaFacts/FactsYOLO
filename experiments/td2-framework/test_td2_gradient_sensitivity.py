"""Detect silent removal of the smallest valid proxy gradient in both gates."""
import copy
import unittest

import reference_oracle as oracle
import td2_native_sgd_replay as replay
import verify_backend as gate


class SmallGradientRegression(unittest.TestCase):
    def test_frozen_report_rejects_zeroed_small_conv_gradient(self):
        reference = oracle.evidence()
        candidate = {
            "backend": gate.BACKENDS["burn"],
            "outputs": reference["outputs"].copy(),
            "loss": reference["loss"],
            "gradients": copy.deepcopy(reference["gradient_oracle"]),
        }
        self.assertTrue(gate.compare(reference, candidate, "burn")["passed"])
        values = reference["gradient_oracle"]["conv"]
        index = min(range(len(values)), key=lambda i: abs(values[i]))
        self.assertGreater(abs(values[index]), 1e-5)
        self.assertLess(abs(values[index]), 4e-4)
        candidate["gradients"]["conv"][index] = 0.0
        with self.assertRaisesRegex(ValueError, "conv autodiff mismatch"):
            gate.compare(reference, candidate, "burn")

    def test_external_sgd_replay_rejects_zeroed_small_conv_gradient(self):
        state = {key: [replay.f32(v) for v in values]
                 for key, values in oracle.fixture().items()}
        gradients = oracle.finite_difference_gradients(state)
        candidate = {
            "backend": replay.BACKENDS["candle"],
            "outputs": oracle.forward(state),
            "loss": oracle.loss(state),
            "gradients": copy.deepcopy(gradients),
        }
        replay.validate_report(candidate, "candle", state)
        values = gradients["conv"]
        index = min(range(len(values)), key=lambda i: abs(values[i]))
        self.assertGreater(abs(values[index]), 1e-5)
        self.assertLess(abs(values[index]), 4e-4)
        candidate["gradients"]["conv"][index] = 0.0
        with self.assertRaisesRegex(AssertionError, "conv gradient mismatch"):
            replay.validate_report(candidate, "candle", state)


if __name__ == "__main__":
    unittest.main()
