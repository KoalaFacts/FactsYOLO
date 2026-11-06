"""Negative controls for the Rust framework report verification gate."""
import copy
import importlib.util
import pathlib
import unittest

root = pathlib.Path(__file__).parent


def load(name):
    path = root / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load("reference_oracle")
gate = load("verify_backend")


class FrameworkVerificationTests(unittest.TestCase):
    def setUp(self):
        self.reference = oracle.evidence()
        self.report = {
            "backend": "candle-cpu-0.11.0",
            "outputs": self.reference["outputs"].copy(),
            "loss": self.reference["loss"],
            "gradients": copy.deepcopy(self.reference["gradient_oracle"]),
        }

    def test_valid_complete_reference_passes(self):
        measured = gate.compare(self.reference, self.report, "candle")
        self.assertTrue(measured["passed"])
        self.assertEqual(measured["max_forward_abs_error"], 0)
        self.assertTrue(all(v == 0 for v in measured["max_gradient_abs_error"].values()))

    def test_wrong_backend_and_oracle_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "backend identity"):
            gate.compare(self.reference, self.report, "burn")
        other = copy.deepcopy(self.reference)
        other["seed"] += 1
        with self.assertRaisesRegex(ValueError, "oracle fixture"):
            gate.compare(other, self.report, "candle")

    def test_missing_gradient_and_wrong_length_rejected(self):
        bad = copy.deepcopy(self.report)
        del bad["gradients"]["x"]
        with self.assertRaisesRegex(ValueError, "gradient fields"):
            gate.compare(self.reference, bad, "candle")
        bad = copy.deepcopy(self.report)
        bad["gradients"]["depthwise"].pop()
        with self.assertRaisesRegex(ValueError, "depthwise"):
            gate.compare(self.reference, bad, "candle")

    def test_forward_output_or_loss_corruption_rejected(self):
        bad = copy.deepcopy(self.report)
        bad["outputs"][13] += 0.01
        with self.assertRaisesRegex(ValueError, "forward mismatch"):
            gate.compare(self.reference, bad, "candle")
        bad = copy.deepcopy(self.report)
        bad["loss"] += 0.01
        with self.assertRaisesRegex(ValueError, "MSE"):
            gate.compare(self.reference, bad, "candle")

    def test_zero_or_wrong_gradient_and_nonfinite_rejected(self):
        bad = copy.deepcopy(self.report)
        bad["gradients"]["conv"] = [0.0] * 36
        with self.assertRaisesRegex(ValueError, "conv autodiff"):
            gate.compare(self.reference, bad, "candle")
        bad = copy.deepcopy(self.report)
        bad["gradients"]["gamma"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "gamma"):
            gate.compare(self.reference, bad, "candle")
        bad = copy.deepcopy(self.report)
        bad["outputs"][0] = True
        with self.assertRaisesRegex(ValueError, "backend output"):
            gate.compare(self.reference, bad, "candle")

    def test_frozen_oracle_payload_rejects_changed_input_and_gradient(self):
        # The same seed and operator-chain labels do not pin the actual bytes.
        # Both changes previously passed comparison against the old report.
        bad = copy.deepcopy(self.reference)
        bad["inputs"]["x"][0] += 0.5
        with self.assertRaisesRegex(ValueError, "frozen oracle payload"):
            gate.compare(bad, self.report, "candle")
        bad = copy.deepcopy(self.reference)
        bad["gradient_oracle"]["gamma"][0] += 1e-5
        with self.assertRaisesRegex(ValueError, "frozen oracle payload"):
            gate.compare(bad, self.report, "candle")


if __name__ == "__main__":
    unittest.main()
