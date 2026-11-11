"""Independent Stage B reference-oracle regressions (Python stdlib only)."""
import importlib.util
import math
import pathlib
import unittest

spec = importlib.util.spec_from_file_location(
    "td2_reference_oracle", pathlib.Path(__file__).with_name("reference_oracle.py")
)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class OracleTests(unittest.TestCase):
    def test_repeatable_fixture_and_shape(self):
        a = oracle.fixture()
        self.assertEqual(a, oracle.fixture())
        self.assertEqual(len(a["x"]), 32)
        self.assertEqual(len(a["conv"]), 36)
        self.assertEqual(len(a["depthwise"]), 18)
        self.assertEqual(len(a["target"]), 32)
        self.assertEqual(len(oracle.forward(a)), 32)
        self.assertTrue(math.isfinite(oracle.loss(a)))

    def test_zero_weights_leave_only_batch_norm_beta(self):
        f = oracle.fixture()
        f["conv"] = [0.0] * len(f["conv"])
        out = oracle.forward(f)
        for c in range(2):
            beta = f["beta"][c]
            expected = beta / (1.0 + math.exp(-beta))
            self.assertEqual(out[c*16:c*16+16], [expected]*16)
        self.assertAlmostEqual(
            oracle.loss(f),
            sum((o-t)**2 for o, t in zip(out, f["target"])) / 32.0,
            places=12,
        )

    def test_gradient_checks_are_finite_and_deterministic(self):
        f = oracle.fixture()
        grads = oracle.finite_difference_gradients(f)
        self.assertEqual(set(grads), {"x", "conv", "depthwise", "gamma", "beta"})
        for k, values in grads.items():
            self.assertEqual(len(values), len(f[k]))
            self.assertTrue(all(math.isfinite(v) for v in values))
        self.assertEqual(grads, oracle.finite_difference_gradients(oracle.fixture()))
        self.assertTrue(any(abs(g) > 1e-7 for g in grads["conv"]))
        tighter = oracle.finite_difference_gradients(f, h=5e-5)
        self.assertLess(
            max(abs(a-b) for key in grads for a, b in zip(grads[key], tighter[key])),
            5e-7,
        )

    def test_mutation_changes_loss_and_output(self):
        f = oracle.fixture()
        old_loss, old_result = oracle.loss(f), oracle.forward(f)
        f["conv"][0] += 0.5
        self.assertNotEqual(old_result, oracle.forward(f))
        self.assertNotAlmostEqual(old_loss, oracle.loss(f), places=10)
        self.assertNotEqual(
            oracle.finite_difference_gradients(oracle.fixture())["conv"],
            oracle.finite_difference_gradients(f)["conv"],
        )

    def test_invalid_lengths_and_nonfinite_values_rejected(self):
        f = oracle.fixture()
        f["x"].pop()
        with self.assertRaisesRegex(ValueError, "x"):
            oracle.forward(f)
        f = oracle.fixture()
        f["depthwise"][17] = float("nan")
        with self.assertRaisesRegex(ValueError, "depthwise"):
            oracle.loss(f)

    def test_emitted_evidence_provenance_and_recompute(self):
        evidence = oracle.evidence()
        self.assertEqual(evidence["seed"], 20261106)
        self.assertEqual(evidence["operator_chain"], "conv3x3>depthwise3x3>train_bn>silu>nearest_2x>mse")
        self.assertEqual(evidence["outputs"], oracle.forward(evidence["inputs"]))
        self.assertEqual(evidence["loss"], oracle.loss(evidence["inputs"]))
        self.assertEqual(evidence["gradient_oracle"], oracle.finite_difference_gradients(evidence["inputs"]))

    def test_supported_frozen_reference_interpreter_contract(self):
        # Verified byte-equivalent oracle generation is narrower than Python 3.
        for minor in (12, 13, 14):
            oracle.require_supported_python((3, minor), "cpython")
        for minor in (10, 11, 15):
            with self.subTest(minor=minor):
                with self.assertRaisesRegex(RuntimeError, "CPython 3.12-3.14"):
                    oracle.require_supported_python((3, minor), "cpython")
        with self.assertRaisesRegex(RuntimeError, "CPython 3.12-3.14"):
            oracle.require_supported_python((3, 13), "pypy")

    def test_current_runtime_emits_pinned_frozen_bytes(self):
        import hashlib
        import json

        oracle.require_supported_python()
        payload = json.dumps(oracle.evidence(), sort_keys=True, indent=2,
                             allow_nan=False) + "\n"
        self.assertEqual(
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "0fb5ef34f70be76cd498090b40a91b69b566c1b50bb08156512a7df14d4adbea",
        )

if __name__ == "__main__":
    unittest.main()
