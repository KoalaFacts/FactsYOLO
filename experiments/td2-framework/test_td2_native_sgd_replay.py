"""Offline negative controls for native-proxy SGD replay (no Rust claims)."""
import copy
import unittest

import reference_oracle as oracle
import td2_native_sgd_replay as replay


def oracle_executor(backend, state):
    return {"backend": replay.BACKENDS[backend], "outputs": oracle.forward(state),
            "loss": oracle.loss(state), "gradients": oracle.finite_difference_gradients(state)}


class TestSgdReplay(unittest.TestCase):
    def test_reference_simulation_is_external_and_descends(self):
        result = replay.replay("burn", oracle_executor, steps=20)
        self.assertEqual(len(result["states"]), 21)
        self.assertLess(result["states"][-1]["loss"], result["states"][0]["loss"])
        self.assertIsNone(result["framework_throughput"])
        self.assertFalse(result["rust_inprocess_optimizer_executed"])

    def test_nonfinite_wrong_backend_and_gradient_detected(self):
        state = oracle.fixture()
        valid = oracle_executor("burn", state)
        bad = copy.deepcopy(valid)
        bad["backend"] = "candle-cpu-0.11.0"
        with self.assertRaises(ValueError):
            replay.validate_report(bad, "burn", state)
        bad = copy.deepcopy(valid)
        bad["gradients"]["conv"][0] = float("nan")
        with self.assertRaises(ValueError):
            replay.validate_report(bad, "burn", state)
        bad = copy.deepcopy(valid)
        bad["gradients"]["conv"][0] += 0.02
        with self.assertRaises(AssertionError):
            replay.validate_report(bad, "burn", state)

    def test_f32_update_freezes_input_and_target(self):
        state = {k:[replay.f32(v) for v in arr] for k,arr in oracle.fixture().items()}
        original_x, original_target = state["x"].copy(), state["target"].copy()
        grads = oracle.finite_difference_gradients(state)
        replay.sgd_update(state, grads, 0.002)
        self.assertEqual(state["x"], original_x)
        self.assertEqual(state["target"], original_target)
        self.assertNotEqual(state["conv"], [replay.f32(v) for v in oracle.fixture()["conv"]])


if __name__ == "__main__":
    unittest.main()
