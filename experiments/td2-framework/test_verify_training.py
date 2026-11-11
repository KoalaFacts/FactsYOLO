"""Offline negative controls for the B3a in-process training check (no Rust claims)."""
import copy
import unittest

import verify_training as vt

REFERENCE = vt.reference_trajectory()


def oracle_report(backend="burn"):
    losses, finals = REFERENCE
    return {"backend": vt.BACKENDS[backend], "mode": "in-process-tensor-sgd",
            "steps": vt.STEPS, "learning_rate": vt.LEARNING_RATE,
            "losses": list(losses), "final_parameters": copy.deepcopy(finals)}


class TestVerifyTraining(unittest.TestCase):
    def test_reference_descends_and_matches_itself(self):
        losses, _ = REFERENCE
        self.assertEqual(len(losses), vt.STEPS + 1)
        self.assertLess(losses[-1], losses[0])
        result = vt.validate(oracle_report("candle"), "candle", REFERENCE)
        self.assertEqual(result["max_loss_error"], 0.0)
        self.assertFalse(result["framework_selected"])

    def test_identity_and_plan_mismatches_rejected(self):
        for mutate in (lambda r: r.update(backend="candle-cpu-0.11.0"),
                       lambda r: r.update(mode="external-python-driver"),
                       lambda r: r.update(steps=19),
                       lambda r: r.update(steps=20.0),
                       lambda r: r.update(learning_rate=0.003),
                       lambda r: r["losses"].pop(),
                       lambda r: r["losses"].__setitem__(3, float("nan")),
                       lambda r: r["final_parameters"].pop("beta"),
                       lambda r: r["final_parameters"]["conv"].pop()):
            report = oracle_report("burn")
            mutate(report)
            with self.assertRaises(ValueError):
                vt.validate(report, "burn", REFERENCE)

    def test_numerical_drift_rejected(self):
        report = oracle_report()
        report["losses"][10] += 3 * vt.ATOL_LOSS
        with self.assertRaises(AssertionError):
            vt.validate(report, "burn", REFERENCE)
        report = oracle_report()
        report["final_parameters"]["gamma"][1] += 3 * vt.ATOL_PARAMETER
        with self.assertRaises(AssertionError):
            vt.validate(report, "burn", REFERENCE)

    def test_skipped_updates_are_caught(self):
        # A backend that never applies its update keeps the initial parameters.
        report = oracle_report()
        initial = {k: [vt.replay.f32(v) for v in vt.oracle.fixture()[k]] for k in vt.FIELDS}
        report["final_parameters"] = initial
        with self.assertRaises(AssertionError):
            vt.validate(report, "burn", REFERENCE)

    def test_wrong_sign_update_is_caught(self):
        # Gradient ascent with the same step size must not pass as training.
        state = {k: [vt.replay.f32(v) for v in arr] for k, arr in vt.oracle.fixture().items()}
        ascent = []
        for step in range(vt.STEPS + 1):
            ascent.append(vt.oracle.loss(state))
            if step < vt.STEPS:
                grads = vt.oracle.finite_difference_gradients(state)
                grads = {k: [-g for g in grads[k]] for k in vt.FIELDS}
                vt.replay.sgd_update(state, grads, vt.LEARNING_RATE)
        report = oracle_report()
        report["losses"] = ascent
        report["final_parameters"] = {k: state[k] for k in vt.FIELDS}
        with self.assertRaises(AssertionError):
            vt.validate(report, "burn", REFERENCE)


if __name__ == "__main__":
    unittest.main()
