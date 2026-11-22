"""Offline negative controls for the B3a/B3b in-process training checks (no Rust claims)."""
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
                       lambda r: r.update(learning_rate=float("nan")),
                       lambda r: r.update(learning_rate=float("inf")),
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

    def test_scaled_learning_rate_is_caught(self):
        # A backend that applies 1.5x the declared step size must not pass.
        losses, finals = vt.reference_trajectory(learning_rate=1.5 * vt.LEARNING_RATE)
        report = oracle_report()
        report["losses"] = losses
        report["final_parameters"] = finals
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



ADAMW = vt.adamw_trajectory()


def adamw_report(backend="candle", trajectory=None, clip_norm=vt.GRAD_CLIP_NORM):
    losses, finals, rates, norms = ADAMW if trajectory is None else trajectory
    return {"backend": vt.BACKENDS[backend], "mode": "in-process-tensor-adamw-recipe",
            "steps": vt.STEPS, "grad_clip_norm": clip_norm, "learning_rates": list(rates),
            "gradient_norms": list(norms), "losses": list(losses),
            "final_parameters": copy.deepcopy(finals)}


# Probe threshold inside the proxy's gradient norms (0.905 falling to 0.704):
# the clip is active on the first three steps and inactive after, so both
# branches and the threshold itself are compared. Adam cancels a constant
# gradient scale, so a threshold below every norm would hide a wrong value.
# The frozen threshold of 10 is never reached on this proxy.
CLIP_PROBE = 0.8
ADAMW_CLIPPED = vt.adamw_trajectory(clip_norm=CLIP_PROBE)


class TestVerifyAdamWRecipe(unittest.TestCase):
    def test_schedule_is_the_frozen_d2b2a_one(self):
        self.assertEqual(vt.recipe_lr(0, 20), 2e-3)
        self.assertEqual(vt.recipe_lr(1, 20), 2e-3)
        self.assertAlmostEqual(vt.recipe_lr(19, 20), 1e-4, places=18)
        self.assertEqual(vt.recipe_lr(0, 40), 1e-3)
        for step, total in ((0, 19), (20, 20), (-1, 20)):
            with self.assertRaises(ValueError):
                vt.recipe_lr(step, total)

    def test_reference_descends_and_differs_from_sgd(self):
        losses, finals, _, norms = ADAMW
        self.assertLess(losses[-1], losses[0])
        self.assertLess(max(norms), vt.GRAD_CLIP_NORM)
        sgd_finals = REFERENCE[1]
        self.assertGreater(abs(finals["conv"][0] - sgd_finals["conv"][0]), 1e-4)
        result = vt.validate_adamw(adamw_report("burn"), "burn", ADAMW)
        self.assertEqual(result["max_loss_error"], 0.0)
        self.assertFalse(result["clip_active"])

    def test_plan_mismatches_rejected(self):
        for mutate in (lambda r: r.update(mode="in-process-tensor-sgd"),
                       lambda r: r.update(backend="burn-flex-0.22.0"),
                       lambda r: r.update(steps=21),
                       lambda r: r["learning_rates"].pop(),
                       lambda r: r.update(grad_clip_norm=0.5),
                       lambda r: r.pop("grad_clip_norm"),
                       lambda r: r["gradient_norms"].__setitem__(0, float("nan"))):
            report = adamw_report("candle")
            mutate(report)
            with self.assertRaises(ValueError):
                vt.validate_adamw(report, "candle", ADAMW)

    def test_recipe_drift_rejected(self):
        for mutate in (lambda r: r["learning_rates"].__setitem__(0, 1e-3),
                       lambda r: r["gradient_norms"].__setitem__(5, r["gradient_norms"][5] * 1.01),
                       lambda r: r["losses"].__setitem__(10, r["losses"][10] + 1e-3),
                       lambda r: r["final_parameters"]["gamma"].__setitem__(0, r["final_parameters"]["gamma"][0] + 1e-4)):
            report = adamw_report("candle")
            mutate(report)
            with self.assertRaises(AssertionError):
                vt.validate_adamw(report, "candle", ADAMW)

    def test_decay_on_batch_norm_scale_is_caught(self):
        wrong = dict(vt.DECAYED, gamma=True)
        original = vt.DECAYED
        try:
            vt.DECAYED = wrong
            mutated = vt.adamw_trajectory()
        finally:
            vt.DECAYED = original
        losses, finals, rates, norms = mutated
        report = {"backend": vt.BACKENDS["burn"], "mode": "in-process-tensor-adamw-recipe",
                  "steps": vt.STEPS, "grad_clip_norm": vt.GRAD_CLIP_NORM, "learning_rates": rates, "gradient_norms": norms,
                  "losses": losses, "final_parameters": finals}
        with self.assertRaises(AssertionError):
            vt.validate_adamw(report, "burn", ADAMW)

    def test_active_clip_path_is_compared(self):
        _, clipped_finals, _, clipped_norms = ADAMW_CLIPPED
        self.assertGreater(max(clipped_norms), CLIP_PROBE)
        report = adamw_report("burn", ADAMW_CLIPPED, CLIP_PROBE)
        result = vt.validate_adamw(report, "burn", ADAMW_CLIPPED, clip_norm=CLIP_PROBE)
        self.assertTrue(result["clip_active"])
        self.assertGreater(abs(clipped_finals["conv"][0] - ADAMW[1]["conv"][0]), 1e-6)
        # An unclipped run reported as a clip probe, a clip applied with the
        # wrong threshold, and a clip probe checked as the frozen recipe all fail.
        with self.assertRaises(AssertionError):
            vt.validate_adamw(adamw_report("burn", ADAMW, CLIP_PROBE), "burn",
                              ADAMW_CLIPPED, clip_norm=CLIP_PROBE)
        wrong = vt.adamw_trajectory(clip_norm=0.75)
        with self.assertRaises(AssertionError):
            vt.validate_adamw(adamw_report("burn", wrong, CLIP_PROBE), "burn",
                              ADAMW_CLIPPED, clip_norm=CLIP_PROBE)
        with self.assertRaises(ValueError):
            vt.validate_adamw(report, "burn", ADAMW)


if __name__ == "__main__":
    unittest.main()
