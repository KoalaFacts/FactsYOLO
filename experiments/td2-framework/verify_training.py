"""Stage B3: check in-process Burn/Candle training against an independent f64 trajectory.

``td2_proxy <backend> <oracle.json> --sgd <steps> <lr>`` trains the frozen
proxy inside one Rust process: forward, framework autodiff and the plain-SGD
update all run on the framework's own tensors. This script replays the same
seeds, initialization, learning rate and step count with the pure-Python
forward and central-difference gradients of ``reference_oracle`` and compares
every per-step loss and the final parameters. It makes no throughput, memory,
accuracy or framework-selection claim.

With ``--adamw-recipe`` (stage B3b) the report comes from
``td2_proxy <backend> <oracle.json> --adamw-recipe 20``: the frozen D2b2a
optimizer recipe (AdamW, decoupled decay on convolution weights only, global
L2 clip at 10, warmup then cosine) on framework tensors. The replay below
re-implements that recipe in f64 from spec section 6, not from the Rust code.
"""

import argparse
import json
import math
from pathlib import Path

import reference_oracle as oracle
import td2_native_sgd_replay as replay

FIELDS = replay.FIELDS
BACKENDS = replay.BACKENDS
STEPS = 20
LEARNING_RATE = 0.002
ATOL_LOSS = 2e-4
ATOL_PARAMETER = 2e-5


def reference_trajectory(steps=STEPS, learning_rate=LEARNING_RATE):
    """Independent losses and final parameters, starting from the f32 fixture."""
    state = {k: [replay.f32(v) for v in arr] for k, arr in oracle.fixture().items()}
    losses = []
    for step in range(steps + 1):
        losses.append(oracle.loss(state))
        if step < steps:
            replay.sgd_update(state, oracle.finite_difference_gradients(state), learning_rate)
    return losses, {k: state[k] for k in FIELDS}


# Frozen D2b2a recipe, transcribed from spec section 6.
BASE_LR = 2e-3
MIN_LR_FRACTION = 0.05
BETA1, BETA2 = 0.9, 0.999
EPSILON = 1e-8
WEIGHT_DECAY = 0.05
GRAD_CLIP_NORM = 10.0
DECAYED = {"conv": True, "depthwise": True, "gamma": False, "beta": False}


def recipe_lr(step, total):
    if total < 20 or not 0 <= step < total:
        raise ValueError("schedule needs total >= 20 and 0 <= step < total")
    w = max(1, total // 20)
    if step < w:
        return BASE_LR * (step + 1) / w
    progress = (step - w) / (total - 1 - w)
    floor = BASE_LR * MIN_LR_FRACTION
    return floor + (BASE_LR - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def adamw_trajectory(steps=STEPS):
    """Independent f64 AdamW-recipe losses, final parameters, rates and norms."""
    state = {k: [replay.f32(v) for v in arr] for k, arr in oracle.fixture().items()}
    m = {k: [0.0] * len(state[k]) for k in FIELDS}
    v = {k: [0.0] * len(state[k]) for k in FIELDS}
    losses, rates, norms = [], [], []
    for step in range(steps + 1):
        losses.append(oracle.loss(state))
        if step == steps:
            break
        grads = oracle.finite_difference_gradients(state)
        norm = math.sqrt(sum(g * g for k in FIELDS for g in grads[k]))
        scale = GRAD_CLIP_NORM / norm if norm > GRAD_CLIP_NORM else 1.0
        lr = recipe_lr(step, steps)
        c1, c2 = 1.0 - BETA1 ** (step + 1), 1.0 - BETA2 ** (step + 1)
        for k in FIELDS:
            for i, raw in enumerate(grads[k]):
                g = raw * scale
                m[k][i] = BETA1 * m[k][i] + (1.0 - BETA1) * g
                v[k][i] = BETA2 * v[k][i] + (1.0 - BETA2) * g * g
                w = state[k][i] * (1.0 - lr * WEIGHT_DECAY) if DECAYED[k] else state[k][i]
                state[k][i] = w - lr * (m[k][i] / c1) / (math.sqrt(v[k][i] / c2) + EPSILON)
        rates.append(lr)
        norms.append(norm)
    return losses, {k: state[k] for k in FIELDS}, rates, norms


def validate_adamw(report, backend, reference, steps=STEPS):
    if not isinstance(report, dict) or report.get("backend") != BACKENDS[backend]:
        raise ValueError("wrong backend identity")
    if report.get("mode") != "in-process-tensor-adamw-recipe":
        raise ValueError("report is not the in-process AdamW recipe")
    if report.get("steps") != steps or type(report.get("steps")) is not int:
        raise ValueError("step count differs from the matched plan")
    reference_losses, reference_finals, reference_rates, reference_norms = reference
    rates = replay.valid_numbers(report.get("learning_rates"), steps, "learning_rates")
    if max(abs(a - b) for a, b in zip(rates, reference_rates)) > 1e-15:
        raise AssertionError("learning-rate schedule differs from the frozen recipe")
    norms = replay.valid_numbers(report.get("gradient_norms"), steps, "gradient_norms")
    norm_error = max(abs(a - b) / max(1.0, b) for a, b in zip(norms, reference_norms))
    if norm_error > ATOL_PARAMETER:
        raise AssertionError(f"gradient norm mismatch: {norm_error}")
    result = validate(dict(report, mode="in-process-tensor-sgd", learning_rate=LEARNING_RATE),
                      backend, (reference_losses, reference_finals), steps)
    result.update({"optimizer": "adamw-d2b2a-recipe", "learning_rate": None,
                   "max_gradient_norm": max(norms), "clip_active": max(norms) > GRAD_CLIP_NORM,
                   "max_gradient_norm_relative_error": norm_error})
    return result


def validate(report, backend, reference, steps=STEPS, learning_rate=LEARNING_RATE):
    if not isinstance(report, dict) or report.get("backend") != BACKENDS[backend]:
        raise ValueError("wrong backend identity")
    if report.get("mode") != "in-process-tensor-sgd":
        raise ValueError("report is not in-process training")
    if report.get("steps") != steps or type(report.get("steps")) is not int:
        raise ValueError("step count differs from the matched plan")
    rate = report.get("learning_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or abs(rate - learning_rate) > 1e-9:
        raise ValueError("learning rate differs from the matched plan")
    losses = replay.valid_numbers(report.get("losses"), steps + 1, "losses")
    finals = report.get("final_parameters")
    if not isinstance(finals, dict) or set(finals) != set(FIELDS):
        raise ValueError("missing or unexpected final parameter")
    reference_losses, reference_finals = reference
    loss_error = max(abs(a - b) for a, b in zip(losses, reference_losses))
    if loss_error > ATOL_LOSS:
        raise AssertionError(f"loss trajectory mismatch: {loss_error}")
    parameter_error = 0.0
    for key in FIELDS:
        actual = replay.valid_numbers(finals[key], oracle.SHAPES[key], key)
        delta = max(abs(a - b) for a, b in zip(actual, reference_finals[key]))
        parameter_error = max(parameter_error, delta)
        if delta > ATOL_PARAMETER:
            raise AssertionError(f"{key} final parameter mismatch: {delta}")
    if losses[-1] >= losses[0]:
        raise AssertionError("in-process SGD did not lower loss")
    return {"backend": BACKENDS[backend], "steps": steps, "learning_rate": learning_rate,
            "initial_loss": losses[0], "final_loss": losses[-1],
            "max_loss_error": loss_error, "max_final_parameter_error": parameter_error,
            "atol_loss": ATOL_LOSS, "atol_parameter": ATOL_PARAMETER,
            "sgd_location": "in-process-framework-tensors",
            "framework_optimizer_api": False, "framework_selected": False,
            "framework_throughput": None, "peak_rss_bytes": None, "model_accuracy": None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=tuple(BACKENDS))
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--adamw-recipe", action="store_true",
                   help="check a --adamw-recipe report (stage B3b)")
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"),
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if args.adamw_recipe:
        result = validate_adamw(report, args.backend, adamw_trajectory())
    else:
        result = validate(report, args.backend, reference_trajectory())
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    label = "AdamW-recipe" if args.adamw_recipe else "SGD"
    print(f"{args.backend}: {STEPS} in-process {label} steps, loss "
          f"{result['initial_loss']:.8f} -> {result['final_loss']:.8f}, "
          f"max loss error {result['max_loss_error']:.3g}, "
          f"max parameter error {result['max_final_parameter_error']:.3g}")


if __name__ == "__main__":
    main()
