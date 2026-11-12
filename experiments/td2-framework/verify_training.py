"""Stage B3a: check in-process Burn/Candle SGD against an independent f64 trajectory.

``td2_proxy <backend> <oracle.json> --sgd <steps> <lr>`` trains the frozen
proxy inside one Rust process: forward, framework autodiff and the plain-SGD
update all run on the framework's own tensors. This script replays the same
seeds, initialization, learning rate and step count with the pure-Python
forward and central-difference gradients of ``reference_oracle`` and compares
every per-step loss and the final parameters. It makes no throughput, memory,
accuracy or framework-selection claim.
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
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"),
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    result = validate(report, args.backend, reference_trajectory())
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(f"{args.backend}: {STEPS} in-process SGD steps, loss "
          f"{result['initial_loss']:.8f} -> {result['final_loss']:.8f}, "
          f"max loss error {result['max_loss_error']:.3g}, "
          f"max parameter error {result['max_final_parameter_error']:.3g}")


if __name__ == "__main__":
    main()
