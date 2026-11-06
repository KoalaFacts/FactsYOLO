"""Stage B: native Burn/Candle autodiff with *external* fixed-step F32 SGD.

This development-only validation repeatedly runs the already built Rust
``td2_proxy`` executable. SGD is computed here, in Python, not inside either
Rust framework. Process launch and Python-oracle time are not training speed.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import reference_oracle as oracle

FIELDS = ("conv", "depthwise", "gamma", "beta")
GRAD_FIELDS = ("x", *FIELDS)
BACKENDS = {"burn": "burn-flex-0.22.0", "candle": "candle-cpu-0.11.0"}
ATOL_OUTPUT = 2e-4
ATOL_GRADIENT = 1e-5


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def valid_numbers(value, size, name):
    if not isinstance(value, list) or len(value) != size or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in value
    ):
        raise ValueError(f"{name}: expected {size} finite values")
    return value


def validate_report(report, backend, state):
    if not isinstance(report, dict) or report.get("backend") != BACKENDS[backend]:
        raise ValueError("wrong backend identity")
    observed = valid_numbers(report.get("outputs"), 32, "outputs")
    value = report.get("loss")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("missing or nonfinite loss")
    gradients = report.get("gradients")
    if not isinstance(gradients, dict) or set(gradients) != set(GRAD_FIELDS):
        raise ValueError("missing or unexpected gradient")
    expected = oracle.forward(state)
    error = max(abs(a - b) for a, b in zip(observed, expected))
    if error > ATOL_OUTPUT:
        raise AssertionError(f"forward mismatch: {error}")
    oracle_loss = oracle.loss(state)
    loss_error = abs(value - oracle_loss)
    if loss_error > ATOL_OUTPUT or abs(value - sum((a-b)**2 for a,b in zip(observed, state["target"])) / 32) > ATOL_OUTPUT:
        raise AssertionError(f"MSE mismatch: {loss_error}")
    expected_grads = oracle.finite_difference_gradients(state)
    grad_error = 0.0
    for key in GRAD_FIELDS:
        actual = valid_numbers(gradients.get(key), oracle.SHAPES[key], key)
        delta = max(abs(a - b) for a, b in zip(actual, expected_grads[key]))
        grad_error = max(grad_error, delta)
        if delta > ATOL_GRADIENT:
            raise AssertionError(f"{key} gradient mismatch: {delta}")
    return error, loss_error, grad_error


def sgd_update(state, grads, lr):
    # No optimizer state, no learnable image/target. Explicit F32 step semantics.
    for key in FIELDS:
        state[key] = [f32(f32(w) - f32(f32(lr) * f32(g)))
                      for w, g in zip(state[key], grads[key])]


def invoke(binary, backend, state):
    with tempfile.TemporaryDirectory(prefix="td2-sgd-") as tmp:
        path = Path(tmp) / "input.json"
        path.write_text(json.dumps({"inputs": state}, allow_nan=False), encoding="utf-8")
        completed = subprocess.run([str(binary), backend, str(path)],
                                   capture_output=True, text=True, timeout=30, check=False)
        if completed.returncode:
            raise RuntimeError(f"td2_proxy exited {completed.returncode}: {completed.stderr[:1000]}")
        return json.loads(completed.stdout, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def replay(backend, executor, steps=20, learning_rate=0.002):
    if backend not in BACKENDS or type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("invalid backend or step count")
    if type(learning_rate) not in (int, float) or not 0 < learning_rate <= 1 or not math.isfinite(learning_rate):
        raise ValueError("invalid learning rate")
    # Initial state, input and target are explicitly rounded to native F32.
    state = {k: [f32(v) for v in arr] for k, arr in oracle.fixture().items()}
    immutable = (state["x"].copy(), state["target"].copy())
    records = []
    start = time.perf_counter()
    for i in range(steps + 1):
        report = executor(backend, copy.deepcopy(state))
        out_err, loss_err, grad_err = validate_report(report, backend, state)
        records.append({"index": i, "loss": report["loss"],
                        "reference_loss": oracle.loss(state),
                        "max_output_error": out_err, "loss_error": loss_err,
                        "max_gradient_error": grad_err})
        if i < steps:
            sgd_update(state, report["gradients"], learning_rate)
        if (state["x"], state["target"]) != immutable:
            raise AssertionError("input or target mutated")
    if records[-1]["loss"] >= records[0]["loss"]:
        raise AssertionError("SGD did not lower loss")
    return {"backend": BACKENDS[backend], "steps": steps,
            "learning_rate": learning_rate, "states": records,
            "sgd_location": "external-python-driver",
            "rust_inprocess_optimizer_executed": False,
            "walltime_including_process_and_oracle_seconds": time.perf_counter() - start,
            "framework_throughput": None, "peak_rss_bytes": None,
            "model_accuracy": None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    binary = args.binary.resolve(strict=True)
    if not binary.is_file():
        raise ValueError("td2_proxy must be an executable file")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    with binary.open("rb") as f:
        magic = f.read(4)
    if magic != b"\x7fELF" and magic[:2] != b"MZ":
        raise ValueError("expected a native ELF or PE executable, not a Python mock")
    result = replay(args.backend, lambda backend, state: invoke(binary, backend, state))
    result.update({"execution": "native-Rust-binary-subprocess-with-external-SGD",
                   "binary_sha256": digest, "binary_magic": magic.hex(),
                   "source_commit_attested_by_binary": False})
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.backend}: {result['steps']} updates, loss {result['states'][0]['loss']:.8f} -> {result['states'][-1]['loss']:.8f}")


if __name__ == "__main__":
    main()
