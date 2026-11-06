"""Validate real Rust Burn/Candle proxy autodiff against the independent f64 oracle.

This gate intentionally has no Rust/tensor dependency. It rejects missing,
nonfinite, reordered, truncated or fabricated outputs and parameter gradients.
It does not measure training throughput, memory, or GPU capability.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

FIELDS = ("x", "conv", "depthwise", "gamma", "beta")
BACKENDS = {"burn": "burn-flex-0.22.0", "candle": "candle-cpu-0.11.0"}
OUTPUT_ATOL = 2e-4
GRADIENT_ATOL = 4e-4
# SHA-256 of reference_oracle.py --output JSON: sorted keys, indent=2, newline.
# The seed/chain strings alone cannot guard against changed fixture parameters.
FROZEN_ORACLE_SHA256 = "0fb5ef34f70be76cd498090b40a91b69b566c1b50bb08156512a7df14d4adbea"


def _numbers(values, size, label):
    if (not isinstance(values, list) or len(values) != size
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for v in values)):
        raise ValueError(f"{label}: expected {size} finite numbers")
    return values


def compare(reference, actual, backend):
    """Return maximum errors, or fail a malformed/inaccurate framework report."""
    if backend not in BACKENDS or actual.get("backend") != BACKENDS[backend]:
        raise ValueError("backend identity mismatch")
    if reference.get("operator_chain") != "conv3x3>depthwise3x3>train_bn>silu>nearest_2x>mse":
        raise ValueError("oracle operator contract changed")
    if reference.get("seed") != 20261106:
        raise ValueError("oracle fixture changed")
    try:
        canonical = (json.dumps(reference, sort_keys=True, indent=2,
                                allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed frozen oracle payload") from exc
    if hashlib.sha256(canonical).hexdigest() != FROZEN_ORACLE_SHA256:
        raise ValueError("frozen oracle payload changed")
    expected = _numbers(reference["outputs"], 32, "reference output")
    observed = _numbers(actual.get("outputs"), 32, "backend output")
    max_output_error = max(abs(a-b) for a, b in zip(expected, observed))
    if max_output_error > OUTPUT_ATOL:
        raise ValueError(f"forward mismatch max={max_output_error:.9g}")
    loss = actual.get("loss")
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise ValueError("backend loss is nonfinite or missing")
    expected_loss = reference["loss"]
    target = _numbers(reference["inputs"]["target"], 32, "target")
    recomputed = sum((a-b)**2 for a, b in zip(observed, target)) / 32
    if abs(loss-expected_loss) > OUTPUT_ATOL or abs(loss-recomputed) > OUTPUT_ATOL:
        raise ValueError("backend MSE loss mismatch")
    gradients = actual.get("gradients")
    if not isinstance(gradients, dict) or set(gradients) != set(FIELDS):
        raise ValueError("missing or unexpected gradient fields")
    max_errors = {}
    for key in FIELDS:
        length = len(reference["gradient_oracle"][key])
        expected_grad = _numbers(reference["gradient_oracle"][key], length, f"reference {key}")
        actual_grad = _numbers(gradients[key], length, f"backend {key}")
        discrepancy = max(abs(a-b) for a, b in zip(expected_grad, actual_grad))
        if discrepancy > GRADIENT_ATOL:
            raise ValueError(f"{key} autodiff mismatch max={discrepancy:.9g}")
        max_errors[key] = discrepancy
    return {"backend": backend, "passed": True, "max_forward_abs_error": max_output_error,
            "max_gradient_abs_error": max_errors,
            "max_loss_abs_error": abs(loss-expected_loss),
            "output_atol": OUTPUT_ATOL, "gradient_atol": GRADIENT_ATOL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=sorted(BACKENDS))
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    actual = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare(reference, actual, args.backend)
    report = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
