"""TD2 Stage B independent numerical oracle; no tensor-framework dependency.

The small frozen NCHW proxy exercises the *training* form of batch norm.
Gradients are central differences of a pure Python forward implementation;
these are reference values to check a later Rust autodiff implementation,
not a measured Burn/Candle comparison or a framework selection.
"""

import argparse
import copy
import json
import math
import sys

SEED = 20261106
EPS = 1e-5
STEP = 1e-4
SHAPES = {"x": 32, "conv": 36, "depthwise": 18,
          "gamma": 2, "beta": 2, "target": 32}
TRAINABLE = ("x", "conv", "depthwise", "gamma", "beta")


def fixture():
    """Produce deterministic, framework-independent input and parameters."""
    state = SEED

    def draw():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state / 4294967296.0 - 0.5) * 0.8

    data = {k: [draw() for _ in range(n)] for k, n in SHAPES.items()}
    # Keep gamma away from zero so BN scale gradients are diagnostic.
    data["gamma"] = [1.2, -0.7]
    data["beta"] = [0.1, -0.2]
    return data


def _validate(data):
    for key, length in SHAPES.items():
        xs = data.get(key)
        if not isinstance(xs, list) or len(xs) != length:
            raise ValueError(f"{key}: expected {length} floats")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in xs):
            raise ValueError(f"{key}: nonfinite or nonnumeric input")


def forward(data):
    """NCHW: [1,2,4,4] -> conv3x3 -> depthwise3x3 -> train BN -> SiLU -> 2x.

    All kernels are cross-correlation; conv stride 1/no padding, depthwise
    stride 1/padding 1, population variance, BN epsilon=1e-5, nearest-repeat.
    Floating-point operations are Python f64 (not a bitwise f32 oracle).
    """
    _validate(data)
    x, conv, depthwise = data["x"], data["conv"], data["depthwise"]
    y = [0.0] * 8  # [2,2,2]
    for oc in range(2):
        for iy in range(2):
            for ix in range(2):
                val = 0.0
                for ic in range(2):
                    for ky in range(3):
                        for kx in range(3):
                            val += (x[ic*16 + (iy+ky)*4 + ix+kx]
                                    * conv[oc*18 + ic*9 + ky*3 + kx])
                y[oc*4+iy*2+ix] = val

    z = [0.0] * 8
    for c in range(2):
        for iy in range(2):
            for ix in range(2):
                val = 0.0
                for ky in range(3):
                    for kx in range(3):
                        sy, sx = iy+ky-1, ix+kx-1
                        if 0 <= sy < 2 and 0 <= sx < 2:
                            val += y[c*4+sy*2+sx] * depthwise[c*9+ky*3+kx]
                z[c*4+iy*2+ix] = val

    activated = [0.0] * 8
    for c in range(2):
        vals = z[c*4:c*4+4]
        mean = sum(vals) / 4.0
        var = sum((v-mean)**2 for v in vals) / 4.0
        for i, v in enumerate(vals):
            norm = (v-mean) / math.sqrt(var+EPS)
            bn = norm*data["gamma"][c] + data["beta"][c]
            activated[c*4+i] = bn/(1.0+math.exp(-bn))

    out = [0.0] * 32
    for c in range(2):
        for iy in range(4):
            for ix in range(4):
                out[c*16+iy*4+ix] = activated[c*4+(iy//2)*2+(ix//2)]
    return out


def loss(data):
    actual = forward(data)
    return sum((a-b)**2 for a, b in zip(actual, data["target"])) / 32.0


def finite_difference_gradients(data, h=STEP):
    """Central differences; independent of autograd and external libraries."""
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    _validate(data)
    mutable = copy.deepcopy(data)
    result = {}
    for name in TRAINABLE:
        grads = []
        for i in range(len(mutable[name])):
            base = mutable[name][i]
            mutable[name][i] = base + h
            plus = loss(mutable)
            mutable[name][i] = base - h
            minus = loss(mutable)
            mutable[name][i] = base
            grads.append((plus-minus)/(2*h))
        result[name] = grads
    return result


def evidence():
    f = fixture()
    return {
        "version": 1,
        "seed": SEED,
        "dtype": "python-f64-reference; Rust f32 comparison requires tolerance",
        "layout": "NCHW",
        "shapes": SHAPES,
        "operator_chain": "conv3x3>depthwise3x3>train_bn>silu>nearest_2x>mse",
        "batch_norm": {"training": True, "population_variance": True, "epsilon": EPS},
        "finite_difference_step": STEP,
        "inputs": f,
        "outputs": forward(f),
        "loss": loss(f),
        "gradient_oracle": finite_difference_gradients(f),
        "framework_gradients_compared": False,
        "framework_selected": False,
        "training_executed": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", help="Write stable JSON reference evidence; stdout otherwise")
    args = p.parse_args()
    payload = json.dumps(evidence(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)


if __name__ == "__main__":
    main()
