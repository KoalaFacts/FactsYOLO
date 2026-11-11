# TD2 Stage B: independent operator/gradient reference (first slice)

This **framework-independent numerical oracle** is the first small, runnable
reference for the [accepted TD2 Stage B framework decision](../../docs/superpowers/specs/2026-11-05-td2-general-detector-design.md).
It does not select Burn or Candle and does not replace verified Rust
framework execution, matched multi-step training, license review or CPU measurements.

The CPython 3.12–3.14 standard-library-only script uses a frozen LCG seed, an NCHW
`[1,2,4,4]` input, 3×3 convolution (`2→2` channels, valid padding), 3×3
depthwise convolution (same padding), **training** batch normalization (spatial
population variance; `epsilon=1e-5`), SiLU, nearest-neighbour 2× upsampling,
and mean-squared error against a fixed target. It computes central-difference
input, convolution, depthwise, gamma and beta gradients using an **independent
f64 forward**. It is explicitly not a Rust-f32 bitwise oracle: tolerance and
finite-difference step sensitivity must be assessed when comparing autodiff.

```sh
python3 -m unittest discover -s experiments/td2-framework -p 'test_reference_oracle.py' -v
python3 experiments/td2-framework/reference_oracle.py --output /tmp/td2-b-oracle.json
```

The output contains the exact inputs, parameters, outputs, scalar loss, all
finite-difference gradients and a machine-readable `framework_selected=false`
boundary. **Generating the byte-pinned reference requires CPython 3.12–3.14**:
3.10/3.11 produce different low-order f64 finite-difference bytes, so the
script now rejects those unsupported generators explicitly. Do not relax the
full-payload SHA-256 check or silently regenerate a new digest. The GitHub
Linux/Windows jobs validate the supported runtime via the oracle tests. No model weights or external datasets are used or published. The
following Stage B gate implements **the same fixed proxy** in Burn Flex
and Candle and checks their forward and autodiff gradients against this reference.
After Rust execution is verified, the next task is to run matched training
steps and measure CPU throughput/peak memory. GPU support,
operator coverage, reproducibility and dependency licensing must still be
assessed before ADR 0002 can select a framework.

Original source: Apache-2.0. This is AI-assisted implementation, with no DCO
sign-off or human certification asserted on another person's behalf.

## First real Rust Burn/Candle candidate check (Stage B, incomplete)

The Stage B gate now includes `td2_proxy`, a **real** eager CPU/autodiff
operator graph for **both** pinned Rust backends (`burn 0.22.0`,
`candle-core 0.11.0`). It consumes the **same** frozen JSON reference:
3×3 convolution, grouped depthwise convolution, train-time population BN
(no running stats), SiLU, nearest ×2 and MSE. It records all 32 outputs,
the loss and gradients for all 90 input/parameter values.

For now the binary lives under the existing **experimental-only**
`experiments/cpu-detector/evaluate` Cargo workspace so the exact
reviewed Burn/Candle versions and lockfile are reused. This is isolated
from the TD1 architecture, weights, artifact/ABI and training scripts.
Do not link this binary or these dependencies into the published core.

```sh
python3 -m unittest discover -s experiments/td2-framework -p 'test_*.py' -v
python3 experiments/td2-framework/reference_oracle.py --output /tmp/td2-b-oracle.json
cargo fetch --manifest-path experiments/cpu-detector/Cargo.toml --locked
for name in burn candle; do
  RAYON_NUM_THREADS=1 cargo run --manifest-path experiments/cpu-detector/Cargo.toml \
    --bin td2_proxy --release --locked --offline -- "$name" /tmp/td2-b-oracle.json \
    > "/tmp/td2-$name.json"
  python3 experiments/td2-framework/verify_backend.py "$name" \
    /tmp/td2-b-oracle.json "/tmp/td2-$name.json"
done
```

The comparison gate rejects missing/nonfinite outputs and gradients,
unmatching model/loss values and wrong backend identity; finite differences
use Python f64 and Rust graphs run F32, so this is a **tolerance-based**
operator check, not a bitwise reproduction. GitHub Actions records the
actual Linux/Windows results; a green Python test alone **does not** mean
either Rust backend compiled or passed.

**Not yet delivered:** training through each framework's optimizer API
(SGD with momentum/Adam), CPU throughput and peak-memory baselines, GPU viability, licensing decision
for any distributed binary, and ADR 0002 framework selection. The oracle
contains no licensed datasets or pre-trained weights.

## External-SGD replay through actual Rust autodiff (development-only)

The optional `td2_native_sgd_replay.py` harness runs the **compiled** native
`td2_proxy` executable at 21 consecutive F32 states on both backends, applying
20 matched SGD updates (learning rate 0.002) to the 58 convolution, depthwise
and BN affine parameters in Python between calls. The image, target, seed and
model are fixed; every state is checked against fresh independent Python f64
forward, MSE and finite-difference gradients. Full per-step loss/errors and the
native executable SHA-256 are preserved by the TD2 CI on Linux and Windows.
Native ELF/PE magic prevents a Python test stub from being silently labelled
a Rust execution, but does not attest that the binary came from a given commit.

This is **external SGD**, not framework-internal optimizer training. Reported
walltime includes subprocess starts and the Python numerical oracle and MUST
NOT be called framework training throughput. No CPU peak-memory, GPU, framework
selection or production accuracy claim follows from this gate.

## Matched in-process SGD on framework tensors (stage B3a)

`td2_proxy <burn|candle> <oracle.json> --sgd 20 0.002` trains the frozen proxy
inside one Rust process: each step runs the forward, the framework's own
autodiff, and the plain-SGD update `w <- w - lr * grad` on the framework's
tensors (Candle `Var::set`; Burn inner-backend tensors re-attached with
`require_grad`). Image and target stay frozen; the 58 convolution, depthwise
and BN affine values train. Seeds, initialization, learning rate and step count
are identical for both backends. The report holds all 21 losses, the final
parameters and the wall time of the training loop alone.

```sh
for name in burn candle; do
  RAYON_NUM_THREADS=1 cargo run --manifest-path experiments/cpu-detector/Cargo.toml \
    --bin td2_proxy --release --locked --offline -- "$name" /tmp/td2-b-oracle.json \
    --sgd 20 0.002 > "/tmp/td2-$name-sgd.json"
  python3 experiments/td2-framework/verify_training.py "$name" "/tmp/td2-$name-sgd.json"
done
```

`verify_training.py` replays the same 20 steps with the pure-Python f64
forward and central-difference gradients, then requires every loss within
2e-4 and every final parameter within 2e-5, and a lower final loss. Its tests
reject wrong identity or plan, non-finite values, loss or parameter drift,
skipped updates and gradient ascent. Steps are capped at 100 and the learning
rate must be in (0, 1]. This is not the frameworks' optimizer API, and the
loop time on a 4×4 proxy is not a throughput figure (stage B4).
