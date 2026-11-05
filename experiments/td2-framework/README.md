# TD2 Stage B: independent operator/gradient reference (first slice)

This **framework-independent numerical oracle** is the first small, runnable
reference for the [accepted TD2 Stage B framework decision](../../docs/superpowers/specs/2026-11-05-td2-general-detector-design.md).
It does not select Burn or Candle and does not replace the required matched
Rust implementations, multi-step training, license review or CPU measurements.

The Python 3 standard-library-only script uses a frozen LCG seed, an NCHW
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
boundary. No model weights or external datasets are used or published. The
next Stage B task is to implement **the same fixed proxy** in Burn Flex and
Candle, compare their forward and autodiff gradients with this reference,
then run matched steps and measure CPU throughput/peak memory. GPU support,
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

**Not yet delivered:** repeated matched-seed SGD/Adam training, CPU
throughput and peak-memory baselines, GPU viability, licensing decision
for any distributed binary, and ADR 0002 framework selection. The oracle
contains no licensed datasets or pre-trained weights.
