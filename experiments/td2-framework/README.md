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
