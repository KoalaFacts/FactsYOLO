# TD1 CPU training experiment

This independent M2a workspace implements a real original single-object detector:
64x64 RGB -> block means ->192->64->7 network,12,807 trainable parameters. It
learns background/class and box coordinates using tensor autodiff in Burn Flex
and Candle, a shared Adam update and original synthetic data. The portable
Rust1.85 reference reads the trained artifact and predicts in a fresh process.
This stage does not yet connect the artifact to `fy_model_open` or C/C#.

The [design](../superpowers/specs/2026-10-14-cpu-detector-design.md) fixes the
architecture, optimizer, format, seeds and pass/fail budgets. The
[dependency review](../../experiments/cpu-detector/DEPENDENCIES.md) records exact
versions, native components and license scope. Root Rust1.85, the native ABI
and existing unsupported-model behavior are unchanged. Both workspaces disable
Cargo publication; CI uploads metrics/logs only, never pixels, weights or binaries.

## Reproduce

Use a Linux x64 host with Python3, C toolchain, rustup and Rust1.99/1.85 installed.
Fetch the locked crates once from the Cargo registry; subsequent gates are offline:

```sh
rustup toolchain install 1.99.0 --profile minimal --component rustfmt --component clippy
rustup toolchain install 1.85.0 --profile minimal
cargo fetch --manifest-path experiments/cpu-detector/Cargo.toml --locked
python3 experiments/cpu-detector/verify.py experiments/cpu-detector/target/evidence
```

The destination must be new. The script records exact commands/exit codes/source
commit/platform, runs formatting, warnings-as-errors clippy, Debug/Release tests,
portable Rust1.85 tests, release build, then actual2,000-step training for each
backend. All training runs use `RAYON_NUM_THREADS=1`. Compilation is excluded
from the600-second per-backend training budget. The CLI can be used directly:

```sh
RAYON_NUM_THREADS=1 experiments/cpu-detector/target/release/fy-cpu-evaluate train burn experiments/cpu-detector/target/my-run
experiments/cpu-detector/target/release/fy-cpu-evaluate infer experiments/cpu-detector/target/my-run/model.fytd experiments/cpu-detector/target/my-run/sample-1.rgb
```

Outputs include raw per-step `loss.tsv`, `metrics.json` with all128 held-out
predictions, a data-only51,284-byte artifact, and six RGB samples for separate
process inference. The verifier compares those fresh-process predictions with
the training runner's reloaded reference results, and rejects corrupt and
oversized artifacts. Unit/contract tests cover invalid images, wrong weights,
nonfinite parameters/gradients, artifact length/header/checksum limits, expected
geometry, finite-difference loss/parameter gradients and full backend agreement.
Changing serialized parameters changes independent-process output. Untrained
initial parameters must fail the detection gates; training must change weights.

## Observed local experiment and failed first configuration

On the2026-10-14 Linux x64 AMD EPYC9V74 environment, the initial512-image training
pool failed unchanged acceptance: both backends produced mean testIoU0.693798
(required>=0.70), validationIoU0.675745, despite trainIoU0.816506. Initial/final
training loss was1.276339/0.001295. Independent Python geometry reproduced the
reported testIoU. This result was reported as failure; it did not qualify for merge.

A documented revision increased only the training pool to4,096 images. The
network,2,000 steps, batch64, Adam, loss and thresholds stayed fixed. Validation
seed202 remained a development diagnostic; the observed test303 was retired.
Seed909/count128 was registered as the new final holdout *before* the next run.
There is no epoch selection or per-backend tuning against the test set.

The coverage-revised local experiment passed for both backends:

| Measurement | Burn Flex0.22.0 | Candle CPU0.11.0 |
| --- | --- | --- |
| Initial/final training loss | 1.273546 /0.00175149 | 1.273546 /0.00175149 |
| Validation meanIoU | 0.787100 | 0.787100 |
| Held-out positive class accuracy | 100% | 100% |
| Held-out correct-class IoU>=0.5 recall | 99.02% | 99.02% |
| Held-out meanIoU | 0.776612 | 0.776612 |
| Background false-positive rate | 0% | 0% |
| Maximum backend/reference absolute error | <=0.000004 | <=0.000004 |
| Exact saved/reloaded parameter and reference bits | pass | pass |

The final exact-head logs and CI artifacts are the source for elapsed times and
revision-specific verification; compilation/caching/concurrent jobs make this
small run unsuitable for framework speed ranking. These numbers describe only
original colored rectangles on the documented distribution,102 positive and26
background holdout images. They establish real train/save/reload/infer behavior,
not natural-image quality, multi-object detection, GPU support or production
latency. Neither candidate is selected for production by this comparison.

## Full-stage gate and next boundary

The additive experiment workflow runs on Linux. Existing root CI on Linux/Windows
and local Rust/C/C# JIT/Native AOT, Miri and compiler-only i686 checks must also
pass. A clean final head, independent review, all available CI and zero unresolved
review threads are required before merging. Missing prerequisites or failed local
checks cannot be replaced with CI success. C/C# real-artifact integration belongs
to the separately reviewed following stage after this one merges.
