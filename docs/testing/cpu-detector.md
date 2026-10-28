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

## Batch-preparation comparison (PR #27 follow-up)

The optional Release-mode `fy-batch-benchmark --measure` compares the historical
per-step allocating batch implementation with the *same* `fill_training_batch`
function used by current training. It first verifies all 2,000 deterministic
batches and RNG state against the original algorithm, checking every feature
and label bit. It then alternates seven runs per implementation with identical
4,096-sample cached inputs, batch64 and sampler seed707, recording raw elapsed
nanoseconds, successful allocator requests and requested allocation bytes.
The benchmark uses a counting system allocator and black-boxed batch outputs.
Its raw JSON and command log are saved under the verification evidence directory
and collected by Linux/Windows experiment CI. Allocator counts are *requests*,
not peak RSS; elapsed times include measurement instrumentation and are subject
to shared-runner noise. The tool deliberately excludes forward/backward, optimizer,
model quality checks and other training work, so **it cannot establish a total
training speedup**. The unchanged 2,000-step Burn/Candle training and accuracy
gates still run separately. Results must be quoted with platform, exact commit
and raw repetitions; no synthetic-data result establishes natural-image quality.

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

The additive experiment workflow runs on Linux x64 and Windows x64. Each host
trains and consumes its own ephemeral outputs; the observed Linux numbers above
remain a Linux measurement, not a cross-host bitwise guarantee. Existing root CI on Linux/Windows
and local Rust/C/C# JIT/Native AOT, Miri and compiler-only i686 checks must also
pass. A clean final head, independent review, all available CI and zero unresolved
review threads are required before merging. Missing prerequisites or failed local
checks cannot be replaced with CI success. C/C# real-artifact integration belongs
to the separately reviewed following stage after this one merges.

The following integration stage adds an opt-in native path; see
[TD1 C/C# integration](td1-native.md). The M2a numbers above remain the original
training evidence and do not imply that default builds enable a model backend.

## Pre-registered small-object development validation (not final acceptance)

Before observing any model predictions on this split, the diagnostic was
registered with seed **20261028**, count **256**, generator
`TD1-synthetic-v1` and full source-independent fingerprint
`3f57fc1b8f9dcad73d40de2bbdd6b253e363e7206ee7a06ea454b155b90880a4`.
The sample generator is independently reimplemented with Python stdlib; the
historical seed909/128 fingerprint is checked to establish equivalence with
the frozen Rust generator. The new split contains **78 small positives**
(target area <=256 pixel²), **126 larger positives**, and **52 backgrounds**.
This grouping boundary was fixed before evaluating predictions, not chosen
after inspecting accuracy.

After unchanged 2,000-step training, verification runs the trained, reloaded
Burn and Candle artifacts on all 256 new images in separate portable inference
processes. It records per-size correct-class IoU>=0.5 recall, mean IoU
**including missed detections**, detection/class accuracy, signed x/y center
error (among detected objects), and background false positives. It preserves
the source revision, model artifact SHA-256 and frozen split fingerprint in
`prospective-validation-{burn,candle}.json` alongside the command logs.

These are **new development diagnostics**, not a replacement for validation
seed202 or final holdout seed909, and do not change training, model, public
ABI, the registered holdout pass/fail thresholds, or model selection. No
accuracy improvement is claimed without actual results; no pixels or weights
are uploaded. Python regressions detect generator drift, malformed detection,
missing boxes, false positives and artifact mutation. Rust/.NET/local source
verification and independent review remain separate gates.
