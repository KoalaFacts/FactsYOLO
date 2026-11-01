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

### Development localization error decomposition

The same frozen 256-image development run additionally emits
`localization_diagnostic` in each prospective JSON report. For the 78 small
and 126 larger targets separately, it counts missing boxes, wrong-class
detections, and correctly classified detections; only the last category
contributes absolute center/width/height error statistics. For each
correct-class detection it independently calculates hypothetical IoU after
**aligning the predicted box center to the target** (keeping predicted size)
or **matching the target width/height** (keeping predicted center). The report
lists indices and both hypothetical IoUs for correct-class predictions below
IoU 0.5. It never publishes pixels or model weights.

The counterfactual boxes are unclipped analytical rectangles, not actual model
outputs. Their effects can interact, and neither correction estimates a
realized training/inference performance gain or proves causal attribution.
Signed center averages in the original summary can cancel across samples;
the absolute errors here avoid that ambiguity. The 128-image final holdout is
**not** reanalyzed or used for tuning. There is no new accuracy threshold,
architecture change, training schedule, dataset or public ABI change.

### Prospective edge-vs-interior validation (separate development sample)

The earlier 256-image development split suggested that the four small-target
IoU50 misses occurred near the image border. That post-hoc finding is a hypothesis,
not evidence of a general edge effect. **Before any predictions on this new split**,
the follow-up registered generator `TD1-synthetic-v1`, seed **20261030** and
candidate budget **4,096**. Samples are drawn in source index order using only
*ground-truth* geometry; take the first **48** in each of five strata:
small edge, small interior, larger edge, larger interior and background (240
inference images). The small-area cutoff stays **256 px²**. Edge is defined as
a ground-truth box within **2 pixels** of any 64×64 image side. The selected
rows have frozen generator fingerprint
`04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4`.
The new pixels are independently checked for non-overlap with training seed101,
validation seed202, final holdout seed909 and the previous development seed20261028.

`edge_validation.py` uses the same **unchanged**, already accepted Burn and
Candle 2,000-step trained artifacts, checks their training provenance and their
SHA-256 before/after inference, then invokes the independent portable infer
process for each selected image. Verification writes
`edge-validation-{burn,candle}.json`, preserving source commit, model hash,
split fingerprint, group counts, missing/wrong-class counts, correct-class
IoU≥0.5 recall, mean correct-class IoU (misses count as zero), absolute center
errors, low-IoU development sample indices and edge-minus-interior recall gaps.
The 48-per-stratum quotas intentionally oversample the rare edge cases: these
ratios are **conditional group diagnostics**, not population prevalence or
natural-image accuracy. No causal explanation or accuracy improvement is claimed
without real predictions and further validation.

This addition changes no architecture, training budget, optimizer, test acceptance
threshold, public ABI or model format. The final 128-image holdout is not inspected
or tuned against. Python negative controls cover sample identity, all stratum
boundaries, wrong-class/missing penalties, invalid predictions, artifact mutation
and training provenance. These checks do not substitute for the root source,
Rust/C/.NET/Miri, CI or independent-review gates.


## Controlled TD1 small-edge training intervention (experimental only)

The edge development evidence in PR #40 motivates **one controlled training
intervention**, not another diagnostic-only reporting task. This experiment
does not change the TD1 model topology, the model artifact, decode logic, ABI,
existing training command, original final holdout, or its pass/fail thresholds.
It runs independently in the evaluation workspace as `fy-edge-training`.

Two arms use the **same** generated seed101/4096 training population, cached
192-feature inputs, seed404 initial parameters, seed707 sampler draw stream,
2000 steps, batch64, Adam, loss, and Burn/Candle backend. The original arm
calls the historical `fill_training_batch` unchanged. In the intervention,
exactly slots 0,16,32,48 of every batch remap their *existing* RNG draw into
the training samples that (a) contain a positive target, (b) have area at
most 256 square pixels, and (c) lie within 2 pixels of an image boundary.
All 60 remaining slots and all RNG advances retain the baseline algorithm.
The source training set, weights, number of gradient updates and optimizer
hyperparameters are not changed. This isolated intervention changes the
sampling distribution, intentionally, so training losses and model weights
are **not** expected to stay bit-identical between arms.

The comparison evaluates both arms against unchanged validation seed202/128
and the already-registered frozen PR #40 *development* edge split: seed
20261030, 4096 candidates, first48 per each of five groups, selected fingerprint
`04739d79ab725a8bef7cf0de7b8d1a8f6c929ed08e9e78b21f7392b81861b9c4`.
Selection uses ground truth only. The runner rejects source fingerprint drift,
empty edge training pool, unchanged weights, invalid predictions and baseline
failure of the original validation thresholds. It records both arms' validation
metrics, development IoU50 recalls, background errors, training time and
source revision for **each** backend, including negative or inconclusive outcomes.

**Predeclared decision rule, before the first intervention result:** only
consider the sampling change for further study if both backends pass the
unchanged validation thresholds, small-edge development recall improves over
uniform sampling, small-interior recall does not fall, and background false
positives do not increase. These are exploratory candidate-selection criteria,
not revised production acceptance gates. A failing comparison is evidence
against this intervention rather than permission to alter thresholds or
discard samples. Do not promote a candidate based on this already-observed
development split: a subsequent untouched, pre-registered independent
verification is required, with no final seed909 reuse for tuning.

The runner **never loads or evaluates final holdout seed909**. It produces
JSON metrics only; no images, weights or trained artifacts are uploaded.
This is a narrow controlled training experiment, **not** a shipped
edge-corrected detector or evidence of natural-image accuracy improvement.


### Untouched prospective evaluation of edge sampling (post-PR #41)

The controlled 4-of-64 edge sampler in PR #41 passed **already observed**
development-only candidate criteria; this does **not** justify adopting it.
Before running any predictions with either trained arm on the next split,
we fixed synthetic generator `TD1-synthetic-v1`, seed **20261101**,
candidate budget **4096**, and the **first 48 ground-truth examples** in each
of the same five fixed strata (small/larger by 256 px²; edge by <=2 px; plus
background). The **240** selected examples have fingerprint
`0239bac3429d484db479974c10e5a038cef8829762a7d5533f8cd915e3a6ede1`.
The independent Python replay reproduced the existing #40 fingerprint first,
then this prospective fingerprint and confirmed no repeated pixel identities
against seed101/4096 training, seed202/128 validation, seed909/128 final
holdout, seed20261028/256 diagnostics and seed20261030/240 development
selection. A Rust unit test and Python regression pin the new split. No
predictions were observed while selecting or registering it.

The opt-in `fy-edge-training` runner reuses its **same paired original and
four-reserved-slots training arms**, no new training parameters or
model-format/ABI change. It evaluates both newly trained arms once on the
unseen split and emits `baseline_prospective`,
`edge_sampling_prospective`, source identity and each stratum's IoU50
correct-class recall, missing/wrong-class counts and background errors.
Verification rejects fingerprint/provenance drift, invalid sample counts,
nonfinite/fabricated recall and mismatched paired deltas. **Negative
accuracy comparisons pass the evidence-recording gate**; they are not hidden
or transformed into a CI failure.

Predeclared interpretation: qualify the already selected edge sampler for
**further synthetic research only** if small-edge IoU50 recall improves on
this untouched split for both Burn/Candle and Linux/Windows, small-interior
recall does not fall, background errors do not rise and the unchanged
original validation still passes. Otherwise classify this experiment as
negative/inconclusive rather than changing the split, seed or thresholds.
This is one prospective synthetic evaluation, not statistical or
natural-image generalization proof, a production model decision, or
permission to tune on final holdout seed909.


### Opt-in mirrored reserved-slot training (experimental third arm)

The pre-registered PR #42 comparison **did not** confirm that simple 4/64
small-edge oversampling improves untouched synthetic recall: Linux was
44/48 -> 43/48; Windows was 44/48 -> 44/48 on both Burn and Candle.
Do **not** adopt the edge sampler on the strength of the earlier development
split. An independent surrogate-only exploration subsequently found that
the generator's x/y=2..41 distribution has essentially no right/bottom
small-object coverage; its Python/PyTorch mirror results were *not* native
Burn/Candle results or production evidence.

An opt-in **third-arm implementation**, invoked separately with
`fy-edge-training burn --mirror` or `fy-edge-training candle --mirror`,
now trains the same 192→64→7 model, initialized with seed404, using the same
seed101/4096 original source images, seed707 sampling draws, 2,000 steps,
batch64, Adam and loss as the prior two-arm experiment. It first selects
the **identical** four reserved edge-positive samples at batch slots
0/16/32/48; only their *feature/label representation* is altered:
slot0 identity, slot16 horizontal reflection, slot32 vertical reflection,
slot48 both axes. The 60 ordinary slots and full RNG sequence are unchanged.
The 192 block-average F32 features are permuted at the 8×8 grid level and
normalised x/y box centres mirrored; widths, heights, classes and number
of samples are preserved. There is no image or training-data upload, model
format change, ABI change, GPU dependency or automatic production rollout.

The common verifier now executes this third arm in both Burn and Candle on
Linux and Windows, checks the predeclared seed/slot/provenance contract,
validates each stratum's IoU and detection-count arithmetic and enforces
the **unchanged** original validation acceptance. It writes raw
`mirror-training-{burn,candle}.json` and command logs beside the existing
original/edge arms. A real RGB-pixel reflection regression proves feature
and box parity, including untouched slots, while a malformed-label
regression verifies all-or-nothing input rejection. Use the original CPU
experiment reproduction procedure above; no additional dependencies.

**Interpretation boundary:** both existing seed20261030 and seed20261101
stratified reports are already observed development diagnostics. Running
a new trained arm on them is exploratory; it does not turn either into an
unseen evaluation or retrospectively qualify plain edge oversampling. The
approved final holdout seed909 is **not loaded** by this arm. The old
generator still lacks independently sampled small targets near right/bottom
edges. No side-position accuracy or speed improvement is asserted until
a separate, prospective, four-sided source-independent Rust Burn/Candle
comparison is frozen and executed; any observed validation trade-off
must be reported, not hidden.

### Disjoint-source prospective four-side comparison (PR following #51)

The earlier four-sided mirror stress test reuses each source image four times,
so its promising reflected-target IoU50 differences are correlated rather than
four independent source cohorts. Before seeing any model predictions on the next
split, the [new registered evaluation](td1-four-side-disjoint.md) fixes
seed20261123/8,192 candidates and **144 distinct source images**: 24
independent positives per side and 48 untouched backgrounds. Original
right/bottom gaps are filled by reflecting *different* left/top source images,
not the same rows already used in the other side groups.

Existing Burn and Candle training arms, parameters, 2,000 steps, final
holdout isolation and original validation thresholds remain unchanged.
The opt-in edge runner evaluates original, edge-resampled and mirror-resampled
weights against this single new split. Each run records group hits, IoU,
wrong-class/missing counts, background errors and exact provenance, with
negative comparisons preserved. The verifier checks frozen fingerprint,
per-group arithmetic and a predeclared directional criterion; failure of
that *candidate-selection* criterion is a valid CI result, not a hidden
accuracy regression. Linux and Windows evidence must be read together to
establish whether the criterion was met on both hosts and backends.

This is an experiment on disjoint sources from the **same synthetic
generator**, not proof of real-image accuracy or authorization to deploy or
publish an altered TD1 sampler. The baseline paired diagnostic remains
available and is not reinterpreted as unseen evidence.
