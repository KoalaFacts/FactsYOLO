# TD2: first general-purpose detector (design draft)

Date: 2026-11-05. Status: **draft for owner review**. Direction set by the owner
in the planning conversation: the first real detector targets general-purpose
object categories; the training-framework choice is deferred to stage B; TD1
stops receiving new experiments. This document proposes the model, artifact,
data, evaluation and staging contract. It approves no dependency, dataset
download, compute spend, weight release or ABI change by itself; each is a
gated stage below.

## 1. Why TD1 is frozen

TD1 compresses a 64×64 image into 192 tile means. The
[spatial-alias gate](../../testing/td1-spatial-alias.md) proves two distinct
single-object images can produce bit-identical features, so no training change
can recover the lost location information. TD1 also has a single object, two
classes and a fixed input size. Further synthetic diagnostics would keep
re-measuring the same structural ceiling.

From this document onward TD1 is in **maintenance only**:

- Keep: the `td1` feature, FYTDv001, the [v1 compatibility gate](../../testing/td1-v1-compatibility.md),
  the experiment workflow and its C/C# real-artifact consumers, as end-to-end
  regression coverage for train → artifact → C ABI → C# JIT/AOT.
- Accept: security and toolchain patches, CI repairs, and fixes to existing
  contracts.
- Do not add: new TD1 training arms, cohorts, diagnostics or tuning.

## 2. Goals and non-goals

Goals for the first TD2 release candidate:

- Multi-object, 80-class, axis-aligned box detection on natural RGB images of
  arbitrary size, through the **existing** C ABI and C# adapter.
- Original architecture and training implementation; random initialization.
- Portable Rust CPU F32 inference with no training framework in the deployed
  library, matching the TD1 split between `experiments/` and `crates/`.
- Measured, pre-registered accuracy and CPU latency/memory on a named machine.

Non-goals: segmentation, pose, tracking, oriented boxes, GPU inference, quantized
inference, browser/WASM, Go/TS bindings, and any comparison claim against other
products. Each needs its own decision after TD2 exists.

## 3. Native contract impact

The current ABI already carries what TD2 needs: arbitrary `width`/`height` and
row stride on `fy_image`, `max_pixels`/`max_detections` session limits,
`score_threshold` and cap on `fy_detect_options`, and results as clipped
original-image pixel edges with `class_id`, score ordering and `RESULT_TRUNCATED`.
Preprocessing and the inverse transform live in the Rust core, so bindings stay
thin.

Proposed additive change only: a new backend constant
`FY_BACKEND_TD2_CPU UINT32_C(2)` behind an opt-in `td2` cargo feature, reported
through `fy_capabilities_copy`. No struct, function or existing constant changes;
`FY_ABI_MINOR` increments for the new constant. The default build stays
backend-free.

Out of scope here: exposing class names through the ABI. Class names ship in the
model card; a name-query operation needs its own decision.

## 4. Model TD2

All parts are original implementations of published techniques; no external
code or weights are ported. References are for the ideas only.

**Input.** Letterbox the source to S×S, S a multiple of 32. The baseline is
S=320; 416 and 640 are evaluation variants of the same weights. Bilinear resize
preserving aspect ratio, centred, padded with value 0.5 after normalization to
[0,1]. Conversion from RGB/BGR/RGBA happens once in the core.

**Network.** A small convolutional backbone (3×3 convolutions, batch norm,
SiLU, stride-2 downsampling, residual blocks), a top-down plus bottom-up feature
pyramid over strides 8/16/32, and a decoupled head per level: one branch for 80
class logits plus objectness, one for four box distances. Batch norm is folded
into convolution weights at export, so inference has no normalization layers.
Parameter budget **≤ 3.0 M** (≤ 12 MB F32). The exact layer table is fixed in
stage D before training starts and recorded in the artifact.

**Box encoding.** Anchor-free: each feature location predicts non-negative
distances to the left/top/right/bottom edges in stride units, as in
[FCOS](https://arxiv.org/abs/1904.01355). Decoupled classification and
regression heads follow [YOLOX](https://arxiv.org/abs/2107.08430) in idea only.

**Training targets.** Centre sampling within 2.5 strides of the box centre plus
per-level scale ranges. Losses: sigmoid focal loss for classes, binary
cross-entropy for objectness, GIoU loss for boxes. The exact assignment rule,
loss weights, augmentation (horizontal flip, scale jitter, colour jitter) and
schedule are fixed in stage D and recorded, not tuned against the holdout.

**Decoding.** Score = sqrt(objectness × class probability). Drop candidates
below the caller's threshold, keep the pre-NMS top 1,000, run per-class NMS at
IoU 0.65, then apply the existing ABI sort and cap. Nonfinite or inverted
outputs are `FY_MODEL_OUTPUT_INVALID`, as today.

## 5. Artifact FYTDv002

A new, separate format; FYTDv001 is unchanged and stays TD1-only.

- Fixed header: magic, format version, input size S, class count, layer count,
  parameter count, total byte length.
- Layer table: kind, shapes, strides, padding, activation, parameter offsets.
- F32 little-endian parameters, contiguous.
- SHA-256 over the whole payload; `model_id` is that digest.
- Parsing obeys the existing `max_artifact_bytes` and `max_parser_steps`
  options and rejects unknown versions, shape mismatches and nonfinite values.

## 6. Data and licensing (gate C)

Training data must allow the project to publish weights. Evaluation data only
needs to allow local use.

| Use | Dataset | Annotation terms | Image terms | Handling |
| --- | --- | --- | --- | --- |
| Train | Open Images V7 subset mapped to the 80 COCO categories | CC BY 4.0 | Published as CC BY 2.0; not individually verified by the publisher | Keep only images with license metadata; generate an attribution list; never commit pixels |
| Evaluate | COCO 2017 val | CC BY 4.0 | Flickr terms, mixed | Local evaluation only; never redistribute or upload pixels |

Before any download, stage C records the actual terms, the category mapping,
and the attribution procedure in `MODEL_LICENSES.md`, and the owner approves it.
Weight-release licensing is a separate decision at release time; this design
does not promise Apache-2.0 weights.

## 7. Evaluation (pre-registered)

Before the first TD2 training run, freeze and fingerprint:

- **Development split:** COCO val2017 image IDs in the lowest 1,000 by ID.
- **Final holdout:** the remaining 4,000 val2017 images. Evaluated once per
  candidate; never used for selection or tuning.

Metrics: COCO-style mAP50-95 and mAP50, plus per-size AP (small/medium/large),
computed by an independently implemented evaluator, cross-checked against a
reference evaluator in development only.

Proposed first-usable floor at S=320, owner-adjustable **before** training:

- Holdout mAP50 ≥ 0.30 and mAP50-95 ≥ 0.15.
- Untrained initial weights must score below 0.01 mAP50.
- Failure is recorded, not hidden; thresholds are not changed after results.

Inference budget on Linux x64, single thread, S=320, measured end-to-end through
`fy_detect`: p50 latency ≤ 250 ms and peak RSS ≤ 200 MB. The record names the
CPU, toolchain and commit. Initially this is a reported target; it becomes a
gate once a SIMD path is in scope.

## 8. Stages

| Stage | Deliverable | Exit |
| --- | --- | --- |
| A | This design | Owner approval |
| B | ADR 0002: training framework | Burn vs Candle compared on one fixed proxy: matched seeds, initialization and steps; gradient agreement with an independent reference forward; CPU and GPU support; training throughput; peak memory; operator coverage (conv, BN, SiLU, upsample); dependency licenses. One framework selected |
| C | Data tooling and license record | Section 6 recorded and approved; deterministic subset manifests with fingerprints; no pixels in git |
| D | Training and evaluation harness | Layer table frozen; training reproducible from manifests; independent evaluator; one full run reported against section 7 |
| E | Native `td2` backend | Portable Rust inference matches the training framework within tolerance; FYTDv002 parser; C shared/static and C# JIT/AOT consumers on Linux and Windows |
| F | First SDK release | One identically hashed C# package for Linux/Windows assets; model card; release record |

Each stage is reviewed and merged before the next starts, per CONTRIBUTING.

## 9. Compute and cost

Training a detector of this size on an Open Images subset is likely impractical
on CPU alone. Stage B therefore records each framework's GPU path. Any paid
compute (cloud GPUs or hosted runners beyond the existing CI) needs the owner's
explicit approval with an estimated cost before use.

## 10. Decisions requested from the owner

1. Approve TD1 maintenance-only status (section 1).
2. Approve the dataset roles in section 6 so stage C can research and record terms.
3. Accept or adjust the accuracy floor and inference budget in section 7.
4. Confirm whether paid GPU compute is acceptable for stage D, and the budget.

AI-assisted draft; no human certification or DCO sign-off is asserted on another
person's behalf.
