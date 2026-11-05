# TD2: first general-purpose detector for CCTV and edge devices (design draft)

Date: 2026-11-05. Status: **draft for owner review**. Direction set by the owner
in the planning conversation: the first real detector targets general-purpose
object categories; the intended deployment is CCTV cameras, NVRs and other edge
devices; the training-framework choice is deferred to stage B; TD1 stops
receiving new experiments. This document proposes the model, artifact, data,
evaluation and staging contract. It approves no dependency, dataset download,
compute spend, weight release or ABI change by itself; each is a gated stage
below.

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

## 2. Deployment target

FactsYOLO's first real users are video-analytics products on:

- **Edge devices:** aarch64 Linux boards and cameras with 4–8 Arm cores and
  1–8 GB RAM (Raspberry Pi 5 class as the reference board).
- **NVRs and small servers:** x64 Linux and Windows hosts running many camera
  streams.

What this implies for the design:

- Per-frame latency and steady-state memory matter more than peak accuracy.
- The host owns video decoding (RTSP/H.264/H.265); the core receives decoded
  frames. Cameras and decoders commonly emit YUV, not RGB.
- Scenes are fixed-camera, often elevated, with small distant objects, low light,
  infrared (grayscale) night mode, compression artefacts and motion blur.
- Frames arrive continuously, so no heap growth per frame and bounded worst-case
  memory are requirements, matching the existing M1 resource gates.

## 3. Goals and non-goals

Goals for the first TD2 release candidate:

- Multi-object, 80-class, axis-aligned box detection on decoded frames of
  arbitrary size, through the existing C ABI and C# adapter.
- Original architecture and training implementation; random initialization.
- Portable Rust CPU inference with no training framework in the deployed
  library, matching the TD1 split between `experiments/` and `crates/`, and
  SIMD paths for Arm NEON and x64 AVX2.
- Measured, pre-registered accuracy and per-frame latency/memory on named
  hardware, including a reference Arm board.

Non-goals for TD2:

- **Biometric or identity functions:** face recognition, person
  re-identification, licence-plate reading and similar. These carry privacy and
  legal obligations that need a separate decision; TD2 detects object classes
  only.
- Video decoding, streaming protocols, multi-frame tracking and event logic
  (hosts or a later stage own these).
- Segmentation, pose, oriented boxes, GPU/NPU inference, browser/WASM, Go/TS
  bindings, and comparison claims against other products.

INT8 quantization and NPU back ends are expected follow-ups for edge hardware
(section 9, stage G) but are not part of the first release candidate.

## 4. Native contract impact

The current ABI already carries most of what TD2 needs: arbitrary `width`/`height`
and row stride on `fy_image`, `max_pixels`/`max_detections` session limits,
`score_threshold` and cap on `fy_detect_options`, and results as clipped
original-image pixel edges with `class_id`, score ordering and `RESULT_TRUNCATED`.
Preprocessing and the inverse transform live in the Rust core, so bindings stay
thin.

Proposed additive changes, each behind its own decision record before
implementation:

1. `FY_BACKEND_TD2_CPU UINT32_C(2)` behind an opt-in `td2` cargo feature,
   reported through `fy_capabilities_copy`.
2. Camera-native pixel formats: `FY_GRAY8` (infrared/night frames), `FY_NV12`
   and `FY_I420` (decoder output). The existing descriptor can describe GRAY8.
   Planar YUV needs per-plane pointers and strides, which `fy_image` cannot
   express; this requires either a new image record version or a new detect
   operation, and is decided in stage E rather than here.

No existing struct, function or constant changes. `FY_ABI_MINOR` increments with
each additive change. The default build stays backend-free.

Out of scope here: exposing class names through the ABI. Class names ship in the
model card; a name-query operation needs its own decision.

## 5. Model TD2

All parts are original implementations of published techniques; no external
code or weights are ported. References are for the ideas only.

**Input.** Letterbox the frame to S×S, S a multiple of 32. The baseline is
S=320; 416 and 640 are evaluation variants of the same weights. Bilinear resize
preserving aspect ratio, centred, padded with value 0.5 after normalization to
[0,1]. Colour conversion (RGB/BGR/RGBA, and later GRAY/YUV) happens once in the
core. Grayscale input is replicated to three channels so one model serves day
and night modes.

**Network.** A small convolutional backbone (3×3 and depthwise-separable
convolutions, batch norm, SiLU, stride-2 downsampling, residual blocks), a
top-down plus bottom-up feature pyramid over strides 8/16/32, and a decoupled
head per level: one branch for 80 class logits plus objectness, one for four box
distances. Batch norm is folded into convolution weights at export, so inference
has no normalization layers. Budgets: **≤ 3.0 M parameters** (≤ 12 MB F32) and
**≤ 2.5 GFLOPs at S=320**. The exact layer table is fixed in stage D before
training starts and recorded in the artifact. Operators are limited to ones
that quantize well to INT8, so stage G does not need a different architecture.

**Box encoding.** Anchor-free: each feature location predicts non-negative
distances to the left/top/right/bottom edges in stride units, as in
[FCOS](https://arxiv.org/abs/1904.01355). Decoupled classification and
regression heads follow [YOLOX](https://arxiv.org/abs/2107.08430) in idea only.

**Training targets.** Centre sampling within 2.5 strides of the box centre plus
per-level scale ranges. Losses: sigmoid focal loss for classes, binary
cross-entropy for objectness, GIoU loss for boxes. Augmentation includes
horizontal flip, scale jitter, colour jitter, random grayscale, blur, JPEG-style
compression and low-light gain, to approximate camera conditions. The exact
assignment rule, loss weights, augmentation and schedule are fixed in stage D
and recorded, not tuned against the holdout.

**Decoding.** Score = sqrt(objectness × class probability). Drop candidates
below the caller's threshold, keep the pre-NMS top 1,000, run per-class NMS at
IoU 0.65, then apply the existing ABI sort and cap. Nonfinite or inverted
outputs are `FY_MODEL_OUTPUT_INVALID`, as today.

## 6. Artifact FYTDv002

A new, separate format; FYTDv001 is unchanged and stays TD1-only.

- Fixed header: magic, format version, input size S, class count, layer count,
  parameter count, total byte length.
- Layer table: kind, shapes, strides, padding, groups, activation, parameter
  offsets.
- F32 little-endian parameters, contiguous and aligned for SIMD loads.
- SHA-256 over the whole payload; `model_id` is that digest.
- Parsing obeys the existing `max_artifact_bytes` and `max_parser_steps`
  options and rejects unknown versions, shape mismatches and nonfinite values.

## 7. Data and licensing (gate C)

Training data must allow the project to publish weights. Evaluation data only
needs to allow local use.

| Use | Dataset | Annotation terms | Image terms | Handling |
| --- | --- | --- | --- | --- |
| Train | Open Images V7 subset mapped to the 80 COCO categories | CC BY 4.0 | Published as CC BY 2.0; not individually verified by the publisher | Keep only images with license metadata; generate an attribution list; never commit pixels |
| Evaluate (general) | COCO 2017 val | CC BY 4.0 | Flickr terms, mixed | Local evaluation only; never redistribute or upload pixels |
| Evaluate (camera domain) | To be selected in stage C | — | — | Must permit the intended use; candidates are recorded with their actual terms before any download |

Surveillance-style footage depicts identifiable people. Stage C records, for
every camera-domain source: its license, how subjects' privacy was handled, and
whether its terms permit commercial evaluation. A source with non-commercial or
unclear terms is excluded, not worked around.

Before any download, stage C records the actual terms, the category mapping and
the attribution procedure in `MODEL_LICENSES.md`, and the owner approves it.
Weight-release licensing is a separate decision at release time; this design
does not promise Apache-2.0 weights.

## 8. Evaluation (pre-registered)

Before the first TD2 training run, freeze and fingerprint:

- **Development split:** COCO val2017 image IDs in the lowest 1,000 by ID.
- **Final holdout:** the remaining 4,000 val2017 images. Evaluated once per
  candidate; never used for selection or tuning.
- **Camera-domain holdout:** frozen in stage C once a permitted source exists.

Metrics: COCO-style mAP50-95 and mAP50, per-size AP (small/medium/large), and
per-class AP reported separately for the classes CCTV products rely on most:
person, bicycle, car, motorcycle, bus, truck. Computed by an independently
implemented evaluator, cross-checked against a reference evaluator in
development only.

Proposed first-usable accuracy floor at S=320, owner-adjustable **before**
training:

- General holdout: mAP50 ≥ 0.30 and mAP50-95 ≥ 0.15.
- Person AP50 ≥ 0.50 on the general holdout.
- Untrained initial weights must score below 0.01 mAP50.
- Failure is recorded, not hidden; thresholds are not changed after results.

Per-frame budget, measured end-to-end through `fy_detect` on a 1280×720 RGB
frame at S=320, with CPU, OS, toolchain and commit recorded:

| Hardware | Threads | p50 latency | Peak RSS |
| --- | --- | --- | --- |
| Raspberry Pi 5 class (4× Cortex-A76) | 4 | ≤ 100 ms (≈10 FPS) | ≤ 128 MB |
| x64 desktop/NVR (AVX2) | 1 | ≤ 50 ms | ≤ 128 MB |

Steady state: zero heap growth across 10,000 consecutive frames, extending the
existing M1 resource gate. These are gates for stage E, not reporting targets.

## 9. Stages

| Stage | Deliverable | Exit |
| --- | --- | --- |
| A | This design | Owner approval |
| B | ADR 0002: training framework | Burn vs Candle compared on one fixed proxy: matched seeds, initialization and steps; gradient agreement with an independent reference forward; GPU training support; training throughput; peak memory; operator coverage (conv, depthwise conv, BN, SiLU, upsample); dependency licenses. One framework selected |
| C | Data tooling and license record | Section 7 recorded and approved; deterministic subset manifests with fingerprints; camera-domain source chosen or explicitly deferred; no pixels in git |
| D | Training and evaluation harness | Layer table frozen; training reproducible from manifests; independent evaluator; one full run reported against section 8 |
| E | Native `td2` backend | Portable Rust inference matches the training framework within tolerance; NEON and AVX2 paths; FYTDv002 parser; GRAY8 and YUV input decision; aarch64 Linux added to CI with real execution; C shared/static and C# JIT/AOT consumers on Linux x64/aarch64 and Windows x64; section 8 budgets met |
| F | First SDK release | One identically hashed C# package with Linux x64/aarch64 and Windows x64 assets; model card; release record |
| G | Edge acceleration (follow-up) | INT8 post-training quantization with a measured accuracy delta; optional NPU back ends evaluated separately |

Each stage is reviewed and merged before the next starts, per CONTRIBUTING.

## 10. Compute and cost

Training a detector of this size on an Open Images subset is likely impractical
on CPU alone. Stage B therefore records each framework's GPU path. Any paid
compute (cloud GPUs, hosted runners beyond the existing CI, or reference
hardware purchases) needs the owner's explicit approval with an estimated cost
before use.

## 11. Decisions requested from the owner

1. Approve TD1 maintenance-only status (section 1).
2. Confirm the deployment targets and reference hardware in section 2.
3. Confirm the biometric/identity non-goal in section 3.
4. Approve the dataset roles in section 7 so stage C can research and record terms.
5. Accept or adjust the accuracy floor and per-frame budgets in section 8.
6. Confirm whether paid GPU compute and a reference Arm board are acceptable,
   and the budget.

AI-assisted draft; no human certification or DCO sign-off is asserted on another
person's behalf.
