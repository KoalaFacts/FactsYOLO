# TD2 and the CCTV vision task roadmap (design draft)

Date: 2026-11-05. Status: **accepted** (stage A complete; owner decisions in
section 12). Direction set by the owner in the planning conversation:

- FactsYOLO is embedded in CCTV cameras, NVRs and other edge devices.
- Customers install and operate it themselves, on premises.
- Required capabilities: general object and animal detection, scene
  recognition, licence-plate recognition and face recognition.
- The training-framework choice is deferred to stage B.
- TD1 stops receiving new experiments.

This document proposes the task family, the first model (TD2 detection), its
artifact, data, evaluation and staging contract, and the follow-up tasks that
build on it. It approves no dependency, dataset download, compute spend, weight
release or ABI change by itself; each is a gated stage below.

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

## 2. Deployment model

**Hardware.**

- Edge devices: aarch64 Linux boards and cameras with 4–8 Arm cores and 1–8 GB
  RAM (Raspberry Pi 5 class as the reference board).
- NVRs and small servers: x64 Linux and Windows hosts running many streams.

**Operation.** Customers install and run FactsYOLO inside their own products on
their own premises. FactsYOLO is a library: it has no cloud service, no
telemetry, and sends nothing off the device.

**Data stays under the customer's control.** No frame, crop, embedding, plate
text, result or diagnostic is ever written or sent anywhere the customer does
not control. The deployed library contains no networking code and opens no
files beyond the artifact bytes the host passes in. Stage E adds a CI gate that
fails if a deployed build gains a networking dependency or imports socket/HTTP
symbols. Training-side tooling likewise never uploads customer data; customers
fine-tuning on their own footage do so locally. The host owns video decoding
(RTSP/H.264/H.265), storage, retention, face galleries, plate watch-lists and
all business logic; the core receives decoded frames and returns results.

**What this implies.**

- Per-frame latency and steady-state memory matter more than peak accuracy.
- Cameras and decoders commonly emit YUV; night mode is infrared grayscale.
- Scenes are fixed-camera and often elevated: small distant objects, low light,
  compression artefacts, motion blur.
- Frames arrive continuously, so no per-frame heap growth and bounded
  worst-case memory, extending the existing M1 resource gates.

## 3. Task family

| Task | Output | Builds on | Stage |
| --- | --- | --- | --- |
| T1 Detection (TD2) | Boxes + classes: objects, animals, people, vehicles, **faces**, **licence plates** | — | A–G |
| T2 Plate recognition | Plate text + per-character confidence for a plate box | T1 plate boxes | H |
| T3 Scene recognition | Scene class scores for a whole frame | T1 backbone | I |
| T4 Face recognition | Face embedding vector for a face box; host compares embeddings | T1 face boxes | J |

T1 comes first because T2 and T4 both need its boxes, and T3 reuses its
backbone. The recommended order after T1 is plate recognition (synthetic
training data is feasible), then scene recognition, then face recognition, which
is bound by data licensing (section 8).

**Non-goals for every task:**

- Inferring age, gender, ethnicity, emotion or other personal attributes.
- Liveness and anti-spoofing (a separate decision if customers need it).
- Video decoding, streaming protocols, multi-frame tracking and event logic.
- Segmentation, pose, oriented boxes, browser/WASM, Go/TS bindings, and
  comparison claims against other products.

## 4. Customer obligations and product safeguards

Face recognition and plate recognition process personal information. In
Australia, biometric information used for verification or identification is
sensitive information under the Privacy Act; the Privacy Commissioner's 2024
determination against Bunnings, upheld in part on review in 2026, shows that
signage alone is not adequate notice and that even momentary capture counts as
collection. Comparable rules apply in other markets (for example, the EU AI Act
restricts remote biometric identification).

Because customers operate the system, **legal compliance in each deployment is
the customer's responsibility**. FactsYOLO's responsibility is to make a
compliant deployment possible and to source its own training data lawfully.
Product safeguards:

- T2 and T4 ship as separate opt-in cargo features and back ends, absent from
  the default build, and reported through capabilities.
- The core stores nothing: no images, embeddings, galleries or plate lists
  persist beyond a call's result.
- Embeddings are returned to the host only; the core performs no identity
  lookup against any stored list. A comparison helper takes two caller-owned
  embeddings and returns a similarity score.
- Deployer documentation lists the obligations to assess (notice, consent,
  retention, purpose limitation, privacy impact assessment) without giving
  legal advice.

## 5. Native contract impact

T1 fits the current ABI: arbitrary `width`/`height` and row stride on
`fy_image`, session limits, `score_threshold` and cap on `fy_detect_options`,
and results as clipped original-image pixel edges with `class_id`, score
ordering and `RESULT_TRUNCATED`. Preprocessing and the inverse transform live in
the Rust core, so bindings stay thin.

Proposed additive changes, each behind its own decision record before
implementation:

1. `FY_BACKEND_TD2_CPU UINT32_C(2)` behind an opt-in `td2` cargo feature.
2. Camera-native inputs: `FY_GRAY8`, `FY_NV12`, `FY_I420`. GRAY8 fits
   `fy_image`; planar YUV needs per-plane pointers and strides and therefore a
   new image record version or operation, decided in stage E.
3. T2–T4 need new result shapes. The header says fixed-stride arrays are
   extended only by new operations or types, so each task adds its own
   operation and owned result type (text, class scores, embedding), taking a
   source frame plus a box where applicable. Designed in each task's stage.

No existing struct, function or constant changes. `FY_ABI_MINOR` increments with
each additive change. The default build stays backend-free.

Class names ship in the model card; a name-query operation needs its own
decision.

## 6. Model TD2 (T1 detection)

All parts are original implementations of published techniques; no external
code or weights are ported. References are for the ideas only.

**Classes.** A curated set fixed in stage C, of at most 120 classes: the 80
COCO categories, plus `human face` and `vehicle registration plate`, plus
additional common animals and CCTV-relevant objects available in the Open Images
hierarchy. The mapping is recorded before training.

**Input.** Letterbox the frame to S×S, S a multiple of 32. The baseline is
S=320; 416 and 640 are evaluation variants of the same weights (larger sizes
help small faces and distant plates). Bilinear resize preserving aspect ratio,
centred, padded with value 0.5 after normalization to [0,1]. Colour conversion
happens once in the core. Grayscale input is replicated to three channels so
one model serves day and night modes.

**Network.** A small convolutional backbone (3×3 and depthwise-separable
convolutions, batch norm, SiLU, stride-2 downsampling, residual blocks), a
top-down plus bottom-up feature pyramid over strides 8/16/32, and a decoupled
head per level: one branch for class logits plus objectness, one for four box
distances. Batch norm is folded into convolution weights at export. Budgets:
**≤ 3.0 M parameters** (≤ 12 MB F32) and **≤ 2.5 GFLOPs at S=320**. The exact
layer table is fixed in stage D before training. Operators are limited to ones
that quantize well to INT8, and the backbone exposes its pyramid features so T3
can reuse them.

**Box encoding.** Anchor-free: each feature location predicts non-negative
distances to the left/top/right/bottom edges in stride units, as in
[FCOS](https://arxiv.org/abs/1904.01355). Decoupled classification and
regression heads follow [YOLOX](https://arxiv.org/abs/2107.08430) in idea only.

**Training targets.** Centre sampling within 2.5 strides of the box centre plus
per-level scale ranges. Losses: sigmoid focal loss for classes, binary
cross-entropy for objectness, GIoU loss for boxes. Augmentation includes
horizontal flip, scale jitter, colour jitter, random grayscale, blur, JPEG-style
compression and low-light gain. The exact assignment rule, loss weights,
augmentation and schedule are fixed in stage D and recorded, not tuned against
the holdout.

**Decoding.** Score = sqrt(objectness × class probability). Drop candidates
below the caller's threshold, keep the pre-NMS top 1,000, run per-class NMS at
IoU 0.65, then apply the existing ABI sort and cap. Nonfinite or inverted
outputs are `FY_MODEL_OUTPUT_INVALID`, as today.

## 7. Artifact FYTDv002

A new, separate format; FYTDv001 is unchanged and stays TD1-only.

- Fixed header: magic, format version, task kind, input size S, class count,
  layer count, parameter count, total byte length.
- Layer table: kind, shapes, strides, padding, groups, activation, parameter
  offsets.
- F32 little-endian parameters, contiguous and aligned for SIMD loads.
- SHA-256 over the whole payload; `model_id` is that digest.
- Parsing obeys the existing `max_artifact_bytes` and `max_parser_steps`
  options and rejects unknown versions, shape mismatches and nonfinite values.

The task-kind field lets T2–T4 models reuse the container with their own
layer kinds, each defined in its stage.

## 8. Data and licensing (gate C, then per task)

Training data must allow the project to publish weights for commercial use.
Evaluation data only needs to allow local commercial evaluation.

| Task | Use | Source | Notes |
| --- | --- | --- | --- |
| T1 | Train | Open Images V7 subset (includes faces, plates, animals) | Annotations CC BY 4.0; images published as CC BY 2.0, not individually verified by the publisher. Keep only images with license metadata; generate an attribution list |
| T1 | Evaluate | COCO 2017 val (COCO classes); Open Images V7 validation (faces, plates, extra classes) | Local only; never redistribute pixels |
| T1 | Evaluate (camera domain) | Selected in stage C | Must permit commercial evaluation |
| T2 | Train | Synthetic plates rendered from published AU/NZ plate formats, plus any real plate data with commercial terms | Fonts and layouts need their own licence check |
| T3 | Train / evaluate | Selected in stage I | Popular scene datasets are often non-commercial; excluded unless terms permit |
| T4 | Train / evaluate | Selected in stage J | Most public face-recognition datasets are research-only or have been withdrawn. Expect to license a commercial dataset or collect one with documented consent; cost needs owner approval |

Footage of people is personal information. For every source, record its licence,
how subjects' privacy and consent were handled, and whether its terms permit the
intended commercial use. A source with non-commercial or unclear terms is
excluded, not worked around. Before any download, the stage records the terms,
category mapping and attribution procedure in `MODEL_LICENSES.md`, and the
owner approves it. Weight-release licensing is a separate decision at release
time; this design does not promise Apache-2.0 weights.

## 9. T1 evaluation (pre-registered)

Before the first TD2 training run, freeze and fingerprint:

- **Development split:** COCO val2017 image IDs in the lowest 1,000 by ID, plus
  a fixed Open Images validation development subset.
- **Final holdouts:** the remaining 4,000 COCO val2017 images and a disjoint
  Open Images validation subset. Evaluated once per candidate; never used for
  selection or tuning.
- **Camera-domain holdout:** frozen in stage C once a permitted source exists.

Metrics: COCO-style mAP50-95 and mAP50, per-size AP (small/medium/large), and
per-class AP reported separately for person, face, licence plate, bicycle, car,
motorcycle, bus, truck and the animal classes. Computed by an independently
implemented evaluator, cross-checked against a reference evaluator in
development only.

The evaluator is `experiments/td2-eval` (std-only Rust, not linked into the
library). Object size ranges use box area, since TD2 data carries boxes only.
Its tests freeze 12 random scenes scored by pycocotools 2.0.11, which was run
once in development to produce them and is not a dependency.

Proposed first-usable accuracy floor at S=320, owner-adjustable **before**
training:

- COCO holdout: mAP50 ≥ 0.30 and mAP50-95 ≥ 0.15.
- Person AP50 ≥ 0.50; face AP50 ≥ 0.40; licence plate AP50 ≥ 0.40.
- Untrained initial weights must score below 0.01 mAP50.
- Failure is recorded, not hidden; thresholds are not changed after results.

Per-frame budget, measured end-to-end through `fy_detect` on a 1280×720 RGB
frame at S=320, with CPU, OS, toolchain and commit recorded:

| Hardware | Threads | p50 latency | Peak RSS |
| --- | --- | --- | --- |
| Raspberry Pi 5 class (4× Cortex-A76) | 4 | ≤ 100 ms (≈10 FPS) | ≤ 128 MB |
| x64 desktop/NVR (AVX2) | 1 | ≤ 50 ms | ≤ 128 MB |

Steady state: zero heap growth across 10,000 consecutive frames. These are
gates for stage E. T2–T4 define their own accuracy and latency gates in their
stages, before their first training run.

## 10. Stages

| Stage | Deliverable | Exit |
| --- | --- | --- |
| A | This design | Owner approval |
| B | ADR 0002: training framework | Burn vs Candle compared on one fixed proxy: matched seeds, initialization and steps; gradient agreement with an independent reference forward; GPU training support; throughput; peak memory; operator coverage (conv, depthwise conv, BN, SiLU, upsample); dependency licenses. One framework selected for all tasks |
| C | T1 data tooling and license record | Section 8 T1 rows recorded and approved; class set frozen; deterministic subset manifests with fingerprints; no pixels in git |
| D | T1 training and evaluation harness | Layer table frozen; reproducible training from manifests; independent evaluator; one full run reported against section 9 |
| E | Native `td2` backend | Portable Rust inference matches training within tolerance; NEON and AVX2 paths; FYTDv002 parser; GRAY8/YUV input decision; aarch64 Linux CI with real execution; C and C# JIT/AOT consumers on Linux x64/aarch64 and Windows x64; section 9 budgets met |
| F | First SDK release (detection) | One identically hashed C# package with Linux x64/aarch64 and Windows x64 assets; model card; deployer documentation; release record |
| G | Edge acceleration | INT8 post-training quantization with a measured accuracy delta; optional NPU back ends evaluated separately |
| H | T2 plate recognition | Design, data record, recognizer, text-result operation, AU/NZ formats first |
| I | T3 scene recognition | Design, permitted data, classification head on the shared backbone, class-score operation |
| J | T4 face recognition | Design, licensed or consented data, alignment and embedding model, embedding and comparison operations, safeguards from section 4 |

Each stage is reviewed and merged before the next starts, per CONTRIBUTING.
H–J may start their design and data work in parallel with E–G.

## 11. Compute and cost

**Current budget: zero.** Owner decision on 2026-11-05: there is no budget for
paid compute, hardware or datasets for now; this is revisited later. Until then
every stage uses only resources that cost nothing:

- GitHub-hosted runners, which are free for this public repository, including
  Linux arm64 runners. These serve as the interim aarch64 execution and
  measurement host; their CPU is named in each record. The Raspberry Pi 5-class
  latency gate in section 9 stays defined but is deferred until a board exists.
- The development cloud workspace used by automated development sessions
  (2 CPUs, no GPU), for short local gates and analysis only.

When a stage genuinely needs a GPU, the notebook or scripts are prepared in the
repository and the owner is asked to run them on a free service under their own
account; no account is created on the project's behalf.

Consequences:

- Stage B compares frameworks on CPU, at proxy scale, and records each
  framework's GPU path for later without using one.
- Stage D starts with a reduced-scale T1 run that fits the free runners
  (smaller class subset, image count and steps). Results are reported against
  section 9 honestly; a full-scale run waits for compute.
- T4 face-recognition training data cannot be licensed or collected at zero
  cost, so T4 is limited to design work until a budget exists.

Any paid compute, hardware or licensed data needs the owner's explicit approval
with an estimated cost before use.

## 12. Owner decisions

Recorded 2026-11-05:

- Task family and order T1 → T2 → T3 → T4 (section 3): approved.
- Budget: none for now; revisit later (section 11). Only GitHub-hosted runners
  (including `ubuntu-24.04-arm` as the interim aarch64 host) and the development
  workspace are used.
- Deployment: on premises, installed and operated by the customer. No data
  leaves the customer's control; the deployed library has no networking code,
  writes no files and stores nothing (section 2).
- Face recognition (T4) and licence-plate recognition (T2) are required; scene,
  animal and object recognition are very important.
- Training framework: decided by stage B evidence and recorded in ADR 0002.
- TD1 maintenance-only status (section 1), the deployment model and reference
  hardware (section 2), the safeguards (section 4), the T1 dataset roles
  (section 8) and the accuracy floors and per-frame budgets (section 9) are
  adopted as written. If later evidence shows one of them is wrong, a pull
  request changes this document and states why.

Still the owner's, per stage:

- Approving the recorded licence terms of each external dataset before it is
  downloaded for training (stage C onward).
- Any spending or account, publishing weights or SDK packages, tags and
  releases, and licence or trademark changes.

Work items, status and evidence for every stage are tracked in a roadmap
ledger, `docs/plans/td2-roadmap.md`, added by the first pull request after this
design is merged.

AI-assisted draft; no human certification or DCO sign-off is asserted on another
person's behalf. Section 4 is a product-design summary, not legal advice.
