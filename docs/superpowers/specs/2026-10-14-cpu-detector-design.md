# CPU trainable detector evaluation

Date: 2026-10-14. Scope authorized by the owner's explicit agreement to an original
small detector, original synthetic data, CPU training/save/reload/inference, and
instruction to write the specification and implement the nearest independent stage.
This document records implementation decisions within that scope; it does not
select a production framework, publish weights, or change the native ABI.

## Deliverable and sequencing

M2a delivers a reproducible, offline CPU F32 training experiment for the same
original single-stage, single-object detector using Burn Flex and Candle. Both
must execute real tensor forward/backward operations and optimize random initial
parameters. A portable fixed-network reference runs saved parameters independently.
M2a is reviewed and merged before M2b integrates a real artifact through the existing
C/C# contract. The model deliberately predicts at most one axis-aligned object per
image; it is not a general-purpose multi-object YOLO implementation or accuracy claim.

Use an independent workspace under `experiments/cpu-detector/`, with pinned
Rust 1.99 and its own lockfile. The root workspace and default library retain
Rust 1.85, existing exports, no executable normal backend, and all M1 checks.
The portable model crate is also tested with Rust 1.85. Framework dependencies
belong only to evaluation binaries. No training runtime enters native deployment.

## Model TD1

Input is exactly 64x64 RGB bytes. Preprocessing computes a mean per channel for
each nonoverlapping 8x8 pixel block, divided by 255, in row/column/channel order.
The resulting 192 F32 features feed `linear(192,64) -> ReLU -> linear(64,7)`.
Outputs 0..3 are background/class0/class1 logits. Outputs 3..7 are sigmoid
normalized center-x, center-y, width and height. Decode to pixel-edge corners,
clip to the image, and reject nonfinite outputs. Return no box when background
wins or the winning class probability is below 0.5; otherwise return one box.
This is an original small box-regression network; no external model implementation
or parameters are reused. One-pass box/class regression is the algorithmic idea
in [the original YOLO paper](https://arxiv.org/abs/1506.02640); this is not its architecture.

Parameter order is W1 [192,64] row-major, b1 [64], W2 [64,7], b2 [7]: 12,807 F32.
The PRNG uses a wrapping u64 LCG (multiplier6364136223846793005,
increment1442695040888963407), returning the high32 bits. Uniform F32 uses its
high24 bits divided by2^24; Xavier bound is sqrt(6/(fan_in+fan_out)), biases zero.
Initialization seed404; dataset seeds101/202/303 for train/validation/test.
Batch sampling uses seed707 with replacement.
This deterministic PRNG supplies uniform Xavier weights and zero biases.
Frameworks receive identical parameter bytes; their own random initialization is
not used. The scalar reference is specific to TD1, not a new tensor framework.

## Original data and training

Generate images programmatically; no photos, external dataset or checkpoint.
Each sample is either background (one in five) or one filled rectangular object,
class0 red-dominant / class1 blue-dominant. Background and object intensities vary
with the seeded generator. Object top-left coordinates are 2..41 inclusive and
width/height 12..22 inclusive, so every box is inside the image. Noise must not
encode coordinates, classes, sample index or split identity outside the image.
Labels are geometry and class chosen before rasterization. Train/validation/test
use distinct fixed seeds and 512/128/128 samples, respectively. Split fingerprints
and generator version are recorded; tests verify no pixel-identical overlap.

Train 2,000 steps, batch64, deterministic sampling, Adam lr0.003, beta1=0.9,
beta2=0.999, epsilon1e-8. Loss is mean softmax cross-entropy over three classes plus
5 times mean squared sigmoid box error over four coordinates, masked to positives
and divided by total batch size. Its analytically derived output gradient is
finite-difference tested. Each tensor backend computes the vector-Jacobian product
of its real network with that gradient. The common host Adam update receives real
parameter gradients. This keeps loss and optimizer identical across frameworks.

No early-stop/checkpoint selection on test samples. Record initial/final train
loss, validation metrics, final held-out metrics, maximum gradient/reference errors,
training duration and raw sampled loss history. Both candidates run the fixed
budget; no claiming an interface call is successful detection. Corrupt labels or
frozen weights must not accidentally satisfy the acceptance conditions.

## Predeclared acceptance

- Both candidates: final train loss <=25% of initial train loss.
- Held-out positive class accuracy >=95%; positive recall at correct class and
  IoU>=0.5 >=85%; mean positive IoU >=0.70. Background false-positive rate <=5%.
  Evaluation decodes actual predictions; a missing detection has IoU0.
- Fixed CPU F32 inference/reference error: abs<=1e-4 + 1e-4*abs(reference).
- Directional/selected parameter finite differences: abs<=2e-3 + 2e-2*abs(reference),
  perturbation1e-3, excluding ReLU kink cases. Compare full backend gradients on
  the same non-kink batch with abs<=2e-5 + 3e-4*abs(reference).
- Save/reload preserves every parameter bit and reference prediction exactly.
  Reloaded candidate inference meets the same reference tolerance.
- Training execution <=600s per candidate on recorded Linux x64 host; compilation
  excluded. Record thread settings and hardware, not a product latency promise.
- Exact final source: root debug/release build/tests, C, C# JIT/AOT, Miri, i686,
  new portable-model Rust1.85 tests, candidate tests and full real training all pass.
  Linux CI runs the actual experiment; the existing Windows gates remain mandatory.

Budgets are declared before experiments. A failure is reported, not hidden by
changing seeds, lowering thresholds after seeing results, or replacing training
with hardcoded geometric detection. Any necessary spec revision is explicit.

## Experimental artifact TD1 v1

Data only, no executable graph/pickle, network access or external resolution.
Exactly 56+12,807*4 bytes: ASCII `FYTDv001` [8], architecture=1 LE u32,
parameter_count=12,807 LE u32, payload_bytes=51,228 LE u64, SHA-256 [32], then
LE F32 parameter payload. Digest covers the first24 bytes followed by payload;
it detects corruption, not authenticates a publisher. Reject unknown magic,
architecture/count/length, truncation, trailing bytes, checksum mismatch,
nonfinite or abs(weight)>10,000, and input image size mismatch. Limit reads to
64KiB before allocating from untrusted lengths. Serialize deterministic bytes.

Generated weights stay in ignored local/CI build outputs and are not committed,
uploaded as public artifacts or licensed for release by this experiment. Record
random-init/source/data provenance; public weight/data release remains a separate
explicit decision. Original code remains Apache-2.0; third-party license records
are retained. No Ultralytics code, weights, conversions or derived artifacts.

## Candidate adaptation and remaining boundary

Burn0.22.0/Flex/Autodiff declares Rust1.95: the experiment uses1.99 without raising
root or portable-model MSRV. Candle0.11.0 is the comparison candidate. Restrict
the network to matmul/add/ReLU, avoiding documented unsupported backward paths.
Pin direct/transitive dependencies and inventory exact licenses/native components.
No BLAS/GPU/JIT feature is enabled for this initial CPU comparison. Neither
candidate is selected for production merely by appearing in this experiment.

M2b may reuse the portable artifact/inference and existing admission/owned-result
logic, derive class metadata from the architecture, add capabilities only for real
available execution, and exercise the real model from C and C# JIT/AOT. It begins
only after M2a passes its complete review/merge gate.
