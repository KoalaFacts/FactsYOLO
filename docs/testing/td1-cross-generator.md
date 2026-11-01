# TD1 independent-generator shift evaluation (experimental)

Frozen **2026-11-01 before any inference on this cohort**. This checks whether
the *existing*, unchanged 2,000-step TD1 Burn/Candle trained artifacts retain
correct-class box IoU >= 0.5 on images **not produced by TD1-synthetic-v1**.
It is not a natural-image benchmark, a new final holdout, a model-selection
dataset or authorization to ship an altered sampler. All generated pixels are
ephemeral. No weights, images, or third-party data are committed or uploaded.

## Generator / cohort

Source: original project-owned Python stdlib implementation in
`experiments/cpu-detector/cross_generator_validation.py`; no imported model
code, training image generator, external images or datasets. Exact constants:

- Domain: `TD1-cross-generator-rectangles-v1`; seed `20261101`.
- Each image independently seeds XorShift32 from the first four little-endian
  bytes of SHA-256(`TD1-cross-generator-rectangles-v1:20261101:<index>`).
  Each XorShift32 draw is a 32-bit (13,17,5) shift, reduced by modulo range.
- Output: 64x64 interleaved RGB. Four background patterns (random dark
  textured, checker bands, horizontal/vertical gradient-like waves) are
  constructed per pixel, unlike the prior flat 0..11 RGB noise.
- 96 positives: 12 each of classes 0/1 at left/top/right/bottom, ordered
  by side and class. Both target channels retain the same class semantics:
  channel 0 strong for class 0, channel 2 strong for class 1; each pixel in
  the box varies independently. Target widths/heights vary 11..23 px.
  Position on the designated edge is within 0..2 px; the other axis is
  sampled independently at >=4 px clearance. Right and bottom boxes are
  *not* reflected copies from the other groups.
- 24 backgrounds, no red/blue target rectangle. Total **120** examples,
  source indices 0..119. Every RGB array is 12,288 bytes and all 120 are
  unique. Backgrounds have no target; they test false positives.

The hash input is ASCII `TD1-cross-generator-rectangles-v1`, then for every
source row in index order a single class byte (255 for background), four
little-endian F32 normalized XYWH values (zeros for background), and all
12,288 RGB bytes. Computed before any model predictions, frozen SHA-256:

`41a28eaf9eede38fc1439a7dff1a0b976dfac05bffda19c556a6c7f873ec9f5f`

## Execution, evidence and interpretation

The existing `verify.py` runs the new cohort once per **real trained and
reloaded** Burn/Candle model, through fresh independent `fy-cpu-evaluate
infer` child processes. It checks the unchanged trained-artifact provenance,
registered legacy seed909 fingerprint (without evaluating seed909 here), the
independent cohort fingerprint, immutability of model bytes, every reported
class/side population, correct-class hits, missing/wrong-class counts,
IoU arithmetic and background errors. Each report also includes ordered
per-image detection metadata (class, confidence and corners, but no RGB bytes),
which the verifier replays to reject fabricated aggregate scores. Results are
written as
`cross-generator-{burn,candle}.json` in the Linux and Windows experimental
CI evidence artifacts, with exact source revision and model SHA-256. All
negative and mixed accuracy outcomes remain recorded.

This evaluation has **no minimum accuracy threshold** and cannot be selected
or modified after reading predictions. Failure of model generalization is
not disguised as a CI failure; only invalid evidence or failed pre-existing
acceptance gates make verification fail. The unchanged final seed909/128
holdout is never used to select a treatment. Do not infer that these synthetic
rectangles resemble natural photos or establish production readiness.

A low or uneven result can motivate a separately designed, preregistered
training intervention using independent-source distributions. Do not
retroactively train on this frozen cohort and still claim it is an unseen
test; any later training intervention needs a new evaluation source.

## Opt-in illumination-training follow-up (experimental, not evaluated here)

The previously observed 13/96 cross-generator correct-class IoU>=0.5
result and the centre-location error decomposition motivate an additional,
**controlled training arm**. It does **not** re-label the observed 120-image
cohort as unseen, alter the accepted seed909 final holdout, or establish
production accuracy.

The standalone `fy-edge-training <burn|candle> --texture` arm starts with
exactly the same seed404 weights, seed101/4096 original training examples,
seed707 sampled edge-positive indices, four reserved mirror slots, 2,000
steps, batch64 and Adam/loss as the established `--mirror` arm. Only the
cached 8x8 RGB block-mean training features are illuminated after the mirror.
For each step and slot, phase `(step + slot) % 4` chooses:

- identity (unchanged input);
- checkerboard: lift 0.15 on even-parity feature blocks, 0.08 otherwise;
- horizontal: lift `0.07 + 0.02 * block_x`;
- vertical: lift `0.07 + 0.02 * block_y`.

Each channel becomes `v + (1-v)*lift`. This bounded common-channel
transformation preserves relative red/blue dominance; it uses neither
label geometry nor additional PRNG draws, and cannot alter target boxes or
saved model format. It approximates photometric changes **at TD1's feature
resolution**; it does not reproduce the pixels of natural photographs or
the cross-generator program. The all-slot identity fraction is 1/4.
Full-batch validation precedes mutation, including a corrupt last feature.

The existing CI runs real training for both Burn and Candle on Linux and
Windows, preserving raw `texture-training-{burn,candle}.json` reports.
The unchanged seed202 validation gates must still pass; the registered
development/prospective/four-side datasets report gains **or losses**, not
a post-hoc success criterion. The training arm produces no weights for
publication and does not read the seed909 final holdout. No improvement
claim is permitted until those actual runs pass and a separately frozen,
previously unseen cross-generator distribution evaluates the candidate
against the unmodified mirror strategy on equal conditions.


## Opt-in center-gradient controlled comparison (2026-11-02, experimental)

The independent generator previously exposed **13/96** correct-class
IoU>=0.5 detections and substantial box-*center* displacement. A later
feature-only illumination arm made the registered development scores worse.
Before running a new center intervention, the following comparison is fixed:

- `fy-edge-training <burn|candle> --center` shares the existing `--mirror`
  initialization (404), source images (101, 4,096), sampler stream (707),
  four mirror slots/orientations, model (192->64->7), Adam, batch 64, and
  2,000-step training budget, with no new random draws.
- Only the **positive-label center-x/center-y gradients** (TD1 output logits
  3 and 4) are multiplied by exactly **2.0**, *after* the unchanged,
  validated loss gradient is computed. Class logits, width/height gradients,
  background gradients and all labels remain unchanged. This is equivalent
  to doubling the center terms of the box-regression objective. A whole-batch
  preflight rejects malformed, nonfinite or overflowing gradient buffers
  without partially changing them.
- Linux/Windows CI executes the *real Burn/Candle forward/backward and full
  budget* for mirror, illumination and center interventions. Each center
  report records `center_gradient_multiplier: 2.0`, train duration, source
  commit, model/seed/budget provenance, original seed202 validation and the
  already-known development, prospective, paired and disjoint side probes.
  The verifier requires the same source fingerprints and mirror sampling
  metadata and fails closed if an arm is mislabeled. Raw JSON evidence is
  preserved without uploading model weights or generated pixels.
- Existing seed202 thresholds are unchanged and mandatory for the experimental
  arm. The other scores are **diagnostics already observed in previous
  experiments**, including the disjoint four-side source; they are not new
  unseen accuracy evidence, cannot be repurposed as a model-selection test and
  must be reported even if the center arm loses. The reserved seed909 final
  holdout is not accessed by this arm.
- The center multiplier is a fixed *hypothesis*, not a tuned parameter or
  approved production default. This comparison can falsify that hypothesis;
  no improvement percentage is claimed before actual measurements. New
  cross-generator generalization or promotion requires an independently
  frozen, previously unevaluated source and a separate review. Public model
  format, ABI, library defaults and dependency policy are untouched.


## Frozen second-generator blind paired training comparison (2026-11-02)

This **new evaluation source was committed with a fixed fingerprint before
running inference from either training arm**. The previous 120-image
cross-generator set, four-side geometry pools, and development/validation
splits have already been observed. They are not reused as a blind metric.

- Source implementation: `experiments/cpu-detector/blind_center_comparison.py`,
  project-original Python stdlib only, domain `TD1-independent-striped-boxes-v2`,
  fixed seed `20261102`.
- Samples: 96 new positive RGB images (12 for each class at each left/top/right/
  bottom side; edge inset 0..3 pixels; sizes 11..23), plus 24 new backgrounds.
  Each 64×64 RGB image has a coordinate-varying neutral striped background and
  nonflat red- or blue-dominant target. No photos or external code/weights.
- Each image's XorShift32 state comes from the first four LE bytes of
  SHA-256(`TD1-independent-striped-boxes-v2:20261102:<index>`).
  Per-pixel background and target draws are deterministic, with no model
  outputs involved. The fingerprint hashes the domain, per-row class byte,
  normalized XYWH LE F32 labels, and 12,288 RGB bytes in row order:
  `aed8aaef623c07588159e3319ce59acdd02c0e41ad1622abbaf47a315c2f3ab2`.
- **Arms frozen before prediction**: TD1 mirrored-edge sampling vs exactly the
  same sampler with positive center-x/y gradients doubled. Both use seed404
  initial weights, seed101/4,096 training data, seed707 sampling, four fixed
  mirror slots, 2,000 steps, batch64, Adam and unchanged original thresholds.
  No data-based threshold, checkpoint or treatment choice is added.
- For each Burn/Candle backend on Linux/Windows, the existing experiment
  optionally saves both models to separate ephemeral ignored `*.fytd` files
  **only after** original validation succeeds; files are created without
  overwriting existing files and byte-reloaded for exact parameter equality.
  They are fed through fresh `fy-cpu-evaluate infer` child processes on the
  same 120 samples in the same order. The saved model byte SHA-256 values
  must match the corresponding training evidence and remain unchanged.
- JSON artifact `blind-center-{burn,candle}.json` preserves ordered raw
  per-image detections, model and source SHA-256 identities, group counts,
  hits at correct-class IoU≥0.5, mean IoU including misses, mean pixel-space
  center error for correct-class detections, missing/wrong-class counts,
  background false positives, and the paired hit delta. The verifier replays
  every aggregate from the frozen inputs and raw predictions; tampering,
  missing/reordered detections and invalid provenance fail closed.
- **No trained weights or RGB bytes are uploaded by CI**: evidence upload
  patterns include JSON/logs but exclude `*.fytd` files and raw images.

This is a *fresh synthetic* blind diagnostic, not the seed909 final holdout,
not natural-image performance, and not permission to declare the center-weighted
treatment a default. All positive and negative results are retained. The
prediction difference on one frozen cohort is exploratory; decisions about
architecture, broader distributions or shipping require separate design and
validation. The model format, public ABI and training seed909 test remain
unchanged.
