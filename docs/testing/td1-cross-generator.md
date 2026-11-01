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
IoU arithmetic and background errors. Each report also includes ordered\nper-image detection metadata (class, confidence and corners, but no RGB bytes),\nwhich the verifier replays to reject fabricated aggregate scores. Results are written as
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
