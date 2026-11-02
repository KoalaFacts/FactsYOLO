# Frozen independent background comparison (experimental TD1)

This **diagnostic** follows the negative/inconclusive [background-gradient
training experiment](cpu-detector.md). The original seed202 validation passed
for the opt-in background arm but its already-observed synthetic IoU fell;
that report did **not** evaluate the earlier cross-generator backgrounds
using actual saved background-arm weights.

## Pre-registered source and decision boundary

Before any trained model was run on this cohort, generator
`TD1-independent-mosaic-background-v1`, seed **2026110203**, and exact
SHA-256 **`9366f137ba573d6a63f34bce0649b033bdd4e0bf5f3fdb9cb8edfc6fdcbc8edf`**
were frozen from generator-only execution on 2026-11-02.
`blind_background_comparison.py` is the source of truth. The cohort comprises
**120 distinct 64×64 RGB images**: 12 per class (red/blue) per edge
(left/top/right/bottom), plus 24 negatives with heterogeneous gray
rectangular distractors. Every positive has a single edge-bound colored
rectangle with ground-truth location and nonflat pixels on a separately
seeded mosaic; backgrounds are never labeled as objects. Unlike the already
observed striped cohort from #56, no source image is reused. Pixel hashes,
population, geometry, label/class-vs-color consistency and the frozen cohort
digest are covered by Python negative-control tests.

The intervention is *only* the opt-in `--background` training loss gradient
introduced by #57. Both arms keep initialization seed404, seed101/4096
training samples, sampler seed707, same mirror-reserved sample ordering,
2,000 Adam steps at batch64, same 192→64→7 architecture and original
seed202 validation gate. The mirror comparator uses **identical** training
setup except background-logit gradient multiplier 1 instead of 2.
Neither arm reads final seed909 holdout in this paired comparison.

The `--evidence-artifact <new-path>` argument now also accepts
`--background`. It serializes *only after the original validation gate
passes*, checks exact weight roundtrip, creates a new file without overwrite,
and records its SHA-256. No artifact format, native ABI, model topology or
dependency changes. The CI caller compares each digest with the actual
temporary bytes, loads two independently trained artifacts in **separate
Rust inference processes**, and rechecks immutable weight hashes afterwards.
Artifacts and RGB inputs remain in ignored temporary CI paths and are
**not** included in uploaded GitHub Actions artifacts.

JSON evidence includes every per-image prediction and immutable source/model
identities, per-side/class IoU50, missing/wrong-class counts and mean center
error. For the 24 hard negatives it reports *actual false positives*;
comparison also records IoU50 hit difference and background false-positive
reduction. Negative, zero, or mixed results are valid evidence, not reasons
to silently change test thresholds or discard images. The report verifier
reconstructs group metrics and comparison deltas from the raw predictions
and rejects provenance drift, missing/reordered data, malformed detections,
nonfinite numbers or altered model contents.

**Not a production threshold or deployment decision.** This is a one-time
cross-generator *synthetic* cohort whose labels/generator are known; its first
trained inference is a one-time prospective diagnostic, not evidence of
natural-image accuracy, statistical generalization, released weights/SDKs,
or justification to promote background reweighting to default. Once CI
runs this cohort, treat it as observed and do not tune against it. Both
Linux and Windows, and Burn/Candle, must be read separately; no cross-host
bitwise weight identity is asserted.

### Reproduction

With the pinned toolchain and cached dependencies, the CPU experiment CI
runs `python experiments/cpu-detector/verify.py <new-evidence-dir>`.
It emits `blind-background-burn.json` and
`blind-background-candle.json` in that evidence directory. To check the
generator and tamper controls without performing model training:

```sh
python -m unittest discover -s experiments/cpu-detector -p 'test_blind_background_comparison.py' -v
```

CI must run exact-head Rust/C/.NET gates, independent review, and model
inference before accepting a merge. An opt-in experiment completing this
check is **not** authorization to distribute weights or SDKs.
