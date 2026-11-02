# TD1 small-object spatial-aliasing design gate

**Status:** Constructed counterexample, not a detector improvement or approved
model/ABI/format design. Reproduced on 2026-11-03 from `main` at
`ee99b47ef0bfb0bd98a149da891c4f2b4f60da8d` (research reference).

## Concrete problem

The [accepted TD1 design](../superpowers/specs/2026-10-14-cpu-detector-design.md)
turns a 64×64 RGB image into 192 features: the RGB mean in each 8×8 tile.
This is deterministic spatial compression. Two *single-object* images can
therefore encode distinct ground-truth boxes as exactly the same network input.
No number of training steps, class reweighting or box gradients can recover
information the feature extractor has already discarded.

The executable constructed counterexample places an identical 4×4 red square
in the upper-left or lower-right quadrant of one 8×8 tile. The rest of each
64×64 image is RGB(7,7,7). Both have the same class, object size and pixel
histogram. Their TD1 F32 features are identical **bit for bit**, yet their
true boxes are `[0,0,4,4]` and `[4,4,8,8]`.

They are disjoint equal-area objects of 16 pixels each. If one predicted box P
had IoU≥0.5 with *both*, writing its area as `p`, and its intersections as
`i1`, `i2`, each IoU inequality requires `3*i >= p+16`. Disjointness gives
`i1+i2 <= p`; hence `p >= 32`. Each intersection is at most 16; hence
`p <= 32`. Together these imply `p=32` and each intersection=16, so P would
have to contain both complete objects. But their minimum enclosing rectangle
has area 64, not 32. **No single predicted axis-aligned box can hit both at
IoU50**. A deterministic TD1 model gets identical features and therefore
identical output for the pair; even an oracle choice can achieve at most one
of the two IoU50 hits (50% recall on this constructed pair).

This proof is for a deliberately constructed, out-of-current-TD1-generator
4×4 object. It does **not** establish that this exact collision caused the
previously observed misses on 12–22px training boxes or other small-object
development cohorts. It demonstrates a structural limitation as future object
sizes decrease. The existing Burn/Candle, FYTDv001 and native contract remain
unchanged.

## Reproduce without a model or dataset

```sh
python3 -m unittest discover -s experiments/cpu-detector \
  -p 'test_spatial_alias_audit.py' -v
python3 experiments/cpu-detector/spatial_alias_audit.py
```

The source material is constructed in memory. The report records exact pixel
SHA256s, a common F32 feature SHA256, the different target boxes, a geometric
IoU50 upper bound and the flags `training_executed=false`,
`final_holdout_evaluated=false`, `detector_accuracy_claimed=false`. No held-out
split, weights, images or outside code are loaded. The negative control with
two adjacent 4×4 targets verifies that **some** disjoint boxes *can* share a
predicted IoU50 box, so diagonal separation is essential to the proof.

## Current TD1 generator range: exhaustive collision counterbalance

The strict IoU50 impossibility witness above is **outside** the registered
training range. To avoid using it to justify a model change prematurely,
`inrange_alias_audit.py` exhaustively enumerates all
40 × 40 × 11 × 11 = **193,600** allowed pixel-edge rectangles with
x/y origins 2..41 and widths/heights 12..22. It groups rectangles by
the exact count of foreground pixels in each 8×8 block, which gives
identical 192 TD1 F32 means **when both images use the same foreground
RGB and the same uniform background**. No model or held-out data is used.

There are **193,543 distinct signatures**, **57 colliding pairs** and no
groups larger than two. The smallest true-box IoU among those pairs is
**0.60**, attained by [10,8,22,24] versus [8,10,24,22].
Thus every colliding pair in *this restricted geometry/color model*
has at least one common IoU50-capable prediction (one of its true boxes).
An alias here is real information loss, **not automatically an IoU50
impossibility** at the current generator's object sizes.

Reproduce with:
```sh
python3 -m unittest discover -s experiments/cpu-detector -p 'test_*alias_audit.py' -v
python3 experiments/cpu-detector/inrange_alias_audit.py
```

This does **not** enumerate variable-background/foreground collisions,
occlusion, textured images or prediction errors of an actual trained
network. It does not show that 8×8 is adequate, or that 4×4 would
improve detection; matched, real Rust training evidence and approval
remain necessary for any architecture/format change.

## Decision required before a new model

A candidate 4×4 RGB mean encoder distinguishes this exact pair. It has 768
features rather than 192. Holding the existing 64 hidden units and 7 output
values constant *for arithmetic comparison only* would raise parameter count
from 12,807 to **49,671**, a further **147,456 F32 parameter bytes**. This
calculation is not a runtime memory/CPU measurement, a framework selection or
a proposal to reuse the FYTDv001 header with different parameter counts.

Before implementing a new architecture, obtain a separate approved design
specification for feature resolution, model identity and artifact compatibility,
training target(s), correct-class detection/IoU50 on disjoint development
strata, negative-background behavior, inference CPU/memory budgets, and a
fresh pre-registered held-out cohort. Then compare real Rust Burn/Candle
training at matched seeds/budget with independently implemented preprocessing
checks. No current model, default training pipeline, ABI, SDK or format is
changed by this evidence.
