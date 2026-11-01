# Frozen TD1 disjoint-source four-side evaluation

Frozen on **2026-11-01 before any predictions on this split**. This is one
prospective *synthetic* model-accuracy comparison, not a natural-image benchmark,
production qualification, artifact publication, or reason to change the TD1 ABI.

## Why this test is different

The earlier [four-side stress test](td1-four-side-paired.md) reuses each of
48 source images four times. Its observed right/bottom mirrored-training gains
are encouraging but the 192 reflected observations are dependent. This next
test uses **144 distinct source images**: 24 per side plus 48 backgrounds.
Each positive source appears **once in exactly one group**. The left/top
sources are not reused for right/bottom reflections.

Train exactly the existing unchanged three arms using the already implemented
2,000-step Burn/Candle experiment: original sampling, four reserved edge
samples, and the existing mirrored reserved-slot training. Do not retrain on
this new source, select checkpoints, change hyperparameters, or examine final
seed909 predictions for intervention selection.

## Source and registered selection

- Generator: existing `TD1-synthetic-v1`, seed **20261123**, exactly
  **8,192** candidate examples in source index order.
- Eligible positive: class 0/1, target area <=256 square pixels, within
  two pixels of **exactly one** side. The original generator only yields
  small left-only and top-only targets, never right/bottom small targets.
- Take the first **48** eligible small left-only sources and first **48**
  eligible small top-only sources, in their own source order. Original
  left positions 0..23 belong to **left** unchanged; left positions 24..47
  belong to **right** with one horizontal RGB+box reflection. Original top
  positions 0..23 belong to **top** unchanged; top positions 24..47
  belong to **bottom** with one vertical RGB+box reflection.
- Take the first **48** background sources unchanged. These are disjoint
  from positive sources and include no reflected copies.
- Output groups and fingerprint order are **left, top, right, bottom,
  background**, preserving source order within each. Four side groups have
  24 rows each; background has 48. Total **144 unique source images**.
- Domain-separated SHA-256 fingerprint: ASCII
  `TD1-four-side-disjoint-v1`, followed for each selected output row by
  class byte (255 for background), four little-endian F32 normalized XYWH
  values, and exactly 12,288 RGB bytes. Frozen digest:

  `e156306394733b94c5d99daec4ae22647819cab6976bc3dd55938eb988500a56`

An independent C99 implementation of the published deterministic LCG,
rasterization and RGB reflection reproduced the previously frozen paired
fingerprint `59a4a885ff0ca2c5c8e700f8ccaaefbdaf454f2f2d0e6a72d280026b8398b3bd`
before calculating this new fingerprint. Its last selected source index was
**6876**. 144 distinct base-source pixel hashes were checked against each
other and against source seeds 101/4096, 202/128, 909/128, 20261028/256,
20261030/4096, 20261101/4096 and 20261113/4096: **zero overlap**.
The C99 generator verification here is preparation, not a Rust/CI test pass.
No model or previously measured results were consulted to select these rows.

## Predeclared interpretation (before new-split inference)

Report each arm and backend on all groups: correct-class IoU>=0.5 hits,
missing detections, wrong-class detections, mean correct-class IoU including
misses as zero, background false positives and exact training provenance.
Keep negative or mixed results; do **not** fail CI solely because a strategy
does not improve accuracy.

Consider the *existing* mirror sampler for **further synthetic research**
only if, on this disjoint-source split on **both** Burn and Candle, on **both**
Linux and Windows: (1) relative to unchanged baseline, **right and bottom
each gain >=8 of 24 correct-class IoU50 hits**; (2) left and top each lose
**at most two** correct-class IoU50 hits; (3) background false positives
do not increase; (4) the unchanged seed202 validation passes for both arms.
Report the per-host/per-backend outcome even when these conditions fail.
No tweaking the frozen cohort, rule, architecture, initialization, optimizer,
sampling intervention, dataset or final seed909 threshold afterward.

This exercise tests disjoint *source images* on the same simple synthetic
generator, not independence from that generator, hidden dataset shift, or
generalization to natural images. It cannot justify replacing the default
sampler, publishing TD1, declaring YOLO production ready or raising a public
performance/accuracy claim. A real deployment decision needs licensed
natural-image data, independently reviewed evaluation and separate design.
