# Frozen TD1 four-side *paired* location diagnostic

Frozen before any Burn/Candle inference on this split (2026-10-29). This is an
**experimental synthetic distribution stress test**, not an independent 192-image
generalization sample, trained weight release, SDK release, or promotion gate.

The previous TD1 generator seed101/4096 (training), seed202/128 (validation),
seed909/128 (registered final holdout), seed20261030/4096 (observed development)
and seed20261101/4096 (observed prospective) remain unchanged. No predictions
from seed909 are used for this experiment. The pre-existing 2,000-step control,
4/64 edge and 4/64 mirrored arms retain seed404 initialization, seed707 sampling,
batch64, optimizer, topology and validation thresholds.

## Frozen geometry-only selection (before predictions)

Generator: existing original `TD1-synthetic-v1`, seed **20261113**, first
**4,096** candidate rows. Select the first 24 *small left-only* and first 24
*small top-only* class-positive rows in source index order (area <=256 pixels²,
within 2 pixels of that border, but not within 2 pixels of any other border).
The last selected original index is **3953**. No model outputs affect selection.

For each of the 48 distinct source images, produce identity, horizontal,
vertical and both-axes pixel reflections with the exact matching normalized
XYWH target reflections, without new RNG draws. Map the left-origin rows to
the left/right groups according to horizontal reflection and top-origin rows
to top/bottom according to vertical reflection. Each side thus contains 48
observations from **24 unique source images, each reused twice**. The total
192 observations are four views per 48 original images. Side groups are
correlated: do not quote them as four disjoint independent tests.

Order groups left, top, right, bottom, preserving source order within each.
Fingerprint SHA256 over ASCII `TD1-four-side-paired-v1` then for each row:
class as one byte, four little-endian F32 XYWH fields, and 12,288 packed RGB
bytes. Independent C++-generator plus Python pixel-reflection freeze:

`59a4a885ff0ca2c5c8e700f8ccaaefbdaf454f2f2d0e6a72d280026b8398b3bd`

Frozen group sizes: 48 each; 192 distinct transformed pixel arrays. The
source-index selection and fingerprint are verified before backend inference.

## Predeclared reporting and interpretation

Run *all three unchanged training arms* with Burn and Candle on Linux and
Windows. For each, report every side's correct-class IoU>=0.5 numerator,
correct-class recall, mean IoU with misses/wrong-class as zero, missed
detections and wrong-class counts. Report negative outcomes, including any
original validation regression; never select a favorable seed or threshold.
Keep the existing validation gate and the previous two observed diagnostic
splits. Verify exact-head provenance and aggregate arithmetic, and retain raw
evidence in CPU experiment CI artifacts.

This one paired stress test can identify position asymmetry but **cannot**
demonstrate natural-image accuracy, independent 192-sample generalization,
causal benefit, production fitness, or justify adopting the mirror sampler.
Such claims require a separately designed, genuinely independent source-image
test, accuracy criteria fixed before prediction, and independent review.
No public ABI, model format, ownership, licensing or inference behavior changes.
