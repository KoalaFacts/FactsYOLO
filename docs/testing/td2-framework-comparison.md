# Stage B: training-framework comparison (predeclared plan)

Original text: Apache-2.0. Stage B of the
[TD2 design](../superpowers/specs/2026-11-05-td2-general-detector-design.md)
selects one training framework for every TD2-family task. This record fixes the
proxy, the measurements, the pass/fail thresholds and the decision rule
**before any measurement is taken**. Thresholds and the rule below are not
changed after results exist; a wrong threshold is corrected only by a later,
explained revision that keeps both the original and the new result visible.

Status: complete. The plan below was committed (`481ced5`, refined in
`5001c06` before any throughput or memory measurement) before the measured
results that follow it; thresholds and the rule were not changed after results. Results are added
to the "Results" section by the implementation pull requests and summarised in
ADR 0002. Work items are tracked in the [TD2 roadmap ledger](../plans/td2-roadmap.md).

## Candidates

The same exact versions already reviewed for the TD1 experiment
([dependency review](../../experiments/cpu-detector/DEPENDENCIES.md)), so stage B
adds no new direct dependency:

| Candidate | Version / features | Execution |
| --- | --- | --- |
| Burn | `burn` 0.22.0, `default-features = false`, `std`, `flex`, `autodiff` | Eager CPU (Flex) |
| Candle | `candle-core` 0.11.0, `default-features = false` | Eager CPU, `Var` + `backward` |

GPU paths (Burn CubeCL/WGPU/CUDA, Candle CUDA/Metal) are recorded from upstream
documentation and feature lists only. The budget is zero and the workspace has no
GPU, so no GPU path is built or executed in stage B.

The framework is a **training-side** dependency only. The deployed library keeps
its own portable Rust inference (stage E) and links neither candidate.

## Proxy (identical for both candidates)

A new, independent experiment workspace `experiments/td2-framework-proxy`
(`publish = false`, not a member of the root workspace) holds:

- **Synthetic data.** An original seeded generator (SplitMix64): 3×64×64 images
  in [0,1], uniform noise background, one axis-aligned filled rectangle of random
  colour, size 8–40 px and position. No external image, annotation or weight is
  used. Pixels are generated in memory and never written or uploaded.
- **Network `fyp1`.** The operator mix TD2 needs, at proxy scale:

  | Layer | Operation | Output (N=batch) |
  | --- | --- | --- |
  | stem | conv 3×3, stride 2, 3→16, BN, SiLU | N×16×32×32 |
  | dw | depthwise conv 3×3, stride 2, groups 16, BN, SiLU | N×16×16×16 |
  | pw | conv 1×1, 16→32, BN, SiLU | N×32×16×16 |
  | down | conv 3×3, stride 2, 32→64, BN, SiLU | N×64×8×8 |
  | lateral | conv 1×1, 64→32 | N×32×8×8 |
  | up | nearest upsample ×2, add `pw` output | N×32×16×16 |
  | head | conv 1×1, 32→5 (objectness + four box distances) | N×5×16×16 |

  Convolutions have no bias before BN; `lateral` and `head` have a bias. BN uses
  training-mode batch statistics, ε = 1e-5, learnable γ/β.
- **Loss, outside the framework.** Shared plain-Rust F64 code computes the loss
  and its gradient with respect to the head output Y (N×5×16×16, stride 4):
  `L = (1/N) Σ_n [ (1/256) Σ_cells BCE(Y0, t0) + (1/4) Σ_k |softplus(Yk) − dk| ]`,
  binary cross-entropy with logits for objectness at every cell (t0 = 1 only at
  the cell containing the box centre) and L1 on softplus-encoded
  left/top/right/bottom distances dk from that cell's centre to the box edges in
  stride units. Each framework back-propagates that upstream gradient, the
  pattern the TD1 adapters already use.
- **Initialization.** One seeded generator produces every parameter as F32
  (Kaiming-uniform with bound √(6/fan_in) for every kernel, BN γ=1, β=0, biases 0; 22,021 parameters). Both candidates load the
  same bytes; parameter count and SHA-256 are recorded.
- **Optimizer, outside the framework.** Shared plain-Rust SGD with momentum 0.9,
  learning rate 0.05, no weight decay, applied to the gradients each framework
  returns. Framework optimizers are not used, so only autodiff differs.
- **Schedule.** Seed 20261105, batch 16, 300 steps, one thread
  (`RAYON_NUM_THREADS=1`) for the agreement and trajectory runs.

## Independent reference

A separate plain-Rust **F64** forward of `fyp1` (written from the layer table,
sharing no code with either adapter) provides:

1. Forward agreement: loss from each candidate vs the reference, same batch and
   weights.
2. Gradient agreement: central finite differences of the reference loss
   (h = 1e-4 in F64) for a fixed predeclared sample of 125 parameters: 8 from
   each of the 16 parameter tensors (four unbiased kernels, the `lateral` and
   `head` kernels and biases, four BN γ and four BN β), except all 5 elements of
   the `head` bias; indices drawn by the same seeded generator.

## Gates (pass/fail, per candidate)

| Id | Measurement | Pass |
| --- | --- | --- |
| G1 | Operator coverage: every `fyp1` operation runs forward and backward through the public API with no hand-written gradient | all operations supported |
| G2 | Forward loss vs F64 reference, 4 fixed batches | relative error ≤ 1e-5 |
| G3 | Sampled gradients vs F64 finite differences, 125 parameters | every sample: \|g − g_fd\| ≤ 1e-4 + 1e-2·\|g_fd\| |
| G4 | Full gradient vector, Burn vs Candle, step 0 | relative L2 difference ≤ 1e-4 |
| G5 | 300-step training with the shared optimizer | finite throughout; mean loss of steps 281–300 ≤ 0.6 × mean of steps 1–20 |
| G6 | Determinism: two runs, same host and thread count | bitwise-identical loss history |

A candidate failing G1, G2, G3, G5 or G6 is not selectable. G4 failing for both
pairs with the reference passing is recorded as an inter-framework numerical
difference, not a disqualification. The trajectory difference between the two
candidates at steps 100/200/300 is reported, not gated, because F32 summation
order makes long trajectories diverge legitimately.

## Measurements (reported, compared by the decision rule)

Measured on GitHub-hosted `ubuntu-24.04` (x64) and `ubuntu-24.04-arm` (aarch64)
runners, release build, CPU model, core count, OS and toolchain recorded from the
runner:

- **Training throughput:** images per second for a full step (one taped forward,
  shared loss, backward, shared optimizer); 10 warm-up steps then 5 timed windows
  of 20 steps; median and min/max reported; batch 16 and 64; with 1 thread and
  with all runner cores (`RAYON_NUM_THREADS`).
- **Peak memory:** peak RSS (Linux `VmHWM`) of a separate process per candidate
  and batch size, read after its throughput run (10 warm-up and 100 timed
  steps), at batch 16 and batch 64.
- **Build cost:** clean release build time and selected crate count on the
  measuring runner.
- **Dependency and licence review:** the selected Cargo graph of each candidate
  separately (crate count, declared licences, native/C components, build
  scripts, MSRV), using the existing inventory tooling pattern.
- **Operator notes:** how each candidate implements grouped/depthwise
  convolution and upsample backward (from its published source), and what is
  missing for TD2 stage D (GIoU, focal loss and SIMD-friendly export are written
  in shared code either way).
- **GPU path:** documented feature and platform availability, not executed.

## Decision rule (ADR 0002)

Among candidates passing every selectable gate, in order, the first criterion
with a material difference decides:

1. **Correctness margin:** worst G3 ratio (error / allowed error).
2. **Dependency and licence footprint:** fewer copyleft (e.g. MPL-2.0) or
   bundled native C components in the selected graph; fewer crates.
3. **aarch64 training throughput**, all cores, batch 16; material = ≥ 20 %.
4. **x64 training throughput**, all cores, batch 16; material = ≥ 20 %.
5. **Peak RSS** at batch 64; material = ≥ 20 %.
6. **GPU path** for later stages without new code (documented support for
   conv/BN/upsample autodiff on a free-tier-reachable GPU).

If neither candidate passes, ADR 0002 records that, and the next step is a
re-scoped plan (for example, a pinned newer release or an own minimal autodiff
for the TD2 operator set), not a silent choice. If both pass and no criterion is
material, Burn is preferred: its convolution backward takes the group count as an
operation parameter (`burn-backend` 0.22.0 conv module), while Candle 0.11.0
implements grouped convolution by splitting into per-group convolutions and
concatenating (`candle-core` `conv.rs`), which costs more as TD2 adds depthwise
layers. This tiebreak is stated now, before measuring.

## Implementation status

- **B2 core** (`fyp-core`, no dependencies, Rust 1.85 floor): SplitMix64
  (checked against the published seed-0 sequence), the synthetic generator, the
  16-tensor parameter table and seeded initialization, the shared loss and its
  gradient, shared SGD, the 125-parameter sample plan, and the independent F64
  reference forward with central finite differences. Its 11 tests check
  convolution stride/padding/groups by hand and against per-group convolution,
  batch-norm statistics, nearest upsampling, the loss gradient against finite
  differences of the loss, and that finite differences of the full reference
  network for every `head` bias equal the summed head-output gradient (tying the
  reference forward to the loss gradient).
- **B3 adapters** (`fyp-compare`): Burn and Candle graphs of `fyp1`. Every
  parameter is a flat 1-D leaf reshaped inside the graph, so gradients return in
  the shared layout; batch norm is composed from tensor primitives in both
  (neither reviewed crate has a batch-norm operation). 5 reduced-size tests
  (batch 2, 40-step training) check forward and gradient agreement with the
  reference, Burn–Candle agreement, determinism and input validation. The
  `fyp-compare gates` command runs the full predeclared G1–G6; `fyp-compare
  bench <framework> <batch>` measures one process.
- **B4 measurement** in the `TD2 framework proxy` workflow (`measure` job) on
  both runners; scalar JSON and dependency trees are uploaded as artifacts.

## Results

Run [42017843115](https://github.com/KoalaFacts/FactsYOLO/actions/runs/42017843115)
on commit `f93598be34d5e1ebe231040ada5158c3edf474ca`, Rust 1.99.0, release
build. Initial parameters SHA-256
`3fa5f6a5fd0ece1a4c4da4dec05f4dc117b873825a6f696b67bacb13b2ede752` on both hosts.

| Host | CPU, cores | Kernel |
| --- | --- | --- |
| `ubuntu-24.04` | AMD EPYC 9V74, 4 | 7.0.0-1017-azure |
| `ubuntu-24.04-arm` | Arm Neoverse-N2, 4 | 7.0.0-1017-azure |

### Gates (one thread)

| Gate | Burn x64 | Burn arm64 | Candle x64 | Candle arm64 | Pass bound |
| --- | --- | --- | --- | --- | --- |
| G1 operations | all | all | all | all | all |
| G2 max relative loss error | 2.43e-8 | 2.43e-8 | 3.98e-8 | 2.66e-8 | ≤ 1e-5 |
| G3 worst ratio (125 samples) | 2.22e-4 | 1.94e-4 | 1.55e-3 | 1.48e-3 | ≤ 1 |
| G3 failing samples | 0 | 0 | 0 | 0 | 0 |
| G5 last-20 / first-20 loss | 0.4244 | 0.4232 | 0.4232 | 0.4232 | ≤ 0.6 |
| G6 bitwise-identical rerun | yes | yes | yes | yes | yes |

G4 (Burn vs Candle full gradient, step 0, relative L2): 8.11e-7 on x64 and
8.97e-7 on arm64 (bound 1e-4). **Both candidates pass every gate on both hosts.**

Reported, not gated: loss at steps 1/100/200/300 was 3.1038/0.8140/0.8577/0.8400
for Burn on x64 and 3.1038/0.8140/0.8577/0.8207 for the other three runs; the
trajectories separate after step 200 by F32 summation order, as anticipated.
The local development workspace (x64, 2 CPUs) reproduced the x64 gate values
exactly.

### Throughput and peak RSS

Images per second, median of 5 windows (min–max spread under 5 % in every
cell); peak RSS (`VmHWM`) in MiB; `t` is `RAYON_NUM_THREADS`.

| Host | Batch | Burn t=1 | Burn t=4 | Candle t=1 | Candle t=4 | Burn RSS (t=4) | Candle RSS (t=4) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x64 | 16 | 734.6 | 729.0 | 152.7 | 179.7 | 20.9 | 44.4 |
| x64 | 64 | 668.1 | 669.1 | 152.9 | 186.8 | 65.3 | 158.1 |
| arm64 | 16 | 568.9 | 532.8 | 168.6 | 201.9 | 19.7 | 43.6 |
| arm64 | 64 | 539.7 | 540.7 | 173.9 | 218.5 | 64.0 | 157.5 |

At all cores and batch 16, Burn is 4.06× Candle on x64 and 2.64× on arm64; at
batch 64 its peak RSS is 41 % of Candle's. Burn Flex did not scale with
`RAYON_NUM_THREADS` at this proxy size (single-threaded eager execution here);
Candle gained 18–26 % from four threads. 300 gate-training steps took 6.8 s
(Burn) vs 27.6 s (Candle) on x64 and 9.5 s vs 25.7 s on arm64.

### Build cost and dependency footprint

A clean release build of `fyp-compare` (both candidates together) took 154 s on
x64 and 143 s on arm64. Selected normal+build graphs (`cargo tree -p <crate>`
with the enabled features, unique packages):

| Candidate | Packages | Copyleft | Bundled native code |
| --- | --- | --- | --- |
| Burn 0.22.0 | 152 x64 / 151 arm64 (`cpufeatures` 0.3.1 x64 only) | `colored` 3.1.1, MPL-2.0 | none found |
| Candle 0.11.0 | 126 | none | `onig_sys` 69.9.3 (Oniguruma C, BSD-2-Clause notice) via `tokenizers`, built with `cc` |

All other declared licences are MIT, Apache-2.0, BSD-2-Clause, Zlib, Unlicense,
BSL-1.0 or Unicode-3.0 options; obligations are as recorded in the
[TD1 dependency review](../../experiments/cpu-detector/DEPENDENCIES.md). Neither
graph is linked by the deployed library.

### Operators and GPU path

Both run every `fyp1` operation forward and backward through public APIs.
Burn's convolution takes `groups` as an operation parameter; Candle 0.11.0
splits grouped convolution into per-group convolutions and concatenates them.
Neither reviewed crate provides batch norm, focal or GIoU loss; TD2 composes them
from primitives or computes them in shared code either way. GPU paths, from the
published crates and not executed: Burn has CubeCL-based WGPU, CUDA and ROCm
backends (`burn-wgpu`, `burn-cuda`, `burn-rocm` 0.22.0); Candle has `cuda` and
`metal` features of `candle-core` 0.11.0.

### Decision rule outcome

Both candidates are selectable. Criterion 1, correctness margin, already
differs: Burn's worst G3 ratio is about 7× smaller than Candle's on both hosts
(both far inside the bound). The rule gives no materiality threshold for this
criterion, so the outcome is checked against every later criterion: 2 is split
(Burn has one MPL-2.0 crate and 26 more packages; Candle bundles C code), and 3,
4 and 5 all favour Burn by far more than 20 %. Every reading of the rule selects
**Burn**. See [ADR 0002](../decisions/0002-training-framework.md).

## Limits

The proxy measures framework correctness and cost at a scale free runners can
run. It does not predict TD2 accuracy, GPU performance or deployment latency, and
it is not a model release; generated pixels and trained proxy weights are
ephemeral.
