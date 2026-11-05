# ADR 0002: training framework for the TD2 task family

Date: 2026-11-05

Status: accepted under the owner's 2026-11-05 delegation ("framework decided
by stage B evidence").

## Context

Stage B of the [TD2 design](../superpowers/specs/2026-11-05-td2-general-detector-design.md)
selects one training framework for T1–T4. The candidates, the `fyp1` proxy
(convolution, depthwise convolution, batch norm, SiLU, nearest upsample,
addition), the pass/fail gates and the decision rule were fixed in
[the comparison record](../testing/td2-framework-comparison.md) before any
measurement. The budget is zero, so the comparison ran on CPU on free
GitHub-hosted `ubuntu-24.04` (AMD EPYC 9V74) and `ubuntu-24.04-arm` (Neoverse-N2)
runners.

## Decision

**Burn 0.22.0** (`std`, `flex`, `autodiff`; eager CPU execution) is the training
framework for the TD2 task family, beginning with stage D.

## Evidence

From run [42017843115](https://github.com/KoalaFacts/FactsYOLO/actions/runs/42017843115),
both hosts:

- Both candidates pass every gate: all operations differentiable through public
  APIs; forward loss within 4e-8 of the independent F64 reference; all 125
  sampled gradients within the finite-difference bound; Burn–Candle full
  gradients within 9e-7 relative L2; loss reduced to 0.42–0.43 of its start
  over 300 steps; bitwise-reproducible reruns.
- Correctness margin (rule criterion 1): Burn's worst gradient ratio is
  1.9–2.2e-4 against Candle's 1.5e-3.
- Throughput (criteria 3–4), all cores, batch 16: Burn 532.8 vs Candle 201.9
  images/s on arm64 (2.64×), 729.0 vs 179.7 on x64 (4.06×).
- Peak RSS at batch 64 (criterion 5): Burn about 64–65 MiB vs Candle about
  158 MiB.
- Dependency footprint (criterion 2) is split: Burn selects 151–152 packages
  including `colored` 3.1.1 under MPL-2.0; Candle selects 126 packages and
  compiles bundled Oniguruma C code (`onig_sys`, BSD-2-Clause notice) through
  `tokenizers`.

The rule gives no materiality threshold for criterion 1; Burn is ahead on it,
and every later criterion except the split criterion 2 favours Burn by more
than the predeclared 20 %. The selection does not depend on how criterion 1 or
2 is read.

## Consequences

- Stage D training code uses Burn. Losses (focal, GIoU), the optimizer, target
  assignment and the evaluator remain plain Rust shared code where practical,
  so a later framework change touches the graph adapter, not the method.
- Burn is a **training-side** dependency only. The deployed library keeps its
  own portable Rust inference (stage E) and links no training framework, so the
  `colored` MPL-2.0 obligation applies only if a training binary is ever
  distributed (not planned), as already recorded for TD1.
- Burn Flex did not scale with threads at proxy size. Reduced-scale CPU training
  in stage D must budget for single-threaded steps or use data-parallel
  processes; the documented Burn GPU backends (WGPU, CUDA, ROCm) are the path
  for full-scale training once compute exists, not exercised here.
- The exact version is pinned (`=0.22.0`). Upgrading Burn later is a dependency
  change: rerun the stage B gates on the new version before adopting it.

## Not decided here

GPU backend choice, mixed precision, distributed training, INT8 export (stage G)
and any training-binary distribution.

## References

- [Stage B plan and results](../testing/td2-framework-comparison.md)
- [Proxy dependency record](../../experiments/td2-framework-proxy/DEPENDENCIES.md)
- [TD1 dependency review](../../experiments/cpu-detector/DEPENDENCIES.md)
