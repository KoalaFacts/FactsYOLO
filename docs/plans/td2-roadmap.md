# TD2 roadmap ledger

Original text: Apache-2.0. Tracks every work item of the accepted
[TD2 and CCTV vision task design](../superpowers/specs/2026-11-05-td2-general-detector-design.md)
(stages A–J). Updated in every pull request that advances an item.

## Goal

An original, Apache-2.0, Rust-first vision core with a C ABI and thin C# (later
Go/TypeScript) bindings, embedded by customers in their own CCTV cameras, NVRs
and edge devices on premises. Tasks in order: T1 detection (TD2: objects,
animals, people, vehicles, faces, licence plates) → T2 licence-plate recognition
→ T3 scene recognition → T4 face recognition. No data leaves the customer's
control; the deployed library has no networking, writes no files and stores
nothing. Budget is zero: free GitHub-hosted runners and the development
workspace only.

Status values: `todo`, `in progress (PR #n)`, `done (PR #n, commit)`,
`blocked (reason)`. Evidence names the tests, CI runs and measured numbers with
their hardware.

## Work items

### A — design

| Id | What | Status | Evidence |
| --- | --- | --- | --- |
| A1 | TD2 and task-roadmap design; owner decisions recorded | in progress (PR #68) | Docs only; spec section 12 |

### B — training framework (ADR 0002)

| Id | What | Status | Evidence |
| --- | --- | --- | --- |
| B1 | Predeclared comparison plan: proxy, gates G1–G6, measurements, decision rule | in progress (PR #69) | [td2-framework-comparison.md](../testing/td2-framework-comparison.md), committed before measuring |
| B2 | Proxy workspace core: synthetic generator, seeded init, shared loss and SGD, independent F64 reference forward and finite differences | in progress (PR #69) | `fyp-core`: 11 tests, Debug/Release, Rust 1.99 and 1.85, x64 and arm64 |
| B3 | Burn and Candle adapters; gates G1–G6 as tests | in progress (PR #69) | `fyp-compare`: 5 tests; full G1–G6 pass for both on both hosts (run 42017843115) |
| B4 | Workflow on `ubuntu-24.04` and `ubuntu-24.04-arm`: throughput, peak RSS, build cost; scalar JSON artifacts | in progress (PR #69) | Burn 729.0 / 532.8 img/s vs Candle 179.7 / 201.9 (x64 EPYC 9V74 / arm64 Neoverse-N2, batch 16, 4 threads) |
| B5 | Per-candidate dependency and licence inventory | in progress (PR #69) | Burn 152 packages, 1 MPL-2.0; Candle 126, bundled Oniguruma C |
| B6 | ADR 0002 selecting one framework by the predeclared rule | in progress (PR #69) | [ADR 0002](../decisions/0002-training-framework.md): Burn |

### C — T1 data tooling and licence record

| Id | What | Status | Evidence |
| --- | --- | --- | --- |
| C1 | Research and record Open Images V7 / COCO terms, attribution procedure and class mapping (≤ 120 classes) in MODEL_LICENSES.md | todo | |
| C2 | Owner approval of the recorded terms before any download | todo (owner gate after C1) | |
| C3 | Deterministic subset manifests with fingerprints; no pixels in git | todo | |

### D — T1 training and evaluation harness

| Id | What | Status | Evidence |
| --- | --- | --- | --- |
| D1 | Freeze the TD2 layer table within ≤ 3.0 M params / ≤ 2.5 GFLOPs at S=320 | todo | |
| D2 | Independent COCO-style evaluator (mAP50-95, mAP50, size and per-class AP) | todo | |
| D3 | Resumable chunked training workflow on free runners (cache checkpoints, scalar metrics) | todo | |
| D4 | Reduced-scale T1 run reported against spec section 9 | todo | |

### E — native `td2` backend

| Id | What | Status | Evidence |
| --- | --- | --- | --- |
| E1 | Decision record: `FY_BACKEND_TD2_CPU`, FYTDv002, GRAY8/YUV inputs | todo | |
| E2 | FYTDv002 parser with existing limits | todo | |
| E3 | Portable Rust inference matching training within tolerance | todo | |
| E4 | NEON and AVX2 paths | todo | |
| E5 | aarch64 Linux CI with real execution | todo | |
| E6 | CI gate: no networking dependency or socket/HTTP symbols in deployed builds | todo | |
| E7 | C and C# JIT/AOT consumers on Linux x64/aarch64 and Windows x64; section 9 budgets | todo | |

### F–J

| Id | What | Status | Evidence |
| --- | --- | --- | --- |
| F1 | First SDK release (detection) | todo (publishing is an owner decision) | |
| G1 | INT8 post-training quantization with measured accuracy delta | todo | |
| H1 | T2 plate recognition design (AU/NZ formats first) | todo | |
| H2 | Original synthetic plate renderer; font/layout licence check | todo | |
| I1 | T3 scene recognition design and permitted-data search | todo | |
| J1 | T4 face recognition design and safeguards | todo | |
| J2 | T4 training data | blocked (owner decision: budget; no free dataset permits commercial face-recognition training) | Spec sections 8 and 11 |

## Run log

- 2026-11-05: baseline gate on main `55cd10f` passed locally (Debug/Release,
  contract fixtures, clippy, fmt, `xtask c`; 322 tests). PR #68 updated with
  the owner decisions and marked ready. B1 plan drafted (this PR).
- 2026-11-05: stage B proxy, gates and measurement (B2–B5) run on free x64 and
  arm64 runners: both candidates pass G1–G6; Burn selected in ADR 0002 (B6).
  One PR (#69) carries B1–B6 so the plan commit precedes the results in history.
