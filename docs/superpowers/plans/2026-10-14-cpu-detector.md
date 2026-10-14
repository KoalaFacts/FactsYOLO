# CPU Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline. Steps use checkbox syntax for tracking.

**Goal:** Reproduce real CPU training/save/reload/inference of one original TD1 detector with two tensor backends.
**Architecture:** Independent evaluation workspace; portable fixed-model/data/artifact crate plus Burn/Candle adapters and a common experiment runner. Root native API stays unchanged until the following merged stage.
**Tech Stack:** Portable Rust1.85; experiment Rust1.99; pinned Burn0.22/Flex and Candle0.11; CPU F32, SHA-256 data-only artifact.
**Spec:** [CPU detector design](../specs/2026-10-14-cpu-detector-design.md).

## Global Constraints

- Preserve root Rust1.85, offline M1 checks, native exports and default unsupported model behavior.
- Exact TD1 topology192->64->7, 12,807 parameters, original data and random initialization.
- No external weights/data, GPU/JIT features, package or weight publication, database changes.
- Fixed2000 steps/batch64/Adam0.003; acceptance thresholds exactly as spec, set before results.
- Keep this stage isolated and merge it before C/C# integration.

## Review Focus

- Empty/background images must suppress detections rather than rewarding box-only regression.
- Split/seed handling must not leak training examples into held-out metrics.
- Invalid artifacts must fail before allocations based on attacker-controlled sizes.
- Candidate gradients and optimizer updates must actually change model behavior, with honest loss/evaluation.
- Dependencies must not raise root/portable MSRV or enable native/GPU requirements accidentally.

### Task1: Portable model, data, loss and artifact

**Files:** `experiments/cpu-detector/model/{Cargo.toml,src/{lib,data,network,artifact,loss}.rs,tests/contract.rs}`; independent workspace manifest/lockfile.
**Interfaces:** `Sample { pixels: Vec<u8>, class: Option<usize>, box_xywh:[f32;4] }`; `dataset(seed,count)->Vec<Sample>`; `Model { weights:Vec<f32> }`, `Model::initialized(seed)`, `forward(&[f32])->[f32;7]`, `detect(&[u8])->Result<Option<Detection>,String>`, `to_bytes()/from_bytes()`; `features(&[u8])->Result<Vec<f32>,String>`; `loss_gradient(logits,labels)->(f32,Vec<f32>)`.
- [x] RED compiled tests for deterministic pixels/geometry/split separation, scalar forward/decoding, finite-difference loss, exact artifact roundtrip and malformed limits/checksums.
- [x] Implement only the defined model/data/contract, checking every input length and finite value.
- [x] GREEN portable tests on1.99 and1.85; record actual outputs. Commit focused task.

### Task2: Real candidate gradients and fixed training experiment

**Files:** `experiments/cpu-detector/evaluate/` manifest/source/tests, `verify.py` and dependency inventory.
**Interfaces:** Adapter `predict(weights,features,batch)->Vec<f32>` and `gradients(weights,features,batch,output_gradient)->Vec<f32>` use actual tensors/autodiff; common runner Adam and metrics consume identical portable data/loss.
- [x] RED gradient/reference and parameter-change tests; CLI absent/invalid args must fail, not print success.
- [x] Implement adapters against exact upstream APIs, common optimizer/metrics and bounded artifact I/O.
- [x] GREEN finite differences, full candidate gradient comparison, actual2000-step runs, saved/reloaded inference and held-out metrics. Preserve raw losses and elapsed times.
- [x] Record dependency features/licenses/MSRV and negative controls. Commit when all experiment gates pass.

### Task3: Required CI/evidence and whole-stage verification

**Files:** `.github/workflows/cpu-detector.yml`, README, `docs/testing/cpu-detector.md`, notices.
- [x] Add pinned toolchain/read-only workflow executing real Linux training, portable1.85 checks and uploading logs/metrics only.
- [ ] Run final exact-head root builds, `xtask verify`, explicit AOT, Miri/i686; portable and candidate suites and complete experiment.
- [ ] Fresh whole-branch independent review; address material findings with reproductions.
- [ ] Draft PR until local/review gates pass, then all CI and zero unresolved comments before expected-head merge. Billing exceptions only after explicit confirmation.
- [ ] Preserve evidence. Start separate M2b native integration only after this merge.

## Execution authority and self-review

Owner explicitly requested writing the specification and implementation in this turn,
with no repeat question about the approved target. Execute inline; final fresh review.
The spec/plan implement the approved CPU synthetic training slice. They do not select
an external model/license or raise core compatibility. Shared interfaces were checked:
data/loss feed both adapters; canonical parameter order feeds adapters/artifact/reference;
runner metrics use actual decode; all gates are additive to M1.
