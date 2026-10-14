# TD1 v1 Compatibility Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, TDD, then a fresh independent whole-branch reviewer. The owner approved this bounded gate and normal merge after all checks; do not repeat authorization.

**Goal:** Make historical TD1 v1 compatibility a repeatable synthetic acceptance gate.
**Architecture:** Shared frozen test vectors independent of current model code feed portable and native FFI tests. Preserve all production format, ABI and inference behavior.
**Tech Stack:** Existing Rust1.85 portable/native code, SHA-2 already in the portable crate, current C/C# gates. No new dependencies.
**Spec:** [TD1 v1 compatibility contract](../../testing/td1-v1-compatibility.md).

## Global Constraints

- Baseline PR13 merge a58f6c44beb6b7c0c35452da2d3057de41b6ccb9; ABI1.0, schema FYTDv001, architecture1,51,284 bytes remain unchanged.
- Synthetic test vectors only; no new architecture, platform, paid data, SDK publication, trained weights, database or permission changes.
- Existing train/parse/binding implementations are reused. Test-only regression controls must never enter production commits.

## Review Focus

- Current serializer/constants accidentally generating their own supposed historical oracle.
- Invalid headers/parameters failing only because a stale checksum hides missing semantic checks.
- Exact +/-10,000, signed-zero, parser-budget and score0.5 boundaries drifting.
- Wrong error precedence or failed creation/detection modifying outputs or preventing recovery.
- Synthetic gate evidence overstating SDK, platform, trained-model or real-image support.

### Task1: Freeze independent vectors and portable compatibility

**Files:** tests/compat/td1_v1_vectors.rs; experiments/cpu-detector/model/tests/historical_v1.rs; compatibility contract.
**Interfaces:** frozen Vector { name, header patches, parameter bit patches, digest, accepted }; bytes(&Vector)->Vec<u8>; vector(name)->&Vector. Consumers may not source constants or expected bytes from current Model.
- [x] Write portable historical byte/digest, bit-preserving roundtrip, predictions, semantic-header and numeric-boundary assertions.
- [x] Run these assertions against disposable coordinated-schema and header-validation regressions. Expected: compiled tests fail at historical acceptance assertions; restore no production files because mutations are isolated copies.
- [x] Run the unchanged PR13 implementation against the frozen vectors. Expected: all historical tests and existing portable suite pass on1.85/1.99; record that production needed no change.
- [x] Commit the independently testable vector/portable contract gate.

### Task2: Native budgets, error categories and recovery

**Files:** crates/facts-yolo/tests/td1.rs; same shared vector module.
**Interfaces:** consumes Task1 vector()/bytes(); uses existing Context/options/image/copy helpers and real fy_* APIs.
- [ ] Add historical model/status/identity tests, zero/N-1/N/N+1 model budgets, ABI/flag controls, failed-open recovery and exact score-threshold/session-reuse cases.
- [ ] Demonstrate compiled failures under disposable parser-budget and score-threshold regressions. Expected: specific boundary assertions fail.
- [ ] Run unchanged native code under TD1/combined Debug/Release,1.85 and actual TD1 Miri. Expected: all pass, production protocol diff empty.
- [ ] Commit native compatibility coverage and evidence-driven CI/documentation wiring.

### Task3: Full acceptance and merge

**Files:** README/testing docs and workflows only if needed to run the added coverage within a measured budget.
- [ ] Fresh independent whole-branch review; address material findings with failing regressions.
- [ ] At final clean head run workspace builds/verify/explicit AOT/Miri/i686, full TD1/combined tests/clippy/MSRV, full experiment and actual Linux C/C# consumers for both trained outputs. Expected: all pass, synthetic scope recorded.
- [ ] Draft PR, then ready after local acceptance; all normal CI success and zero unresolved comments; ordinary expected-head merge and postmerge CI. Preserve command/commit/evidence links.

The test-only change uses controlled mutations for RED because the authorized
contract preserves already-correct production behavior. A green current baseline
is not presented as a newly fixed production bug. The mutation evidence proves
the added gate detects the specific future regressions it promises to reject.
