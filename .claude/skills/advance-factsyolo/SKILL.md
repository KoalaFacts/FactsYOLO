---
name: advance-factsyolo
description: How the scheduled task advances FactsYOLO — goal, headline metric, gate, work order, scope and project rules. Read with agent-harness/autonomous-dev at the start of every run.
---

# Advance FactsYOLO

This skill is the repo-specific half of the scheduled development task ("FactsYOLO 推进计划"). The run loop is `agent-harness/skills/autonomous-dev`, and the owner's seed is `agent-harness/projects/factsyolo.md`. Where the seed and this file differ, the stricter rule wins. The task keeps this file current through its own PRs. It may make rules stricter, never looser, and it never relaxes the hard limits in the scheduled task's prompt.

## Goal and metric
- FactsYOLO is an original, Apache-2.0, Rust-first vision core with a C ABI and thin C# bindings (Go/TS later). Customers embed it in their own CCTV cameras, NVRs and edge devices, on premises.
- Goal: the TD2 general detector and the task family in the TD2 spec (`docs/superpowers/specs/2026-11-05-td2-general-detector-design.md`). Owner-approved order (2026-11-05): T1 detection (objects, animals, people, vehicles, faces, licence plates), then T2 licence-plate recognition, then T3 scene recognition, then T4 face recognition.
- Headline metric: the count of TD2 roadmap rows (spec section 13) marked done, with the commit it was measured on. Once D3 produces weights, add T1 holdout mAP (section 9 protocol) measured by `experiments/td2-eval`.
- Real progress: merged code, tests, measured evidence or a decision record that moves a ledger item forward. Negative results count when they are recorded.

## Setup
- Clone with `GIT_LFS_SKIP_SMUDGE=1`. Install the pinned toolchain (`rust-toolchain.toml`, 1.99.0, rustfmt and clippy) through rustup, then run `cargo fetch --locked` once. Every gate after that runs `--locked --offline`.
- Read first: `CONTRIBUTING.md` (binding: provenance, no Ultralytics code or weights, dependency review, a written decision before ABI/format/dependency changes, tests first), `MODEL_LICENSES.md`, `docs/decisions/`, the TD2 spec and its section 13 ledger, `docs/testing/m1-evidence.md`, and `.claude/skills/verify-factsyolo/SKILL.md`.
- `xtask verify` and `xtask dotnet` need .NET SDK exactly 10.0.403. Install it into a scratch folder as `verify-factsyolo` describes. If that is not possible, the hosted dotnet jobs cover it, and the PR says so.
- The workspace has 2 CPUs and no GPU. Long training or benchmark runs go to GitHub Actions.

## Gate
The local gate is the README "Build and verify" commands that this workspace can run. Warnings are errors.

```sh
cargo test --workspace --locked --offline
cargo test --workspace --locked --offline --release
cargo test --workspace --features contract-fixtures --target-dir target/contract-fixtures --locked --offline
cargo test --workspace --features contract-fixtures --target-dir target/contract-fixtures --locked --offline --release
cargo build -p facts-yolo --locked --offline --release
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo fmt --all -- --check
cargo run -p xtask --locked --offline -- c
cargo run -p xtask --locked --offline -- isolation
cargo run -p xtask --locked --offline -- check-docs
cargo run -p xtask --locked --offline -- verify   # needs dotnet 10.0.403
```

Also run the `experiments/cpu-detector` checks (`verify.py`, including the Rust 1.85.0 portable step) when that experiment is touched. For an experiment crate pinned to an older floor, run its `cargo +<floor> clippy` as well.

## Ledger
Section 13 "Roadmap" of the TD2 spec. `docs/plans/td2-roadmap.md` does not exist, so it must not be created. Every PR updates its row. Run notes live in PR bodies.

## Work order
Take the first row that is not done or blocked, in this order:
1. Stage B: the framework comparison (B2–B5) and then ADR 0002 (B6), which selects Burn or Candle. The comparison is predeclared and uses a fixed CPU proxy: same seeds, initialization and steps, a small conv+BN+SiLU+upsample net on original synthetic data, trained with each framework. It measures gradient agreement against the independent reference forward, plus throughput and peak memory on x64 and arm64 runners, and includes a dependency and licence review. PR #70 is on a `feat/` branch the task did not open, so it is not the task's to merge or push.
2. Stage C: research and record the dataset terms, category mapping and attribution in `MODEL_LICENSES.md`. The owner must approve before any download (PR #74 waits on that).
3. Stage D: the harness (D1, D2a, D2b1 and D2b2a are done; D2b2b augmentation is next, with D2b2b1 flip and photometric prepared on `claude/td2-d2b2b1-flip-photometric`), then reduced-scale training on free runners (D3, after B6 and C2).
4. Stage E: the native td2 backend, NEON/AVX2, aarch64 CI with real execution, and the no-networking CI gate (E2a is done). Then stage F onward.
- Keep implementation WIP at one open PR: finish existing work before opening another. While an item waits on owner approval, CI or external review, do non-conflicting reproduction, measurements or preparation on the next unblocked item without stacking implementation PRs (for example T2–T4 design work).
- T4 training stays blocked until a budget exists. TD1 is maintenance-only.
- Long training and benchmark jobs run as GitHub Actions workflows. They resume in chunks well under 6 hours. Each finished chunk's checkpoint is saved with `actions/cache`, keyed by run identity. Metrics are scalar JSON artifacts. Finished runs are analysed and their results recorded, including negative ones.

## Authority
- Branch prefix `claude/`; housekeeping branch `claude/docs-hygiene-YYYY-MM-DD`; harness install branch `claude/verify-skills`. Squash merge pinned to the head SHA.
- Never merge `codex/` branches or Dependabot PRs. Never push to or merge a branch this task did not open (for example `feat/…`).
- A PR may be merged only after:
  - every applicable hosted check on the **latest head** has completed and passed, hosted CodeQL included;
  - all review bots have finished and all blocking findings/review threads are resolved;
  - a qualified **non-author** reviewer has submitted a formal GitHub `APPROVED` review covering the latest head. `COMMENTED`, an agent's read-only PASS, an older-head review, or the author's own review do not satisfy this gate. Never bypass protection or self-approve.
- If Codex reviews this repo and has not reviewed the head, wait while the PR is younger than 12 hours: up to about 30 minutes in-run, otherwise until the next run. If Codex cannot review (failed review or usage limit), a genuinely independent agent can supply additional technical findings, but **cannot replace the formal GitHub non-author approval**. Do not repeatedly request a quota-blocked review; keep the PR open and work on non-conflicting preparation.
- First line of every PR body: "Opened by the FactsYOLO scheduled task."
- Merging deploys nothing. Releases, tags, packages and weights are owner-only.
- This recurring task must remain enabled at its existing frequency with no end date unless the owner explicitly authorizes a change. CI failures, network/tool issues, review limits, no progress and suppression of repeated notifications are not authorization to stop or slow it.

## Project rules
- The deployed library has no networking code, no telemetry and no file writes. It stores nothing. The host owns decoding, storage, galleries and watch-lists.
- Never upload customer or dataset images. Never commit or upload dataset pixels. Never publish weights trained on external data: no Releases, no committed weights, no public artifacts containing them.
- Never port or translate code from Ultralytics or any other detector, and never use their weights. Cite papers for ideas.
- No biometric attribute inference (age, gender, ethnicity, emotion). Face and plate features stay opt-in and off in default builds.
- Budget is zero. Use only free GitHub-hosted runners (including `ubuntu-24.04-arm` as the interim aarch64 host) and this workspace. Kaggle or Colab need the owner's account: prepare notebooks or scripts and ask. Never create accounts.
- Helper tools follow repo conventions: Rust `xtask`, or TypeScript. The docs checker lives in `xtask`.
- Verification boundary: the built library called through its C ABI (`xtask c`, `xtask verify`) and the C# binding, on original synthetic frames. Contract fixtures are compared and measured numbers recorded. Experiments are verified by their recorded metric JSON or by independent recounts, not by the author's summary. Dataset pixels and external weights are never verification inputs.
- Extra merge conditions: hosted CodeQL ran and passed, and no evidence threshold was weakened.
- Owner decisions the task must not take:
  - approving the recorded licence terms of any external dataset before download;
  - any spending or accounts;
  - publishing weights or SDK packages, tags or Releases;
  - licence or trademark changes.
- Already decided by the owner (2026-11-05):
  - on-prem deployment;
  - face and plate recognition are required, and scene, animal and object recognition are very important;
  - the framework is chosen by stage B evidence (ADR 0002);
  - the spec's other open items are adopted as written unless evidence shows them wrong (fix the spec in a PR and say why).

## Docs scope
- `README.md` indexes the docs; `docs/README.md` is not created. The doc set shrinks, never grows.
- The stage B comparison plan goes in a section of the TD2 spec. The stage C licence record goes in `MODEL_LICENSES.md`.
- ADR 0002 may be a new numbered file in `docs/decisions/` only if `CONTRIBUTING.md` requires numbered decision records for dependency or framework choices. Otherwise it is a section of the TD2 spec.
- Training and benchmark metric JSON artifacts are data, not docs.
- Housekeeping checks all Markdown for:
  - status statements that contradict main (README "Status", `docs/testing/m1-progress.md`, ledger vs merged PRs, spec "Owner decisions");
  - superseded kept evidence records, which are marked historical in place with a pointer to the current one and keep their content;
  - README and CONTRIBUTING reflecting the CCTV/edge, on-prem, TD2 direction without over-claiming.

## Report extras
- Training or benchmark jobs started, resumed or finished, and their numbers.
- Items re-scoped or blocked, and why.
