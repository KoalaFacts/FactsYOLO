---
name: verify-factsyolo
description: How to exercise FactsYOLO for real (built library through its C ABI, the C# consumer, the experiments) and capture evidence before claiming a change works.
---

# Verify FactsYOLO

The real boundary is the built `cdylib`/`staticlib` driven through
`include/facts_yolo.h` by C consumers and by the C# consumer, on original
synthetic frames. Rust unit tests are necessary, not sufficient. Paste actual
command output into the PR and say which runs were local and which were CI.

## Setup

- Rust: `rust-toolchain.toml` pins 1.99.0 (rustfmt, clippy) via rustup. The
  TD1 experiment also needs 1.85.0 (the `portable-msrv` step inside
  `experiments/cpu-detector/verify.py`).
- Clone with `GIT_LFS_SKIP_SMUDGE=1`, then `cargo fetch --locked` once; every
  gate after that runs `--locked --offline`. The TD1 experiment has its own
  lock: `cargo fetch --manifest-path experiments/cpu-detector/Cargo.toml --locked`.
- `xtask c` / `xtask exports`: Linux or Windows x64, a C11 compiler (`cc`, or
  MSVC `cl` in a VS developer shell) and `nm` (`dumpbin` on Windows).
- `xtask dotnet`: .NET SDK exactly 10.0.403 (`sdk/dotnet/global.json`,
  `tests/dotnet/global.json`, `rollForward: disable`) plus nuget.org access for
  restore and the Native AOT ILCompiler package.
- `xtask miri`: `rustup toolchain install nightly-2026-10-13 --profile minimal --component miri`.
- `xtask layout32`: `rustup target add i686-unknown-linux-gnu --toolchain 1.99.0`,
  multilib C headers (Ubuntu `gcc-multilib`, or `CC_I686`) and `objcopy`.
- `xtask verify` runs a preflight first and fails on any missing mandatory tool
  (rustc, cargo, git, dotnet 10.0.403, C compiler, symbol tool).
- Nearest real harnesses: `cargo run -p xtask --locked --offline -- c` (C
  consumers against freshly built normal and fixture libraries) and
  `cargo run -p xtask --locked --offline -- dotnet --mode jit` (C# consumer).
- `xtask` resolves the repo root from its own manifest and writes to fixed
  `target/` subdirectories (`target/normal`, `target/contract-fixtures`,
  `target/c-conformance`, `target/dotnet-conformance`, `target/m1-evidence`).
  `experiments/cpu-detector/verify.py` reads binaries from
  `experiments/cpu-detector/target/release`, so do not set `CARGO_TARGET_DIR` for it.

## Feature map

Bare `src/`, `examples/` and `tests/<name>.rs` paths below are relative to
`crates/facts-yolo/`; paths starting `tests/c/`, `tests/compat/`, `tests/dotnet/`,
`tests/layout32/` and every other path are relative to the repo root.

| feature | code entry points | how to exercise it for real | evidence to capture | neighbours at risk |
|---|---|---|---|---|
| C ABI / header | `include/facts_yolo.h`; `crates/facts-yolo/src/ffi/{exports,types}.rs`; `tests/c/public-symbols.txt`; `examples/abi_layout.rs` | `cargo run -p xtask --locked --offline -- exports`, then `-- c` (verified here) | "11 exact exports" (normal) / "15" (fixture), negative control rejected, "all 113 compiler-produced layout records agree" for current and frozen headers, "C conformance passed"; logs under `target/c-conformance/` | frozen header `tests/c/frozen`, C# `NativeTypes.cs`/`Layout.cs`, `tests/layout32/probe.*`, `sdk/dotnet` native export check |
| Image views | `src/image.rs`, `src/ffi/descriptors.rs` | `cargo test -p facts-yolo --test images --test descriptor_corpus --locked --offline`, then `xtask c` and `xtask miri` (verified here) | test counts; Miri pass line | TD1 stride/RGB/BGR/RGBA handling, descriptor corpus, C# `Detect` buffers |
| Model/session ownership | `src/model.rs`, `src/session.rs`, `src/session/control.rs`, `src/result.rs`, `src/result/owned.rs`, `src/ffi/handles.rs` | `cargo test -p facts-yolo --test lifetimes --test results_ffi --features contract-fixtures --target-dir target/contract-fixtures --locked --offline`; `xtask c` runs `tests/c/ownership.c` and `threads.c` (verified here) | "1000 model/session lifecycle cycles passed", "50 barrier-confirmed BUSY cycles passed" | resources probe, `concurrency.rs`, C# SafeHandle finalization, TD1 "results outlive session" |
| Contract fixtures | `src/fixtures.rs`, `src/ffi/testing.rs`, `tests/fixtures/facts_yolo_test.h` (`fy_test_*`, `FY_CAP_TEST_ONLY`) | `cargo test --workspace --features contract-fixtures --target-dir target/contract-fixtures --locked --offline` (verified here) | all suites ok; fixture library exports exactly the 4 extra `fy_test_*` symbols | normal-export isolation (`xtask exports` negative control), resources, Miri |
| TD1 opt-in backend | `--features td1`; `src/model.rs` (51,284-byte `FYTDv001` check); `experiments/cpu-detector/model` (`fy-tiny-model`); `tests/c/td1.c`; `tests/compat/td1_v1_vectors.rs` | `cargo test -p facts-yolo --features td1 --locked --offline` and `cargo build -p facts-yolo --features td1 --locked --offline --release`; trained artifacts through C/C#: `verify_native.py` (see TD1 experiment) (verified here) | `tests/td1.rs` 10 passed (the default build runs 0 of them) | `docs/testing/td1-v1-compatibility.md` pinned bytes, Rust 1.85 portable floor, backend-free default |
| C# binding JIT/AOT | `tests/dotnet/FactsYOLO.Smoke` (consumer used by xtask); `sdk/dotnet/FactsYOLO` (package, exercised by `sdk-install.yml`) | `cargo run -p xtask --locked --offline -- dotnet --mode jit`, then `--mode aot` (verified here) | "normal jit: 6 passed", "fixture jit: 14 passed", same for aot, ".NET aot linux-x64: normal and fixture executables passed" | layout records, export sets, SafeHandle/dispose semantics, `tests/sdk-install`, `tests/dotnet11` |
| Resources sampling | `examples/resource_probe.rs` (fixture-only), `xtask/src/resources.rs` | `cargo run -p xtask --locked --offline -- resources` (verified here) | rss/private first vs last median within 16 MiB budget, "live objects and Rust allocation bytes returned to baseline", evidence dir in `target/m1-evidence/resources-*` | ownership/result lifetimes, fixture counters |
| Safety | `xtask/src/safety.rs`, `tests/layout32/` | `cargo run -p xtask --locked --offline -- miri`; `-- layout32` (verified here). TD1 Miri: CI job `miri-td1` | Miri pass line + `outcome.txt`; "123 scalars (113 records) agree for current/frozen C and Rust" | any `unsafe`, descriptor/record layout, pointer width |
| Deployed-library isolation (TD2 E2a) | `xtask/src/isolation.rs`; CI step in `.github/workflows/c-conformance.yml` (Linux) | `cargo run -p xtask --locked --offline -- isolation` (verified here, Linux) | per feature set: "N local source files scanned", "N imports checked, 0 forbidden", "N std symbols checked, 0 forbidden", final "isolation gate passed" | any new dependency or `std::fs`/`net`/`process`/`extern` use in `crates/facts-yolo` or `experiments/cpu-detector/model` fails it; `nm` must be on PATH; a `strip` release profile makes it fail (no std symbols) |
| Docs gate | `xtask/src/docs.rs`; README documentation index | `cargo run -p xtask --locked --offline -- check-docs` (verified here) | "docs check passed" | new Markdown must be indexed from README; inline-code paths under `crates/`, `docs/`, `experiments/` ... must exist; hidden dirs (this one) are skipped |
| TD1 experiment | `experiments/cpu-detector/` (`verify.py`, `verify_native.py`, `paired_trained_native.py`, `model`, `evaluate`) | `python3 experiments/cpu-detector/verify.py experiments/cpu-detector/target/evidence` (new dir), then `verify_native.py <evidence>/burn target/td1-evidence-burn` and the same for `candle`; Python gate unit tests: `python3 -m unittest discover -s experiments/cpu-detector -p 'test_*.py'` (verified here) | the JSON reports and `commands.json` in the evidence dir, `target/td1-evidence-*/summary.json`; CI job `train-save-reload` uploads them | native TD1 loader, artifact format, C/C# TD1 consumers |
| TD2 Stage B oracle | `experiments/td2-framework/reference_oracle.py` | as `.github/workflows/td2-framework-oracle.yml` job `independent-reference`: `python3 -m unittest discover -s experiments/td2-framework -p 'test_reference_oracle.py' -v`, `python3 experiments/td2-framework/reference_oracle.py --output td2-b-oracle.json`, then assert `framework_selected` and `training_executed` are false and `outputs` has 32 values (verified here) | the JSON itself (inputs, loss, finite-difference gradients) | any future Burn/Candle comparison that uses this reference |
| TD2 D1 evaluator | `experiments/td2-eval/src/lib.rs` (`evaluate`), frozen cases `experiments/td2-eval/tests/data/pycocotools-crosscheck.txt` | `cargo test -p td2-eval --locked --offline` (verified here, 10/10); for a real cross-check re-score the frozen cases with pycocotools in a scratch venv (never in the repo) and compare every summary and per-class number | test counts; max difference vs pycocotools (0.0 at merge, PR #72) | Stage D reports against spec section 9 floors |
| TD2 D2b1 layer table | `experiments/td2-arch/src/lib.rs` (`table`, `analyze`, `HEADS`) | `cargo test -p td2-arch --locked --offline` (verified here, 13/13, 2026-11-08); also on 1.85 clippy (see Traps) | frozen totals 1,757,463 parameters / 1,454,700,900 FLOPs at S=320, C=120 vs budgets 3.0 M / 2.5 G | FYTDv002 layer kinds (E1b), D3 training, spec section 6 numbers |

`experiments/td2-plates` (synthetic plates, PR #73) is not on main yet; add its row when it merges.

## Gate

Local gate, exactly as README "Build and verify" lists it:

```sh
cargo fetch --locked
cargo test --workspace --locked --offline
cargo test --workspace --locked --offline --release
cargo test --workspace --features contract-fixtures --target-dir target/contract-fixtures --locked --offline
cargo test --workspace --features contract-fixtures --target-dir target/contract-fixtures --locked --offline --release
cargo build -p facts-yolo --locked --offline --release
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo fmt --all -- --check
cargo run -p xtask --locked --offline -- c
cargo run -p xtask --locked --offline -- isolation
cargo run -p xtask --locked --offline -- verify
cargo run -p xtask --locked --offline -- check-docs
```

Add the rows above that your change touches (TD1, Miri, layout32, experiments).

## Traps

- The default build is backend-free: `cargo test` compiles `tests/td1.rs` to
  zero tests. TD1 is only exercised with `--features td1` and by the
  experiment's native verifier.
- Contract fixtures are TEST_ONLY (`FY_CAP_TEST_ONLY`, `fy_test_*` exports in
  their own target dir). A fixture pass says nothing about a real model.
- `verify --preflight` is a tool check, not a pass; a `target/m1-evidence`
  manifest is an inventory, not a pass. Look for `outcome.txt` = `PASS`.
- A Rust panic backtrace in the .NET log is the expected `native-faults`
  control; the pass/fail counts and the final line decide.
- CI jobs blocked by billing finish with zero steps executed; that is neither
  green nor red. Open the job and confirm steps ran before quoting it.
- Experiments are verified by their recorded metric JSON and command logs
  (`summary.json` with `passed: true`, `source_dirty: false` and the HEAD
  commit), not by a summary line or a PR description.
- `verify.py` refuses a dirty tree at start and end, untracked files included
  (e.g. a new `.claude/` file). Commit first or run from a clean worktree;
  evidence directories must be new.
- Never use dataset pixels or external/pretrained weights as verification
  inputs. Use original synthetic frames and artifacts trained in the same job.
- Unit tests of a Rust function do not show that the shipped library, header
  and C# consumer agree; run `xtask c` and `xtask dotnet` for that.

## Last verified

2026-11-06 against origin/main `90d99981099fc5977234c24733f87705dd0d8ea7`
(Linux x64, 2 CPUs; every row and the gate above were run there).
