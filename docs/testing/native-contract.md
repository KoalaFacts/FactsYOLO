# Native contract verification: M1 task 1

Date: 2026-10-12. Original text: Apache-2.0.

## Implemented boundary

`fy_abi_query`, explicit C-layout records/statuses, header-prefix checks, and
per-call diagnostics. The public header declares only this implemented function.
No detector, model/session/result handle operations, backend, trained weights,
SDK publication, or whole-M1 completion is claimed.

## Observed evidence

| Stage | Commit / run | Actual result |
| --- | --- | --- |
| Initial RED | `f41999a`, [run 38231557097](https://github.com/KoalaFacts/FactsYOLO/actions/runs/38231557097), job 114752008185 | Rust 1.85 compiled; all 17 ABI assertions failed against the deliberate stub; exit 101. |
| Unicode RED | `4dfe660`, [run 38231981253](https://github.com/KoalaFacts/FactsYOLO/actions/runs/38231981253), job 114753403972 | `éX` with capacity 2 yielded an invalid 1-byte prefix instead of 0; test failed. |
| GREEN | `3a22992`, [run 38232271009](https://github.com/KoalaFacts/FactsYOLO/actions/runs/38232271009) | All three matrix jobs passed. Linux 1.99 logs show 2 unit + 17 integration tests passing in both Debug and Release; C dynamic/static calls, guard-page checks, exported-symbol check, clippy and fmt passed. |

Green matrix jobs: 114754359354 (Linux x64, Rust 1.85), 114754359601
(Linux x64, Rust 1.99), and 114754359867 (Windows x64, Rust 1.99).
The workflow executes the exact PR merge SHA, which includes the listed feature
commit plus its main base. These results were read from GitHub Actions, not
inferred from source inspection or claimed as local-container tests.

## Reproduce

Use the pinned rustup toolchain and run the commands in [README](../../README.md).
The [workflow](../../.github/workflows/native-contract.yml) contains the exact C
compiler/linker commands. It builds and runs a C consumer against both `.so` and
`.a` outputs on Linux, and checks that the only exported `fy_` symbol is
`fy_abi_query`. Windows runs Rust tests and builds the native libraries; Windows
C consumers, C#/Go/Node bindings and AOT integration have not been exercised yet.

Tests include actual 8-byte prefixes, initialized headers with uninitialized
output payloads, invalid versions/flags/alignment, untouched optional tails,
capacity-zero/error-buffer canaries, UTF-8 truncation, and 16 threads making
100 independent diagnostic calls each. The Linux C guard-page test places the
8-byte prefix immediately before inaccessible memory to catch overreads.

`FACTSYOLO_BUILD_ID` may be set at build time to a nonempty ASCII ID of at most
63 bytes. Unset local builds report `local-unversioned`; this first evidence run
uses that fallback, not an embedded Git provenance claim.

## Execution rulings and limitations

The implementation environment had no Rust compiler and could not resolve
GitHub for a clone. Work was isolated on a remote feature branch; GitHub Actions
provided real execution. Minimal read-only CI and one C smoke consumer were
brought forward from later tasks to establish evidence now. No secrets, package
publishing, branch-protection changes, or third-party runtime crates were added.

Descriptors do not prove that foreign pointers are live or allocations are large
enough: those remain explicit caller safety preconditions. Normal queries do
not allocate/retain image memory. The unwind guard does not contain aborts,
foreign-memory faults, or all allocator failures; full fault-injection/session
poisoning checks remain in task 5. Miri, full per-record cross-language layout
coverage, resource-loop benchmarks and real vision workloads remain later gates.
