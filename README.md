# FactsYOLO

An independent, Rust-first computer-vision core with a C ABI and thin language
bindings (C# first; Go and TypeScript are not part of the current TD2 tasks), for customers to embed in their
own CCTV cameras, NVRs and edge devices and operate on premises.

**Status: native infrastructure under implementation.** The repository contains
a buildable Rust library, a C header, ABI/version diagnostics, checked borrowed
RGB/BGR/RGBA image views, model/session ownership, synchronous execution and owned
result contracts, and single-session admission with recoverable panic containment.
Native contract tests use explicitly test-only fixtures. The optional `td1` feature
also supports trained artifacts from the separate [synthetic CPU experiment](docs/testing/cpu-detector.md)
described below; the default build remains backend-free.

C consumers verify layouts, exports, results, errors and concurrent admission on
Linux and Windows. An unpublished C# smoke adapter uses LibraryImport, SafeHandle
and scoped lifetime leases; its JIT and Native AOT executables are tested on both
platforms. These are conformance tools, not released C#/Go/TypeScript SDKs.

**The default native build has no real model backend.** It reports
zero backend capabilities; validated `fy_model_open` requests return
`FY_UNSUPPORTED_MODEL`. Even test builds do not accept fixture bytes as model
artifacts. There is no trained weight release, published SDK, or production detector
accuracy/performance claim. A separate [CPU training experiment](docs/testing/cpu-detector.md)
implements original TD1 training with Burn Flex and Candle, then loads its data-only
artifact in a portable Rust reference. The optional `td1` native feature now
loads that artifact through the same C ABI and unpublished C# JIT/AOT adapter;
see [real-artifact native evidence](docs/testing/td1-native.md).

An experimental [local C# SDK installation gate](docs/testing/dotnet-sdk.md)
packages the existing ABI for isolated net10.0 consumers. Its temporary test
package is not published; Linux/Windows assets must form one identically hashed
package before cross-platform installation acceptance can be claimed.

An independent, optional [.NET 11 RC3 consumer gate](docs/testing/dotnet11-compatibility.md)
tests the existing net10.0 experiment package without changing its target or
establishing a stable SDK support promise.

The [TD2 and CCTV vision task roadmap](docs/superpowers/specs/2026-11-05-td2-general-detector-design.md)
is the accepted design for the next model work: general detection (objects,
animals, people, vehicles, faces, licence plates), then licence-plate, scene
and face recognition, for on-premises CCTV cameras, NVRs and edge devices. It
is a plan, not implemented functionality; TD1 is maintenance-only.

## Build and verify

Install the pinned Rust toolchain through rustup, fetch the locked optional
dependency metadata once, then run the offline gates:

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
cargo run -p xtask --locked --offline -- verify
cargo run -p xtask --locked --offline -- check-docs
```

`check-docs` fails on broken relative links or anchors, Markdown files missing
from the [documentation index](#documentation), and paths under the repository's
top-level source directories (`crates/`, `docs/`, `experiments/`, ...) named in
inline code that do not exist.

With the pinned .NET SDK and native tools installed, execute the managed adapter:

```sh
cargo run -p xtask --locked --offline -- dotnet --mode jit
cargo run -p xtask --locked --offline -- dotnet --mode aot
```

The runner supports the current Linux x64 or Windows x64 host. Native AOT builds
are executed, not just compiled. See [C# conformance](docs/testing/dotnet-conformance.md)
for required tools, actual evidence and deployment limitations.

The unified `verify` command runs every mandatory local x64 gate, including
resource samples and executed .NET JIT/AOT. Dedicated `miri` and `layout32`
commands supply interpreted Rust safety and compile-only i686 pointer-width
coverage. Missing tools fail verification. See [M1 evidence and limits](docs/testing/m1-evidence.md)
for exact commands, sampling budgets and the remaining real-detector design gate.

Build the opt-in TD1 CPU model path without any training framework in the library:

```sh
cargo build -p facts-yolo --features td1 --locked --offline --release
cargo test -p facts-yolo --features td1 --locked --offline
```

TD1 requires64x64 RGB/BGR/RGBA views, reports CPU backend1/F32/one thread and
accepts only the checked FYTDv001 artifact. The default remains backend-free;
private contract fixtures remain separately marked TEST_ONLY. Real-artifact CI
exercises C shared/static consumers on Linux x64 and C DLL consumers on Windows
x64, plus C# JIT/AOT on both. Each host trains and consumes its own ephemeral
artifacts; Windows static C and cross-host bitwise training identity are not claimed.

The [TD1 v1 compatibility gate](docs/testing/td1-v1-compatibility.md) pins synthetic
historical bytes and boundary behavior independently of the current serializer.
It preserves the experimental format and ABI; it does not establish a stable SDK release.

The native library builds as `rlib`, `cdylib`, and `staticlib`; Cargo publication
is disabled. Rust CI exercises Linux x64 with 1.99.0 and the 1.85.0 floor, and
Windows x64 with 1.99.0. C conformance runs Linux shared/static and Windows DLL
consumers. Exact normal/test export sets are checked; test builds have their own
output directory and must not be shipped as defaults.

See [the initial ABI test record](docs/testing/native-contract.md),
[the ownership test record](docs/testing/model-session-ownership.md), and
[C# conformance](docs/testing/dotnet-conformance.md) for commands, evidence and
limitations. Historical records describe their tested revision. None of these
checks establishes complete vision-platform certification or production
memory/accuracy targets.

## Purpose

Build one reusable native vision core with explicit memory ownership, documented
execution and deployment boundaries, and thin language bindings. YOLO-style
object detection is the first model task, not a requirement to recreate every
computer-vision feature at once.

FactsYOLO will implement its own detection models and training/inference logic;
it is not intended to be merely a wrapper around another YOLO product.
It will not depend on Ultralytics software or distribute its model weights.
General-purpose tensor and hardware-compute libraries can be evaluated separately.
Burn Flex and Candle are experimental training candidates in an independent
workspace. No production compute framework has been selected.

## Intended design direction

- One Rust implementation of the shared vision logic.
- A versioned C ABI for native consumers, with a thin C# SDK first. Go and
  TypeScript bindings may follow later; browser/WASM builds are not planned.
- On-premises deployment: the deployed library has no networking code, writes
  no files and stores nothing; the host owns decoding, storage and business logic.
- Training tooling separate from inference deployment dependencies.
- Explicit resource lifetimes, bounded workloads, reported device selection,
  and tested numerical tolerances across bindings and backends.

The development ABI is not a stable release. Model architecture, broader platform
support, and performance targets require implementation evidence before being
advertised. Rust does not by itself guarantee a dependency-free binary,
zero-copy execution, or the absence of runtime GPU compilation.

## License and provenance

Original FactsYOLO source code, SDKs, tools, examples, and technical documentation
are licensed under the **Apache License 2.0**, except where a file has an explicit
third-party notice. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The project adds no commercial-use fee, revenue threshold, or requirement to
publish an embedding application's source code. This summary does not replace
or modify the Apache license conditions.

Model weights and datasets require their own explicit records; the source-code
license is not a blanket permission for arbitrary model or data artifacts.
No weights or datasets are included. See [MODEL_LICENSES.md](MODEL_LICENSES.md).
Third-party notices, including the DCO document, retain their original terms;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Documentation

Current plan and design:

- [TD2 design and roadmap](docs/superpowers/specs/2026-11-05-td2-general-detector-design.md): accepted CCTV task family, first detector, stages; [section 13](docs/superpowers/specs/2026-11-05-td2-general-detector-design.md#13-roadmap) is the work ledger.
- [TD2 Stage B oracle](experiments/td2-framework/README.md): independent numerical reference for the framework comparison.

Decisions and policy:

- [Project and licence decision](docs/decisions/0001-project-and-license.md), [contributing](CONTRIBUTING.md), [model licences](MODEL_LICENSES.md), [third-party notices](THIRD_PARTY_NOTICES.md), [trademarks](TRADEMARKS.md), [agent instructions](CLAUDE.md).

M1 native foundation (complete; historical design and evidence):

- [Native contract design](docs/superpowers/specs/2026-10-12-native-contract-design.md) and [implementation plan](docs/superpowers/plans/2026-10-12-native-foundation.md).
- Evidence by task: [1 ABI](docs/testing/native-contract.md), [2 image views](docs/testing/image-views.md), [3 ownership](docs/testing/model-session-ownership.md), [5 ledger](docs/testing/m1-task5-ledger.md), [6 ledger](docs/testing/m1-task6-ledger.md), [7 C#](docs/testing/dotnet-conformance.md), [8 resources and safety](docs/testing/m1-evidence.md); [execution ledger](docs/testing/m1-progress.md).

TD1 experiment (maintenance-only; historical evidence):

- [CPU detector design](docs/superpowers/specs/2026-10-14-cpu-detector-design.md), [training experiment](docs/testing/cpu-detector.md), [experiment dependencies](experiments/cpu-detector/DEPENDENCIES.md).
- [Native integration design](docs/superpowers/specs/2026-10-14-td1-native-design.md) and [evidence](docs/testing/td1-native.md); [v1 compatibility gate](docs/testing/td1-v1-compatibility.md).
- Diagnostics: [four-side paired](docs/testing/td1-four-side-paired.md), [four-side disjoint](docs/testing/td1-four-side-disjoint.md), [cross-generator](docs/testing/td1-cross-generator.md), [background-blind](docs/testing/td1-background-blind.md), [spatial alias](docs/testing/td1-spatial-alias.md) (why TD1 is frozen).

SDK packaging and testing:

- [Local SDK design](docs/superpowers/specs/2026-10-15-local-dotnet-sdk-design.md), [plan](docs/superpowers/plans/2026-10-15-local-dotnet-sdk.md) and [installation gate](docs/testing/dotnet-sdk.md); [package README](sdk/dotnet/README.md); [Linux binary audit](docs/testing/linux-sdk-redistribution.md); [.NET 11 consumers](docs/testing/dotnet11-compatibility.md).

## Contributing and project decisions

See [CONTRIBUTING.md](CONTRIBUTING.md) for provenance requirements and DCO sign-off,
[TRADEMARKS.md](TRADEMARKS.md) for project-identity guidance, and
[the project and license decision](docs/decisions/0001-project-and-license.md)
for the accepted baseline and outstanding technical decisions.
