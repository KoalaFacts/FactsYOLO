# FactsYOLO

An independent, Rust-first computer-vision foundation for products written in
C#, Go, and TypeScript.

**Status: native infrastructure under implementation.** The repository contains
a buildable Rust library, a C header, ABI/version diagnostics, checked borrowed
RGB/BGR/RGBA image views, model/session ownership, synchronous execution and owned
result contracts, and single-session admission with recoverable panic containment.
Execution is exercised using explicitly test-only fixtures, not trained detectors.

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
```

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
private contract fixtures remain separately marked TEST_ONLY. Linux x64 actual
trained-artifact C shared/static and C# JIT/AOT execution is verified; Windows
has TD1 Rust contract coverage, not yet a real-artifact C/C# support claim.

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
- A versioned C ABI for native consumers, with thin C# and Go SDKs.
- Node-API bindings for TypeScript on Node.js and a separate WASM build for
  browsers, sharing core logic where supported rather than promising identical
  native and browser binaries.
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

## Contributing and project decisions

See [CONTRIBUTING.md](CONTRIBUTING.md) for provenance requirements and DCO sign-off,
[TRADEMARKS.md](TRADEMARKS.md) for project-identity guidance, and
[the project and license decision](docs/decisions/0001-project-and-license.md)
for the accepted baseline and outstanding technical decisions.
