# FactsYOLO

An independent, Rust-first computer-vision foundation for products written in
C#, Go, and TypeScript.

**Status: native infrastructure under implementation.** The repository contains
a buildable Rust library, a C header, the experimental `fy_abi_query` entry point,
descriptor/diagnostic boundary tests, and a C dynamic/static smoke consumer.
It does not yet contain a working detector, training implementation, model
weights, published SDK, or detector benchmarks. Other records in the header
prepare later tasks; they do not imply that model/session APIs already exist.

## Build and verify

Install the pinned Rust toolchain through rustup, then run:

```sh
cargo test --workspace --locked --offline
cargo test --workspace --locked --offline --release
cargo build -p facts-yolo --locked --offline --release
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo fmt --all -- --check
```

The native library is built as `rlib`, `cdylib`, and `staticlib`; Cargo publication
is disabled. Task 1 CI exercises Linux x64 with Rust 1.99.0 and the 1.85.0 floor,
and Windows x64 with Rust 1.99.0. Linux also executes C callers against the dynamic
and static libraries. These checks are not full vision-platform certification.
See [the test record](docs/testing/native-contract.md) for exact evidence and limits.

## Purpose

Build one reusable native vision core with explicit memory ownership, documented
execution and deployment boundaries, and thin language bindings. YOLO-style
object detection is the first model task, not a requirement to recreate every
computer-vision feature at once.

FactsYOLO will implement its own detection models and training/inference logic;
it is not intended to be merely a wrapper around another YOLO product.
It will not depend on Ultralytics software or distribute its model weights.
General-purpose tensor and hardware-compute libraries can be evaluated separately.
No compute framework has been selected or added at this stage.

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
