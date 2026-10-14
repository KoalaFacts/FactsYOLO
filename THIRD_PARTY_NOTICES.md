# Third-party notices and dependency policy

## Scope

Apache-2.0 applies to original FactsYOLO material; it does not relicense included
third-party content or libraries. This file is an initial record, not a completed
software bill of materials for a future binary release.

## Included third-party documents

| Path | Source | Original notice / terms |
| --- | --- | --- |
| `LICENSE` | https://www.apache.org/licenses/LICENSE-2.0.txt | Standard Apache License 2.0 text, reused as permitted by the Apache Software Foundation. |
| `DCO` | https://developercertificate.org/ | Copyright (C) 2004, 2006 The Linux Foundation and its contributors. The document permits verbatim copying and distribution, not modification. |

License and DCO texts are reproduced from the canonical sources; they are not
FactsYOLO-authored license terms.

## Runtime and build dependencies

The M1 Cargo lockfile contains only the project-owned `facts-yolo` and `xtask`
workspace packages, with no external Cargo dependencies. The implementation
uses Rust's standard library. Rust toolchains, their bundled components, and
platform/system link dependencies retain their own terms; zero external Cargo
dependencies is not a claim of zero binary redistribution obligations.

Rust 1.99.0 is the pinned build toolchain and 1.85.0 is tested as the minimum.
Git, GCC and the GitHub-hosted Linux/Windows build environments are development
tools, not bundled FactsYOLO runtime packages. Workflows request read-only repository permissions.
The development-only pinned Actions and safety tooling are recorded below. The default M1 library adds no native ML runtime, GPU SDK, image/video codec,
or downloaded model. The independent CPU experiment now evaluates Burn and
Candle; see its separate dependency review below.

Development-only tools added/used by the M1 acceptance gate:

- .NET SDK 10.0.401 / runtime 10.0.12 from Microsoft; the unpublished consumer
  uses no explicit PackageReference. SDK Native AOT/compiler/runtime packs and
  their transitive components retain Microsoft's bundled notices. See the
  [.NET source/license](https://github.com/dotnet/runtime/blob/main/LICENSE.TXT)
  and [third-party notices](https://github.com/dotnet/runtime/blob/main/THIRD-PARTY-NOTICES.TXT).
- `actions/setup-dotnet` at `26b0ec14cb23fa6904739307f278c14f94c95bf1` and
  `actions/upload-artifact` at `ea165f8d65b6e75b540449e92b4886f43607fa02` use
  MIT licenses in their upstream repositories. Their bundled JavaScript/toolkit
  dependencies and notices remain part of the Actions, not FactsYOLO packages.
- Rust `nightly-2026-10-13` (rustc 1.101.0-nightly, e9bbadf68) with Miri and
  rust-src is development-only. [Miri](https://github.com/rust-lang/miri) is
  MIT OR Apache-2.0; sysroot preparation builds Rust's bundled standard-library
  dependencies under their upstream notices. It adds no dependency to Cargo.lock.
- GNU GCC/binutils and platform headers compile the i686 object probes. These
  build tools are not redistributed by this repository; their own licenses and
  runtime-library exceptions remain applicable. No 32-bit runtime is published.

No third-party source or license text from these tools is vendored here. This
inventory does not certify a distributable binary SBOM; any future packaging
must account for the exact linked runtime/native components and their notices.

Before adding a dependency, record its exact version/source, SPDX license where
available, required notices, transitive/native components, build features, and
whether it is included in shipped artifacts or used only during development.
Retain applicable license copies and attribution in the source and release
packages as required. Native GPU libraries and codecs need separate reviews;
a permissive wrapper does not determine their redistribution terms.

Before binary publication, produce a versioned SBOM that includes toolchain and
native components and review their license/notice obligations. No binary release
or completed binary-distribution license audit is claimed by this task.

## Reference

Apache's licensing FAQ explains reuse of the license and treatment of third-party
material: https://www.apache.org/foundation/license-faq.html.

## Independent CPU detector experiment

The development-only workspace under `experiments/cpu-detector` pins Burn0.22.0,
Candle0.11.0 and their transitive graph without changing root Cargo.lock. See
[the dependency review](experiments/cpu-detector/DEPENDENCIES.md) and
[exact selected graph](experiments/cpu-detector/dependencies.json), including
MPL-2.0 colored3.1.1 and bundled Oniguruma C/two-clause BSD notices. It is not
an all-MIT/Apache or entirely Rust dependency graph. No compiled experiment,
weight file or dataset is publicly uploaded by the workflow. Source dependencies
are fetched from their registry packages with upstream notices intact. Public
binary packaging still requires license copies, source-offer compliance where
applicable, and a binary SBOM; this source-only evaluation does not grant it.
