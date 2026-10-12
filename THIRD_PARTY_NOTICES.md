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

The task 1 Cargo lockfile contains only the project-owned `facts-yolo` and `xtask`
workspace packages, with no external Cargo dependencies. The implementation
uses Rust's standard library. Rust toolchains, their bundled components, and
platform/system link dependencies retain their own terms; zero external Cargo
dependencies is not a claim of zero binary redistribution obligations.

Rust 1.99.0 is the pinned build toolchain and 1.85.0 is tested as the minimum.
Git, GCC and the GitHub-hosted Linux/Windows build environments are development
tools, not bundled FactsYOLO runtime packages. The workflow uses no third-party
Actions and requests read-only repository permissions. No native ML runtime,
GPU SDK, image/video codec, or downloaded model has been added. Burn remains an
evaluation candidate, not an approved or bundled dependency.

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
