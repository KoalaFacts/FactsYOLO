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

No Rust crates, native ML runtimes, GPU SDKs, image/video codecs, or precompiled
libraries have been added in this documentation foundation. In particular, Burn
is an evaluation candidate, not an approved or bundled dependency.

Before adding a dependency, record its exact version/source, SPDX license where
available, required notices, transitive/native components, build features, and
whether it is included in shipped artifacts or used only during development.
Retain applicable license copies and attribution in the source and release
packages as required. Native GPU libraries and codecs need separate reviews;
a permissive wrapper does not determine their redistribution terms.

Generate a versioned dependency inventory/SBOM when executable artifacts exist.
Do not describe a future dependency set as audited based on this initial file.

## Reference

Apache's licensing FAQ explains reuse of the license and treatment of third-party
material: https://www.apache.org/foundation/license-faq.html.
