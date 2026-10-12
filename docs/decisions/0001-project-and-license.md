# ADR 0001: FactsYOLO project and license baseline

Date: 2026-10-12

Status: project name, Rust implementation direction, and Apache-2.0 license
accepted by the project owner in the planning conversation. Detailed architecture,
implementation plan, and backend selection remain separate decisions.

## Intent

Develop an independent vision foundation that products written in C#, Go, and
TypeScript can embed. Preserve long-term control of the native interface, memory
lifetimes, and deployment boundaries. Implement detection models and their
training/inference logic rather than creating a renamed Ultralytics wrapper.

## Accepted decisions

The project is named FactsYOLO and its core implementation language is Rust.
Original project code, training and inference tooling, examples, SDKs, and
technical documentation use a single Apache-2.0 license. No revenue-based,
commercial-use, or host-application source-disclosure condition is added to that
license. Original third-party notices are preserved.

Model artifacts require explicit, separate license/provenance records, with
Apache-2.0 preferred for project-owned public weights where rights permit.
External code contributors use DCO 1.1 sign-off without copyright assignment.
Project branding is managed separately from the source-code license.

The decision applies to FactsYOLO only. It does not change the license or
commercial terms of FactsPDF, FactsML, or any other KoalaFacts project.

## Rationale and tradeoffs

The goal is broad adoption as an embeddable library, including proprietary
products. Apache-2.0 makes its copyright, contributor patent, redistribution,
and contribution terms explicit. It permits compliant commercial use and does
not require users to publish their modifications or send them upstream.
Consequently, competitors may also use the code without paying project royalties.

The standard license is not modified. Optional support, maintenance, hosted
services, and paid engineering can be considered separately; no such service
is asserted to exist yet. Consult the license text for its actual grants,
conditions, and disclaimers rather than treating this record as a legal opinion.

## Design direction, not implemented functionality

Prefer one shared Rust core, a versioned C ABI, and thin language bindings;
keep model and training logic separate from host-language packaging. Training
should not be a mandatory inference dependency. Public APIs should not expose
types from an unselected tensor framework.

The first technical specification should define ownership and release rules,
input borrowing, error/panic behavior, session concurrency, ABI evolution, model
artifact contracts, and native/browser deployment differences. These are design
requirements, not a frozen interface in this foundation.

## Decisions not made here

No specific YOLO-family architecture, compute framework, model format,
GPU backend, platform support matrix, package-registry name, version-support
promise, latency target, or memory target is approved by this license decision.
Burn remains a candidate for evaluation, not a selected dependency. C# Native AOT,
Go, Node, and browser support must each be validated before being advertised.

## Foundation completion criteria

The standard license, attribution, project-status description, contribution/DCO
policy, third-party record, model-artifact policy, and branding guidance are
present and internally consistent. This milestone does not include Rust code,
a working detector, weights, SDK releases, benchmark results, automated DCO
checks, or branch protection.

## References

- [Apache License 2.0](../../LICENSE)
- [Apache licensing FAQ](https://www.apache.org/foundation/license-faq.html)
- [DCO 1.1](../../DCO), canonical source: https://developercertificate.org/
- [Model artifact policy](../../MODEL_LICENSES.md)
- [Third-party notices](../../THIRD_PARTY_NOTICES.md)
