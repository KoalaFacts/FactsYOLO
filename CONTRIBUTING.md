# Contributing to FactsYOLO

## Contribution license and authority

Submit original contributions under Apache-2.0. Contributors retain their
copyright; this project does not require copyright assignment or a separate
contributor license agreement at this foundation stage. The contribution and
patent terms in [LICENSE](LICENSE) apply.

Only submit material you have the right to contribute. Obtain any necessary
employer or client authorization. Third-party content must retain its applicable
license and notices and be explicitly identified in the pull request.

## Developer Certificate of Origin

External contributors must certify their contributions using the
[Developer Certificate of Origin 1.1](DCO). Add your own `Signed-off-by` trailer to
each contribution commit, normally with:

```sh
git commit -s
```

Read the DCO before signing. Use your own configured identity, not a maintainer's
or another person's identity. A sign-off is an authorship/provenance certification,
not a cryptographic commit signature or a copyright transfer.

DCO compliance is a manual review requirement at this stage. No DCO check,
branch-protection setting, or other automated DCO enforcement is installed by
these documents. Missing sign-offs on external contribution commits block
acceptance until the contributor supplies them.

## Independent implementation and provenance

Do not submit Ultralytics implementation code, translated/ported copies of it,
model weights, or artifacts derived from those weights. A change of language or
file format does not establish independent provenance.

For algorithmic work, document the papers, specifications, and any other sources
used. Distinguish an independently implemented published method from a source-code
port. Provide references and record any code reuse and its license.

For every new dependency, identify the exact version, upstream source, license,
transitive dependencies, native components, and redistribution obligations.
The default distributed core and SDKs should use dependencies whose terms permit
closed-source commercial integration. Any exception needs an explicit dependency
review before introduction; do not silently add license restrictions to the
published Apache-2.0 core.

AI-assisted contributions remain the submitting contributor's responsibility.
Disclose material generated content and review correctness and provenance;
never invent author certifications or sign-offs.

## Change process

Keep changes focused and use a pull request. State the problem, scope, evidence,
and remaining limitations. Changes to the public ABI, memory ownership, threading,
model format, dependency policy, or licensing need a written decision/specification
before implementation.

Preserve the single-core design: bindings adapt the public contract rather than
independently reimplementing preprocessing, model output interpretation, or
postprocessing. Review and merge the current implementation task before starting
the next task in the accepted M1 plan.

## Verification at this stage

Run the workspace Debug and Release tests, native build, clippy, and formatting
commands in [README](README.md), plus the applicable C smoke commands in the
[workflow](.github/workflows/native-contract.yml). Include actual command output
and distinguish local execution from remote CI. CI is not DCO enforcement.

Write failing regression tests before behavior changes, then verify the whole
suite. Do not report a detector or SDK as implemented because the native query
and ABI records exist. Public support/performance claims require reproducible
results on the named build, model, hardware, and runtime configuration. The
current [evidence record](docs/testing/native-contract.md) covers only task 1.

Also check local Markdown links, consistency of scope, license notices, and
provenance statements when changing documentation.

## Model or data contributions

Follow [MODEL_LICENSES.md](MODEL_LICENSES.md). Do not upload weights or datasets
without an explicit, reviewed license/provenance record. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the dependency and attribution
recording requirements.

## Reference

The canonical DCO is available at https://developercertificate.org/.
The copy in this repository retains its own notice and verbatim-copy terms.
