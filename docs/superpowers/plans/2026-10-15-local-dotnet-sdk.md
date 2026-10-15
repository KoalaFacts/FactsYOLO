# Local .NET SDK Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans.
> The user authorized continued inline implementation and a draft PR.

**Goal:** Install one local dual-RID test package outside the repository and
exercise existing TD1 models with JIT and NativeAOT on Linux/Windows x64.

**Architecture:** Thin public C# wrappers preserve the existing native contract.
Two native asset stages feed one package; detached consumers install that exact
package. Existing ABI/fixture/CPU gates remain independent.

**Tech Stack:** Rust 1.99.0 (existing MSRV unchanged), .NET 10.0.401/10.0.12,
Python standard library, Linux x64 and Windows x64/MSVC.

**Spec:** ../specs/2026-10-15-local-dotnet-sdk-design.md

## Global Constraints

- No ABI, model format, training/inference core or dependency changes.
- Temporary FactsYOLO.LocalTest / 0.0.0-local.<source-sha-prefix>; no public release.
- Exactly eleven public native exports, no test fixtures in package assets.
- Same clean source SHA and identical nupkg SHA on both consumer hosts.
- No model/pixel uploads; host-local current-source existing training evidence.
- Explicitly distinguish local and Windows CI execution; no inferred passes.

## Review Focus

- Dispose racing operation admission; native ownership must remain retained.
- Loader failure must not fall through to a different ambient native library.
- Diagnostics/truncated buffers must preserve primary status without replay.
- Package source/cache contamination must not disguise a missing local package.
- Native provenance, license coverage and architecture must survive packaging.

### Task 1: Public managed boundary

**Files:** sdk/dotnet/FactsYOLO/*; tests/sdk-install/Consumer/*; .gitignore.
**Consumes:** existing eleven-entrypoint ABI and proven SafeHandle lease pattern.
**Produces:** FactsYOLO Model/Session/Result, immutable options/results,
NativeRuntime.Query(), FactsYoloException with native diagnostic/status.

- [ ] Write public-only TD1/lifetime/error consumer with explicit image shape.
- [ ] Run compile probe against absent SDK; expect missing public SDK failure.
- [ ] Extract non-fixture interop and lifetime algorithm; implement bounded
  diagnostics and installed-asset loading. Keep fixture smoke unchanged.
- [ ] Execute consumer against real host TD1 artifact and six reference samples;
  expect exact class/count and 1e-4 absolute+relative float agreement, lifetime
  and error assertions; publish/run JIT and AOT.
- [ ] Commit the independently testable managed boundary.

### Task 2: Native inventory and one local package

**Files:** sdk/dotnet/build_package.py; sdk/dotnet/native_assets.py;
tests/sdk-install/test_packaging.py; SDK package metadata/docs.
**Consumes:** Task 1 assembly and native td1 build, same source SHA on each RID.
**Produces:** two verified asset bundles and one nupkg with manifests/licenses.

- [ ] Write packaging regressions: absent RID, wrong source/hash/architecture,
  fixture exports, missing licenses; run and observe rejection behavior missing.
- [ ] Implement host staging (actual exports/dependencies, selected graph/license
  copies), both-RID validation, packing and package content validation.
- [ ] Run regressions and stage actual Linux asset; obtain Windows asset through
  ordinary CI, pack once, record package SHA256. Never fabricate a Windows asset.
- [ ] Commit packaging with explicit inventory limitations and local-only docs.

### Task 3: Detached installation and dual-host gate

**Files:** tests/sdk-install/verify.py; tests/sdk-install/loader_fixture.c;
.github/workflows/sdk-install.yml; docs/testing/dotnet-sdk.md; README.md.
**Consumes:** exact Task 2 package and current-source Burn/Candle evidence.
**Produces:** command logs, source/package/native hashes, host JIT/AOT outcomes.

- [ ] Run detached restore before package exists; expect NU1101 from local source.
- [ ] Implement fresh cache/feed isolation, RID publish, unrelated run directory,
  native hash/source checks, public-only model consumer and loader negatives.
- [ ] Wire native build matrix → single package → consumer matrix; retain only
  temporary package/native artifacts required to transfer test inputs, never
  weights/pixels; text evidence records exact shared package hash.
- [ ] Run final local workspace builds/tests/verify/explicit AOT and real SDK
  model tests; submit draft PR and inspect actual Windows/Linux CI evidence.
- [ ] Independent whole-branch review; resolve findings with regression tests.
- [ ] Merge only when final local gates, comments, CI and protections permit.
