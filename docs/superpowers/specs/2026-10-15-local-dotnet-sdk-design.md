# Local .NET SDK installation gate

Status: implementation authorized in the 2026-10-15 recovery instruction,
based on the preceding read-only SDK assessment. This is a local test package,
not approval of a public package name, version, support promise or release.

## Scope

Expose the existing experimental ABI 1.0 through a thin net10.0 C# library.
Use temporary PackageId `FactsYOLO.LocalTest`, version
`0.0.0-local.<source-sha-prefix>`, assembly/namespace `FactsYOLO`. Produce one
package containing Linux x64 and Windows x64 native assets from the same clean
source commit. Both hosts must install the identical package SHA256. Do not
publish NuGet packages, register names, add credentials, upload models/pixels,
change Rust inference/training/model formats or the C ABI, or claim natural-scene
accuracy. Preserve the merged PR15 gates.

## Managed boundary

Public Model.Open, Model.CreateSession, Session.Detect, Result.Copy/Info and
Dispose adapt the eleven existing entrypoints. Public immutable options carry
budgets, CPU/backend selection and image shape; there are no 2x2 test defaults.
SafeHandle leases retain native ownership across concurrent disposal; model,
session and result lifetime independence remains the native contract. Copy owned
results into ordinary managed records. Preserve numeric statuses and bounded
per-call UTF-8 diagnostics without replaying failed inference. No fixture imports
or public raw pointers. Existing fixture smoke adapter remains an independent
ABI regression oracle; shared lifetime code is extracted without changing its
ownership algorithm.

Use net10.0, SDK 10.0.401 and runtime 10.0.12. Keep the selected native module
loaded for process lifetime. Resolve only the installed application's selected
RID asset or published asset beside the application, never the current working
directory, repository target directories or an arbitrary library on PATH.
Reject unsupported process architecture/OS, missing files/exports and ABI major
mismatch before allocating handles. Diagnostics retain RID, process architecture,
attempted paths and the original loader exception. Linux libc/Windows CRT
requirements are recorded from actual binaries; RID labels are not universal OS
compatibility claims.

## Package and provenance

Use ref/net10.0 and runtimes/any/lib/net10.0 for managed assets, and
runtimes/{linux-x64,win-x64}/native for shared libraries. Package no import/static
library, model or pixel data. Stage each native binary with exact source commit,
SHA256, toolchain, selected Cargo dependency inventory, export set and platform
dependency inspection. Include Apache LICENSE/NOTICE, applicable dependency
license texts and per-target SBOM/inventory with explicit coverage limits for
Rust standard library and external system runtimes. Do not mislabel a Cargo tree
as a complete linked-binary SBOM. Packing fails on absent/mismatched RID assets,
fixture exports, hash/provenance mismatch or absent notices.

## Acceptance

1. Public-only consumer is copied outside the checkout, with PackageReference
   only, an empty isolated NuGet/global HTTP cache and source mapping restricting
   FactsYOLO.LocalTest to the local feed. Official nuget.org supplies .NET packs.
2. Publish JIT and NativeAOT for the actual host RID, execute from a different
   working directory, assert installed native hash, ABI, build ID and actual TD1
   capability. Assert restore paths do not reference project/source files.
3. Use each host's existing Burn/Candle training verifier and its ephemeral
   passed current-source models and six held-out examples. Keep the existing
   absolute+relative 1e-4 tolerance; check formats, budgets, corrupt models,
   disposed admission, double Dispose, model/session/result independence and
   repeated ownership cycles. No model generation inside the SDK.
4. Separate processes exercise missing native files, malformed/wrong-architecture
   library images, missing exports, wrong ABI and loader dependency failures.
   Preserve original diagnostics and fail rather than silently falling back.
5. Same source final local build/tests/explicit AOT, all applicable CI, independent
   code review and no unresolved review comments gate merging. Direct billing
   evidence may explain unavailable CI, per the recovery instruction, but does
   not supply missing platform execution evidence or waive branch protection.

## Sources

- Existing project/license ADR, native contract and TD1 design documents.
- Microsoft: [native package assets](https://learn.microsoft.com/en-us/nuget/create-packages/native-files-in-net-packages),
  [source mapping](https://learn.microsoft.com/en-us/nuget/consume-packages/package-source-mapping),
  [NativeAOT](https://learn.microsoft.com/en-us/dotnet/core/deploying/native-aot/).
