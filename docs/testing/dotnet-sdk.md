# Local SDK installation gate

This gate tests a temporary `FactsYOLO.LocalTest` package. It does not publish a
NuGet package, register a name, promise a stable API, or distribute weights.
The package's version is `0.0.0-local.<12-character-source-sha>`; both native
assets and the package metadata must identify the same full source commit.
The managed assembly/namespace is FactsYOLO, targeting net10.0 with SDK10.0.401.
Public package naming, versioning and release clearance remain separate decisions.

## Reproduce

Start from the same clean checkout on Linux x64 and Windows x64. Install Rust
1.99.0 with rust-docs (original standard-library notices), the existing C tools
and pinned .NET SDK. On Windows use the existing MSVC x64 developer environment;
Git Bash's GNU link.exe must not replace MSVC link.exe. Fetch the pinned crates
using the existing registry configuration, then stage each host's actual asset:

```sh
cargo fetch --locked
python3 sdk/dotnet/native_assets.py target/sdk-assets
```

Use `python` on Windows. Each stage creates its own RID subdirectory. Transfer
both directories to one packaging host, preserving their contents. Then:

```sh
python3 -m unittest discover -s tests/sdk-install -p test_packaging.py -v
python3 sdk/dotnet/build_package.py target/sdk-assets target/sdk-feed
```

The packer rejects missing/mismatched assets, fixtures, architecture or hash
mismatches, extra files and absent license inventories. It creates a single
nupkg and `package.json` with its SHA256. Do not build a different package on each
OS using the same ID/version. Transfer this **same feed** to both consumer hosts.
No native file is manually copied into the consumer's output.

The packaging tests include an actual ordinary `dotnet pack` with the pinned
SDK, followed by the same archive-entry validator used by the SDK packer. Exactly
one core-properties entry is allowed: SDK 10.0.401's `nuget.psmdcp` or the legacy
hexadecimal filename. Duplicate entries, unsafe paths and unlisted content remain
rejected. This small managed-only regression creates no native assets and does
not substitute for the dual-platform package installation gate.

A second regression builds the actual SDK into an isolated output directory,
reads its evaluated managed pack items, and strictly packs a Linux-only projection
with a real verified native bundle. It requires `target/sdk-assets/linux-x64`
(already staged in CI), or a local path set via
`FACTSYOLO_SDK_TEST_LINUX_ASSETS`. Missing real assets fail the test. The expected
managed entry is `lib/net10.0/FactsYOLO.dll`, used for compilation and runtime;
native entries remain RID-specific. No NU5131 suppression or relaxed warning
policy is used. This projection isolates the layout regression and does not
replace the actual production project's dual-RID pack gate or Windows execution.

On each host use the unchanged existing training verifier to generate current
source Burn/Candle models and six held-out examples locally, then install/run:

```sh
cargo fetch --manifest-path experiments/cpu-detector/Cargo.toml --locked
python3 experiments/cpu-detector/verify.py target/sdk-training
python3 tests/sdk-install/verify.py target/sdk-feed target/sdk-training target/sdk-install-evidence
```

Every output directory must be new. Installation uses a new OS temporary directory
outside the checkout and fresh NuGet global/HTTP caches. Only PackageReference is
accepted; the SDK ID is source-mapped to the local feed and official nuget.org
supplies .NET tool packs. JIT and NativeAOT publishes use the explicit host RID.
Execution occurs from another working directory. Evidence checks cached nupkg
hash, published native hash, ABI, source build ID, actual backend and execution
mode, then public-model results, diagnostics, formats, disposal and ownership.
Loader rejection tests use test-only C stubs, never inference fixtures or packaged
assets. A valid native library in the working directory must not rescue a missing
or broken installed asset.

A separate synchronized ownership test compiles the exact SDK lease/SafeHandle
source into a test-only executable. It holds an admitted lease across Dispose,
uses the retained actual TD1 model to create a native session after Dispose, and
asserts exactly one physical model release after the lease ends. It runs with JIT
and AOT and adds no accessors/fixture imports to the shipped assembly. The public
consumer's 32 scheduling/disposal attempts are stress checks, not proof of an
admitted-operation overlap by themselves.

The CI workflow builds both RID assets, packs once, and supplies the identical
archive to Linux/Windows consumers. Pull-request jobs use the explicit PR head
for native source provenance, so the same package can be verified locally.
Native asset and package transfer artifacts expire after one day; no weights or
pixels are uploaded. Final evidence artifacts contain text logs and JSON only.
This workflow is not a release workflow.

## Platform and license limits

Linux assets use GNU/Linux x64; actual ELF NEEDED entries and versioned symbols
are retained. They do not imply arbitrary glibc/musl or older-distribution support.
Windows assets use x64 MSVC; dumpbin imports record system/CRT dependencies.
The managed AnyCPU assembly does not expand native architecture support.
NativeAOT still loads the external native asset and its OS dependencies.

Original code is Apache-2.0. Package LICENSE/NOTICE and per-crate license copies
are included. Target SPDX2.3 records are source/build inventories with the exact
native binary hash: they distinguish native-source and build-only Cargo entries
but do not determine the dead-stripped linked subset. The original Rust standard
library copyright report/license files are included conservatively. External
system prerequisites are recorded separately. A complete linked-binary SBOM,
redistribution audit and supported-OS matrix remain required before public release.
No inference SDK dependency on Burn/Candle is added; those remain test/training
tools. Models keep their separate provenance and license requirements.

## Evidence status

Development Linux JIT/NativeAOT public-model, ownership and error regressions
have run. Cross-host package installation and final revision gates must be read
from actual command logs and CI, not inferred from this workflow's presence.
Do not treat an approval/billing/runner failure as a successful execution.

See the [design](../superpowers/specs/2026-10-15-local-dotnet-sdk-design.md),
[implementation plan](../superpowers/plans/2026-10-15-local-dotnet-sdk.md),
[SDK usage and limitations](../../sdk/dotnet/README.md) and
[existing ABI conformance](dotnet-conformance.md).
