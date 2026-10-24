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

The production-nuspec regression requires both real bundles under
`target/sdk-assets`, or a local root selected by `FACTSYOLO_SDK_TEST_ASSETS`.
It runs the production project's nuspec generation targets with
`ContinuePackingAfterGeneratingNuspec=false`, checks every generated payload
path against both verified inventories, and asserts that it creates no nupkg.
This catches duplicated recursive buildinfo paths without creating a second
shared package or weakening warnings. The full pack and both installed consumers
remain separate required gates.

Before training, each consumer host also runs
`python3 tests/sdk-install/test_loader_fixtures.py -v` (`python` on Windows).
This compiles all four loader-fixture variants using the actual host compiler
with warnings as errors, including MSVC `/W4 /WX`. Those four variants are
compilation checks. Separate host-generated export fixtures in the same test
also link shared libraries and execute loader probes. The installed consumer's
full JIT/AOT rejection checks remain required.

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

The real-model consumer also copies empty and nonempty results into a 65,536-record
destination slice. It checks the copied records, unchanged tail and surrounding
canaries, and less than 64 KiB of managed allocation for the warmed copy call.
The destination is allocated before measurement. `CopyTo` queries the immutable
native result count under the same lifetime lease and bounds temporary storage by
the smaller of that count and the caller's capacity; a short destination still
receives the original native error. This is a result-copy resource regression,
not an inference allocation benchmark or a stable SDK performance promise.

A separate synchronized ownership test compiles the exact SDK lease/SafeHandle
source into a test-only executable. It holds an admitted lease across Dispose,
uses the retained actual TD1 model to create a native session after Dispose, and
asserts exactly one physical model release after the lease ends. It runs with JIT
and AOT and adds no accessors/fixture imports to the shipped assembly. The public
consumer's 32 scheduling/disposal attempts are stress checks, not proof of an
admitted-operation overlap by themselves.

A second source-linked executable tests the exact `Result.CopyTo` and lease
sources against a controlled, test-only `Native` boundary. Separate query/copy
barriers pause an admitted call while the same Result is disposed twice. New
copies must be rejected, the admitted copy must preserve data and destination
tail/canaries, and the boundary's unmanaged records must be physically freed
exactly once after the operation. A detached negative-control source copy ends
the lease after the count query; both schedules must fail specifically for
premature release, before dereferencing freed memory. Baseline and negative
control run in JIT/NativeAOT on both hosts, including optional RC2 consumers.
This is deterministic managed ownership evidence using a controlled boundary,
not real native inference or a PackageReference consumer. Production SDK source
and public APIs contain no test hooks; the actual installed-package checks above
remain separate.

A third source-linked executable covers the production `Session.Detect` BUSY
contract with a test-only `Native` boundary. A barrier holds the first call while
a dedicated second worker must complete with status 6 before that barrier is
released. It preserves the UTF-8 diagnostic text, required-byte count and
truncation flag, and must reach the BUSY boundary exactly once. After resumption,
the first result remains usable and the same session can detect again. Both
complete and truncated diagnostics run in JIT/NativeAOT, including RC2.
Detached source mutants swallow BUSY, serialize the whole Detect call, or retry
BUSY; each must fail for its specific contract violation. The serialization
watchdog tests completion while the first call is paused, not a millisecond
latency promise. Cleanup always resumes and joins both workers. This proves the
managed wrapper contract against a simulated boundary; the separate installed
package consumers still verify actual trained-model/native detection.

The CI workflow builds both RID assets, packs once, and supplies the identical
archive to Linux/Windows consumers. Pull-request jobs use the explicit PR head
for native source provenance, so the same package can be verified locally.
Native asset and package transfer artifacts expire after one day; no weights or
pixels are uploaded. Final evidence artifacts contain text logs and JSON only.
This workflow is not a release workflow.

Bundle verification also reads the binary's actual `fy_*` exports independently
of the manifest. An accurate file hash does not prove a declared export list:
an actual fixture library with a revised hash and a public-only declaration was
previously accepted. The offline reader checks the pinned ELF64 dynamic symbol
section or PE32+ named-export tables, with file bounds and defined-function/RVA
checks, and requires the existing eleven names. It runs on both bundle reception
and package extraction without loading the other host's code or adding tools.
The pinned ABI exports are unversioned: GNU symbol-version entries for `fy_*`
must be globally available without a version (index 1). Versioned `fy_*` entries
are rejected, including non-default versions whose unchanged symbol-table name
cannot be resolved by ordinary `dlsym`. A Linux host-compiled `.symver` fixture
checks the actual lookup failure and bundle rejection; imported dependency
versions remain supported. See the [GNU symbol-version table specification](https://refspecs.linuxfoundation.org/LSB_3.1.1/LSB-Core-generic/LSB-Core-generic/symversion.html).
ELF section inventories must agree with the runtime dynamic tags and unique load
segment mappings. Each API name must be reachable through the active GNU hash
(including bloom bits) or SysV hash; a SysV table cannot rescue a bad GNU lookup.
Function addresses must be nonzero and contained in executable sections and load
segments. PE names must remain strictly sorted for lookup, and function addresses
must lie within executable, file-backed image ranges. Host-compiled GNU/SysV/both
fixtures compare ordinary loader lookups with audit rejection; staged ELF/PE
mutations cover invalid addresses, table ordering, decoys and hash corruption.
For a nonempty GNU hash, the audit derives the terminal symbol index from the
largest bucket and its terminating chain, then requires agreement with the
dynamic-symbol section size. Shrinking that inventory cannot hide a runtime
export; truncated hash tables and unterminated chains are rejected. Empty GNU
hashes retain support for an unhashed import prefix without inventing chain
entries. GNU-unique definitions and the low two visibility bits follow the
Linux/glibc [symbol definitions](https://github.com/bminor/glibc/blob/master/elf/elf.h)
and [runtime lookup](https://github.com/bminor/glibc/blob/master/elf/dl-lookup.c).
These are independently implemented checks, with no upstream code copied.
Host-compiled regressions resolve and call the public stubs plus the extra
symbol in fresh subprocesses before checking rejection of accurately rehashed,
public-only manifests. No model weights or foreign package binaries are loaded
by these export fixtures.
Real staged ELF/PE mutations and host-compiled C stubs exercise rejection; these
are binary validation fixtures, not detection evidence. This is a conservative
audit of the current toolchain outputs, not general executable validation,
publisher authentication or release clearance. It does not validate relocations,
initializers, dependency loading or the behavior of resolved functions. Format references are the
[ELF symbol-table specification](https://gabi.xinuos.com/elf/05-symtab.html) and
[Microsoft PE specification](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format).

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
library copyright report/license files and a pinned
[runtime supplement](../../sdk/dotnet/runtime-notices/inventory.json) are included
conservatively; the supplement covers observed omissions in Rust 1.99.0's report.
Staging rejects a different rustc release/commit until its notices are reviewed.
The [Linux binary audit](linux-sdk-redistribution.md) records the actual 07e1cc2
ELF requirements and the limits of that evidence. External
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
