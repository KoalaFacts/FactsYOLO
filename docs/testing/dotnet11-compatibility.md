# Optional .NET 11 RC2 consumer compatibility

This independently selected gate tests net11.0 consumers of the existing
experimental `FactsYOLO.LocalTest` package. The package still targets net10.0,
is built with SDK 10.0.401, and must not be published. The shipped SDK, default
runtime, C ABI, training/inference implementation and original CI are unchanged.
Passing this gate does not freeze the managed API or promise a stable SDK release.

The `.NET 11 consumer compatibility` workflow runs on PRs changing its workflows,
tooling, package/native inputs, training inputs or the existing consumer,
ownership and loader test sources it reuses. This includes the entire dotnet SDK
and SDK-install helper directories, native crates/headers, Cargo/toolchain files,
native export inventory, shared TD1 compatibility vectors and package notices.
The separate `.NET 11 manual compatibility` entry supports manual dispatch.
Both call one worker: the PR entry pins the PR head, while the manual entry pins
only the selected workflow revision (`github.sha`) and accepts no custom ref.
The worker checks the full SHA and rejects a dispatch revision mismatch before
fetching code, keeping PR inputs out of the manual execution context.
Neither entry has a main-push trigger or adds a branch-protection requirement.
The normal net10.0 workflow remains independent.

## Toolchain and package inputs

`tests/dotnet11/sdk.json` pins SDK 11.0.100-rc.2.26504.105 and runtime
11.0.0-rc.2.26504.105, released 2026-10-13. Both Linux x64 tar and Windows x64 ZIP
URLs and SHA-512 values come from Microsoft's [official release metadata](https://builds.dotnet.microsoft.com/dotnet/release-metadata/11.0/releases.json)
and [download page](https://dotnet.microsoft.com/en-us/download/dotnet/11.0).
The installer checks the actual archive digest before extraction, rejects unsafe
paths and refuses to overwrite an existing SDK directory. The installation is
isolated; it does not change the machine's default dotnet or repository global.json.
The downloaded Microsoft SDK is a test tool, not part of the FactsYOLO package.

Each workflow builds both real native RID assets at its exact source commit,
then uses the existing net10.0 production packer once. Both RC2 consumer hosts
receive that identical archive. This needs no cross-workflow access, new secrets,
package registry write access or model/pixel transfer. Existing source/hash,
export, dependency and license-inventory checks remain required.

## Execute locally

First prepare the actual dual-RID feed and passed host-local current-source
Burn/Candle training evidence as documented in [the existing gate](dotnet-sdk.md).
The feed and training records must identify the clean checkout's full commit;
old artifacts cannot be relabelled. Then, using a new installation and evidence
directory:

```sh
python3 -m unittest discover -s tests/dotnet11 -p 'test_*.py' -v
python3 tests/dotnet11/install_sdk.py /absolute/new/dotnet11
python3 tests/dotnet11/verify.py target/sdk-feed target/sdk-training /absolute/new/dotnet11 target/dotnet11-evidence
```

On Windows use `python` and an actual MSVC x64 developer environment with the
existing NativeAOT prerequisites. A previously downloaded SDK archive can be
supplied using `install_sdk.py <new-directory> --archive <file>`; its pinned
SHA-512 is still checked. No SDK installer bypass or warning suppression is used.

## What is checked

The public Program.cs is copied unchanged outside the checkout. Additional implicit
C# sources (including nested files) and source-tree symbolic links are rejected
before copying, so future baseline consumer changes cannot silently lose code;
generated root bin/obj outputs are
excluded. The same inventory guard applies to the ownership probe. Only the detached
project's target framework/runtime and its global.json select RC2. Restore uses
PackageReference, a fresh global/HTTP cache, a source-mapped local test feed and
official nuget.org for .NET tool packs. It must select `lib/net10.0/FactsYOLO.dll`
and the actual host's native asset; package and native hashes are checked again.
JIT runtimeconfig forbids roll-forward and pins RC2; NativeAOT must restore the
same RC2 ILCompiler. Both executables actually run from an unrelated directory.

Each mode consumes real Burn Flex and Candle training artifacts through the
existing Rust TD1 CPU/F32 backend, compares six held-out samples per artifact with
the existing 1e-4 absolute+relative tolerance, and checks formats, errors, budgets,
disposal and independent model/session/result lifetimes. Burn/Candle remain
training/test tools, not SDK runtime dependencies. These synthetic-rectangle
checks do not establish natural-image quality or production backend selection.

A separate RC2 executable compiles the exact existing Handles/Native/NativeTypes
and synchronized ownership test sources. Its admitted lease must survive Dispose,
allow a real native session, and release the model exactly once after the lease.
It does not substitute source compilation for the package-only public consumer.
The existing full loader rejection suite runs under both RC2 modes as well.

On Linux, an isolated owner-only temporary XDG_RUNTIME_DIR avoids the preview
runtime's fallback to a potentially read-only home directory. HOME and permissions
on existing directories are not changed. All evidence directories must be new.

CI preserves command logs, selected asset/SDK hashes and summaries only. Temporary
native/package transfer artifacts expire after one day; model weights, pixels and
the downloaded SDK are not uploaded. Read actual Linux and Windows results before
claiming either platform passed. Normal exact-head local build/tests/explicit AOT,
independent review, CI and unresolved-comment gates still govern merging changes.
