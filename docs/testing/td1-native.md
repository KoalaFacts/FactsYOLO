# Real TD1 artifacts through C and C#

This M2b integration adds the opt-in `td1` feature to the native library. It loads
the same original trained FYTDv001 artifact validated by the
[CPU experiment](cpu-detector.md), using its shared portable parser/preprocessing/
network/decoder. No model math is reimplemented in a binding. Native inference
has only the optional portable+sha2 graph; Burn/Candle remain training tools.

The [design](../superpowers/specs/2026-10-14-td1-native-design.md) specifies backend1,
CPU0/F32/one thread/no fallback,64x64 images, strict artifact checks, and parser
budget byte units. ABI1.0 records/functions/exports remain unchanged. Default
builds report zero backends and reject artifacts; private fixtures remain explicit
and marked TEST_ONLY. Combined builds advertise TD1 and the fixture separately.
This is a small synthetic single-object model, not a general YOLO product claim.

## Reproduce real training and native consumers

Use Linux x64 or Windows x64, the pinned Rust1.99 and .NET10.0.401 SDK, Rust1.85
and Python3. Linux uses a C compiler/binutils; Windows uses the existing MSVC
x64 tools and Windows SDK in a Visual Studio developer PowerShell. Do not run
the Windows build through Git Bash: its GNU `link.exe` can shadow MSVC's linker.
The [CI workflow](../../.github/workflows/cpu-detector.yml) discovers the installed
MSVC environment in the same way as the existing Windows conformance jobs.
Fetch exact locked packages from the official registry once (use `python` in
place of `python3` on the Windows runner):

```sh
cargo fetch --locked
cargo fetch --manifest-path experiments/cpu-detector/Cargo.toml --locked
python3 experiments/cpu-detector/verify.py experiments/cpu-detector/target/training
python3 experiments/cpu-detector/verify_native.py experiments/cpu-detector/target/training/burn target/td1-burn
python3 experiments/cpu-detector/verify_native.py experiments/cpu-detector/target/training/candle target/td1-candle
```

All destinations must be new. Native verification requires a passed fixed-budget
training report for the current commit; old or failed reports are rejected. It
builds `--no-default-features --features td1`, checks exact normal export sets,
and executes six held-out samples from each candidate (including background).
Linux covers C shared/static libraries; Windows covers the C DLL/import-library
path, matching the existing C target matrix. Both execute C# JIT/Native AOT with
the host's `linux-x64` or `win-x64` runtime identifier. The actual
managed runtime mode is asserted at runtime. Coordinates/score match the portable
reference at abs1e-4+relative1e-4; class/count match exactly. Reference files contain
predictions from saved/reloaded training outputs, not hardcoded fixture boxes.
Each CI host trains both candidates and consumes its own saved artifacts within
the same job. No model is transferred through workflow artifacts. Training can
have small floating-point differences between hosts; exact reload is checked on
each host, without a promise of cross-host bitwise training identity.

Every consumer also checks artifact rejection, shape failure followed by reuse,
RGB/BGR/RGBA with stride padding, score threshold, model release before session
execution, and result ownership after session/input release. Result model_id is
the verified artifact header digest, not a filename or random handle. C# uses the
existing LibraryImport/SafeHandle/lease wrappers. Neither real consumer calls a
private fixture factory; exact exports and capability flags enforce that boundary.

The native verifier writes command lines/exit codes, source/dirty state, actual
artifact SHA256, runtime outputs, compiler layout and comparison cases into its
new evidence directory. The CI workflow uploads logs, scalar summaries and
reference text only. Weights, pixels, native/managed executables and intermediate
objects are not uploaded. Those generated files remain local/ephemeral outputs;
no public model/data or SDK release is implied.

## Native and safety tests

```sh
cargo test -p facts-yolo --features td1 --locked --offline
cargo test -p facts-yolo --features td1 --locked --offline --release
cargo test -p facts-yolo --all-features --locked --offline
cargo test -p facts-yolo --all-features --locked --offline --release
cargo +1.85.0 test -p facts-yolo --features td1 --locked --offline
cargo +nightly-2026-10-13 miri test -p facts-yolo --features td1 --test td1 --locked --offline
cargo run -p xtask --locked --offline -- verify
cargo run -p xtask --locked --offline -- dotnet --mode aot
cargo run -p xtask --locked --offline -- miri
cargo run -p xtask --locked --offline -- layout32
```

Five TD1 Rust FFI tests construct exact known-parameter artifacts at runtime and
prove parameter/input dependence, positive/background decoding, parser/byte
budgets, corrupt/truncated/trailing data, output-slot preservation, capabilities,
unsupported settings and ownership. Full native tests run with default,fixture,
td1 and combined feature selections; Rust1.85 remains tested. The five real-path
tests also run under Miri, alongside the existing53 fixture/descriptor tests.

Initial local C and managed consumers rejected the normal backend-free library;
with TD1, both Burn/Candle trained artifacts passed C shared/static and managed
JIT/AOT. These observations are preflight evidence; final-head commands and CI
artifacts establish the acceptance result for each PR revision. The existing
Linux/Windows M1 gates remain mandatory. The additive consumer matrix now also
executes actual Windows x64 C DLL and C# JIT/AOT consumers, using the same test
sources, model, format and ABI as Linux. Windows static C remains outside the
existing target matrix; no ARM, GPU, SDK release or natural-image claim is added.

The [historical TD1 v1 gate](td1-v1-compatibility.md) additionally uses fixed,
independently encoded synthetic bytes to check format/parameter compatibility,
version and exact-budget errors, score0.5 and failure recovery. These test vectors
add no trained weights, model architecture, platform or SDK support claim.

The experiment lockfile pins `time` to 0.3.47 after checking
[RUSTSEC-2026-0009](https://rustsec.org/advisories/RUSTSEC-2026-0009.html),
which affects RFC2822 parsing before that release. `time` is an unselected optional
transitive dependency: neither the configured CPU experiment nor the native TD1
dependency tree includes it. Its required `time-core`, `time-macros` and `num-conv`
lockfile entries are updated together; the selected dependency inventory and the
root Rust1.85 dependency graph remain unchanged.
