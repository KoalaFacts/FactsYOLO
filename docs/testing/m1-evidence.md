# M1 task 8: resource and safety gates

Original text: Apache-2.0. This implements the final infrastructure task from
the [accepted M1 plan](../superpowers/plans/2026-10-12-native-foundation.md).
Tasks 1–7 are merged, most recently PR10 at
`43746283ae66ef768b1f0a34be4180f25bfe4b85`. No detector, model format, backend,
weights, training implementation, published SDK or binary release is added.

## Reproduce

With Rust 1.99.0, .NET SDK 10.0.401/runtime 10.0.12 and the existing native
compiler prerequisites, run from the repository root:

```sh
cargo run -p xtask --locked --offline -- verify
cargo run -p xtask --locked --offline -- resources
cargo run -p xtask --locked --offline -- manifest
rustup toolchain install nightly-2026-10-13 --profile minimal --component miri
cargo run -p xtask --locked --offline -- miri
rustup target add i686-unknown-linux-gnu --toolchain 1.99.0
cargo run -p xtask --locked --offline -- layout32
```

`verify` requires the local Linux x64 or Windows x64 native/.NET tools and runs
normal/fixture Debug/Release tests, both clippy modes, fmt, C conformance and
export isolation, resource measurement, and real C# JIT/Native AOT executables.
Linux also executes all six existing C ABI/ownership dynamic/static smoke
consumers. Native builds and library paths remain isolated and explicit.
An unavailable mandatory tool or failed command returns nonzero. A preflight
is only a tool check, never a full verification pass. A copied runner prevents
Windows executable locking from interfering with Cargo's own integration tests.

Miri and the i686 probe are separate mandatory jobs in
[M1 acceptance](../../.github/workflows/m1-acceptance.yml). On Linux, the latter
needs i686 standard-library support, multilib C headers and `objcopy`.
`CC_I686` can identify an installed cross/multilib compiler; no flags turn an
unsupported executable into a claimed native run. `layout32` compiles ELF32
objects and compares 123 little-endian u32 values (the existing 113 records,
including size/alignment pairs), against current and frozen C headers. It asserts
four-byte pointers. It does not execute i686 code or establish 32-bit runtime support.

The descriptor/ownership and complete TD1 Miri suites run in independent jobs,
each retaining the 25-minute limit. Successful TD1 runs have taken about
21 minutes on hosted runners; sharing that budget with toolchain setup and the
earlier suite left little margin. The existing `safety` check waits for both jobs
and runs even when a dependency fails, is cancelled or is skipped. It succeeds
only when both jobs succeed. Commands, pinned toolchains and test coverage are
unchanged. TD1 output is also retained in the `m1-miri-td1` artifact; pipeline
failure propagation prevents the log capture from hiding a test failure.

## Resource workload and interpretation

The example `resource_probe`, built only with `contract-fixtures`, runs in a
separate process. It performs 1,000 warm-up create/detect/copy/release cycles and
10,000 measured cycles in 100 batches of 100. Every cycle exercises nonempty
creation/output-slot failures, failed session creation, independent result
lifetime, count queries and short/full repeated copies. Copy retries must not
reexecute detection. Retained result count is fixed at zero between batches.

The four existing native object counters must be zero after each batch. An
allocator wrapper in this executable delegates unchanged to Rust `System` and
counts successful alloc/alloc_zeroed/realloc/dealloc Layout bytes. It covers
the library's controlled heap buffers and other Rust heap allocations in this
process, including fixed harness allocations; exact live bytes must return to
the post-warmup baseline. A real buffer allocation/reallocation/drop first
checks counter sensitivity independently of object counters. The library's
allocator, public ABI and normal exports are unchanged. This count excludes
allocator bookkeeping/caches, foreign allocations, and GPU memory.

Current memory is sampled separately: Linux `/proc/self/smaps_rollup` Rss plus
Private_Clean/Private_Dirty/Private_Hugetlb, or Windows
`K32GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX)` WorkingSetSize/PrivateUsage.
These are current observations, not peak-RSS or cumulative counters. Linux
private resident bytes and Windows private committed bytes have different
meanings and are never compared across platforms. For each metric, the median
of the final ten samples minus the median of the first ten must be <=16 MiB.
Raw bytes are preserved. This fixture-only regression budget is not a production
memory ceiling, a full allocator audit, or a proof of universal leak freedom.

The exceptional deliberately retained secondary panic payload from task 5 is
outside this normal-loop measurement. Existing native fault tests remain
required and separate; no panic suppression or weakened assertion is added.

## Safety coverage boundary

`descriptor_corpus` runs exactly 10,000 fixed-seed, allocated ABI descriptor
cases across eight classes: unsupported versions/flags, short records,
physically short prefixes, misalignment, extension tails, invalid diagnostics,
and valid controls. Rejected storage and output sentinels must be preserved.
Every pointer derives from a live backing allocation. The prior 10,000-case
image-descriptor/u128 oracle corpus also runs under Miri and in ordinary
Debug/Release tests.

Miri uses pinned `nightly-2026-10-13`, its default safety/leak checking, and no
MIRIFLAGS suppression. It interprets abi, images, lifetimes, results_ffi and the
new complete descriptor corpus. Three named OS-thread tests are excluded from
this single-thread descriptor/ownership gate. The existing 10,000-cycle image/
lifetime stress loops, the complete new 10,000-case ABI corpus, and all shorter
ownership/result tests are interpreted. C foreign calls,
subprocess panic controls, native concurrency, RSS sampling and .NET remain
covered by their real native gates, not claimed as Miri coverage. No sanitizers
or production backend correctness are claimed by this task.

## Evidence and negative controls

Each invocation writes a distinct directory under `target/m1-evidence/` with
revision/tree, dirty status, target, feature/fixture metadata, tool outputs,
commands, actual exit statuses, stdout/stderr logs and an outcome when complete.
A manifest alone is an inventory, not a pass. CI preserves these records as
artifacts, including resource samples and compiler-produced layouts/exports.
Final source SHA and remote run links belong in the PR verification record.

During development on Debian 13 x64:

- The new resource/preflight CLI tests first compiled and failed on the absent
  commands, then passed after implementation.
- A temporary retained-result injection at cycle 1,100 made the resource gate
  fail at batch zero with `[0, 0, 0, 1]`; the injection was removed and the
  resource workload and tests passed again. No injected leak is committed.
- The missing-tool test invokes verify preflight with an empty PATH and requires
  a specific mandatory-tool failure. The normal export negative control checks
  the actual compiled fixture library against the normal allowed symbol set.
- The initial full local verify and pinned Miri gate executed successfully.
  These development runs are not a substitute for final-head local and CI checks.
- The executor could compile but not execute an i386 ELF (`Exec format error`).
  The compiler-only probe then compared actual ELF32 sections successfully;
  no permission or execution restriction was changed.

The older [M1 progress ledger](m1-progress.md) records historical task-4 progress.
The PR sequence and this record supersede its pending-task statements. The
next work after reviewed M1 acceptance is a separate design for the first
independent detector, artifact/training contract and predeclared backend
comparison budgets; no candidate framework or weights are approved here.

## Primary tool references

- [Rust Miri and its coverage limitations](https://github.com/rust-lang/miri)
- [Linux proc memory fields](https://www.kernel.org/doc/html/latest/filesystems/proc.html)
- [Windows process memory counters](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex)
- [Native AOT prerequisites](https://learn.microsoft.com/en-us/dotnet/core/deploying/native-aot/)

AI-assisted implementation and evidence; no human certification or DCO sign-off
is asserted on another person's behalf.
