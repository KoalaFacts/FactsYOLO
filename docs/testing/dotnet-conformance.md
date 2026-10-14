# C# lifetime and Native AOT conformance — M1 task 7

Original text: Apache-2.0. This records executed tests, not a production SDK release.

## Scope

`tests/dotnet/FactsYOLO.Smoke` is an unpublished .NET 10 console consumer of the
same native library used by the C tests. It implements no image preprocessing,
coordinate conversion or detector algorithm. LibraryImport declarations mirror
all eleven public C entry points with Cdecl and unmanaged fixed-width records.

Each managed Model, Session or Result owns a SafeHandle. Creation allocates the
managed owner before a native allocation and cleans up every unadopted native
handle on failure. ReleaseHandle calls the allocating library's pointer-to-slot
release, never Marshal.FreeHGlobal or a managed allocator.

A short lifecycle lock admits a scoped DangerousAddRef lease before native calls.
Dispose prevents new leases but does not release native memory until existing
leases end. No lock is held across Detect that would hide native BUSY behavior.
Input spans are pinned only for synchronous execution. Results own native state
independently and copy into managed arrays without exposing borrowed pointers.

The process selects one existing absolute native-library path and retains that
module while cached imports or SafeHandle finalizers can exist. This is not a
hot-unload interface or a full distributable package loader.

## Reproduce

Required: pinned Rust toolchain, .NET SDK **10.0.401** and runtime **10.0.12**, plus
Linux native tools (including nm and the Native AOT compiler/linker prerequisites)
or the Windows x64 Visual Studio developer environment (including dumpbin).
`tests/dotnet/global.json` disables SDK roll-forward. The project has no explicit
PackageReference; SDK/Native AOT runtime packs still require restore access on a
cold build. Offline flags below apply to Cargo, not to .NET package restoration.

From the repository root:

```sh
cargo run -p xtask --locked --offline -- dotnet --mode jit
cargo run -p xtask --locked --offline -- dotnet --mode aot
```

Optional `--rid linux-x64` or `--rid win-x64` must match the host; the runner does
not cross-compile and then pretend it executed the foreign binary. It creates
separate normal and contract-fixtures libraries, verifies their exact exports,
generates a Rust compiler layout manifest, builds the C# consumer and executes
normal/fixture processes separately. AOT mode runs the published native executable
and verifies RuntimeFeature.IsDynamicCodeSupported is false.

## Observed evidence

Source revision **e49f5c4f9dc40b127258060a1ef9ae1a04bdce93**, tested through its PR
merge checkout fb42c9235f1948ee3481113a14740b5af928c701 on 2026-10-13:

| Workflow/job | Result actually read |
| --- | --- |
| Dotnet conformance 38342344219, Linux job 115115924512 | JIT and Native AOT built and executed successfully; full log read |
| Dotnet conformance 38342344219, Windows job 115115924382 | JIT and Native AOT built and executed successfully; full log read |
| Native contract 38342344120 | Completed successfully |
| C conformance 38342344237 | Completed successfully |

Each .NET execution against the normal library passes **6** named test groups;
each execution against the fixture library passes **14**, including those same
six. Both sets execute in both modes on both platforms. These repeated
configurations are not 80 independent test cases. Both JIT builds report zero
compiler warnings and errors. AOT also passes warning-as-error compilation.

The tests verify:

- 113 compiler-derived pointer-width, size, alignment, field-offset and constant
  records agree with Rust (the C conformance gate checks the same manifest).
- Nonzero status values, including unknown/negative values, remain failures.
- Missing libraries fail explicitly; the normal build exposes no test factory or
  counters and still rejects otherwise valid model artifacts.
- Model disposal leaves session state alive; session disposal leaves completed
  results alive; repeated disposal is harmless; later calls fail as disposed.
- Short result copies preserve destination sentinels; result copies are
  independent; threshold, cap, empty-result, native fault and poison rules agree.
- A blocking native fixture confirms Detect is in progress before Dispose. The
  native session/model remain live during forced GC, a second call returns BUSY,
  closed admission is rejected, and physical session release occurs exactly
  once after the first call completes. The result remains readable afterward.
- 1,000 create/detect/copy/release cycles return native object counts to zero;
  forgotten managed owners are reclaimed through SafeHandle finalization.

RED checkpoint **8eb9c3** compiled with zero warnings/errors in run 38341478439,
Linux job 115113217745, then failed the new unfinished behavior: normal JIT had
3 failures and fixture JIT had 12. This was an executed red checkpoint, not just
an absent compiler. The existing Rust tests remained operational; their separate
rustfmt failure was corrected without weakening the formatter gate.

## Test-only measurement and limits

`fy_test_resource_counts` exists only with contract-fixtures, is declared only in
the test header, and requires exactly four writable size_t slots. It reports
public model handles, retained model states, sessions and results. A coherent
snapshot requires quiescent allocation or a controlled test latch; it is not a
transactional live telemetry API. Invalid buffer/capacity cases preserve outputs.
The normal export count remains eleven; the fixture count is fifteen.

These counters do not measure allocator bytes, RSS, native heap peaks, GPU memory
or complete leak freedom. The deliberately exceptional panic-payload retention
from task 5 is not disguised as normal-path resource accounting. Expected fixture
panic text in the log is not a failed test: the status/poison assertions verify
that the recoverable unwind boundary returns control to the caller.

The managed adapters and fixture imports are internal to this executable; no
NuGet package, binary release, trained model, real backend, Go/Node binding or
browser capability is published. Real deployment distribution must separately
account for .NET/native-toolchain notices and platform prerequisites. Resource
byte/RSS measurements, Miri and the remaining M1 safety gates belong to task 8.

The original execution record above was GitHub Actions; its authoring container
did not have those compilers installed. AI-assisted; no human reviewer approval
or contributor certification is asserted by this document.

## Task 7 recovery: fixture-control lifetime

A subsequent source review of `f7c934a7` identified a test-harness lifetime gap:
after Dispose, a raw Resume call relied only on Detect's lease. Resume can wake
Detect, allowing that last lease to end while Resume is still returning. The
fixture's timeout also means an earlier running barrier cannot indefinitely
guarantee a live handle. No observed crash is claimed.

The managed test now leases WaitRunning normally. After the original
Dispose/forced-GC liveness and closed-admission checks, it acquires a separate
test-control lease. SafeHandle acquisition atomically retains a still-live handle
or rejects one already closed; there is no raw-pointer fallback. The control
lease remains through Resume and the Detect join. The test verifies release count
zero and counts `[0, 1, 1, 1]` before dropping control, exactly one release afterward,
closed control/wait rejection, and the original independent result lifetime.
Holding control only after the original liveness checks avoids masking a missing
Detect lease. No native export or ordinary wrapper admission rule changes.

Local recovery execution on Debian 13 Linux x64 used Rust 1.99.0, GCC 14.2.0,
.NET SDK 10.0.401 and runtime 10.0.12. The original-head Rust Debug/Release normal
and fixture suites, clippy in both feature modes, fmt, and C conformance passed.
An executed RED run first added the control-lifetime assertion without the repair:
JIT compilation had zero warnings/errors, normal passed 6 groups, and fixture
passed 13 with exactly one failure at the new assertion. This demonstrates the
missing control ownership, not a reproduced use-after-free. With the repair,
JIT compilation again had zero warnings/errors and normal/fixture passed 6/14.

The recovery environment's local Native AOT attempt is **blocked**, not passed:
MSBuild's `ComputeManagedAssemblies` out-of-process task host cannot create its
Unix-domain named pipe (`SocketException (13): Permission denied`, followed by
MSB4216/MSB4027). A supported command-level escalation produced the same error.
The SDK, task isolation, warnings and AOT gate were not weakened. Final-head
Native AOT still requires an executor that permits that local IPC, using the
unchanged reproduction command above. Historical CI does not substitute for
this pending local check or new-head CI after the repair.
