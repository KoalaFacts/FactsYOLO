# Model/session ownership evidence — M1 task 3

This records executable native infrastructure, not a detector or a completed M1.
Task 2 was merged first as `a72acc2492269d8f64f026f560816ebea876e58a`.

## Implemented boundary

A model handle owns an `Arc` to immutable model state. A session copies its
configuration and retains a separate strong reference. Releasing the public
model handle does not release model state while a session still needs it.

Creation requires an initialized empty output slot. Invalid requests leave it
empty; a nonempty slot is rejected unchanged. Release uses the allocating
library, clears its valid owning slot, and permits a no-op on an already-null
slot value. A null slot pointer is invalid. Copies of freed pointers remain
unsafe: neither numeric validation nor `catch_unwind` proves pointer provenance.
No use/release race or cross-library handle exchange is permitted.

Session description writes only output fields, not the caller's initialized
header or optional extension tail. Configuration is immutable and may be queried
concurrently while its owner remains live. Execution, BUSY/poisoned states,
results, and SDK-managed concurrent disposal belong to later tasks.

The normal build exposes no backend capability and no supported model parser.
`fy_model_open` validates parameters but always returns `UNSUPPORTED_MODEL`, even
in the fixture build. `contract-fixtures` adds a clearly marked CPU/F32 test
capability and `fy_test_model_create`; it does not add a weight-file shortcut.
Six fixture IDs currently identify metadata only, not execution/fault behavior.
The normal and fixture shared libraries have seven and eight `fy_` exports,
respectively. Test-only declarations live outside the public C header.

## Observed red/green execution

Tests were run in GitHub Actions, not in the authoring container, which has no
Rust toolchain or external Git connectivity. These observations name exact runs:

- RED `17c3456`: run `38243149201`, job `114790441424` compiled successfully,
  passed the prior 32 runtime tests, and failed all 18 new lifecycle tests against
  deliberately rejecting stubs. This was a behavior failure, not a missing tool.
- Implementation `b2e7321`: run `38244113960` passed default and fixture runtime
  tests, native builds, Clippy, and Linux ABI/export checks. Rustfmt reported
  formatting differences; those were fixed rather than disabling the gate.
- GREEN `c6b7a21d7ab0771d58165835a279cd57f98cdfef`: run `38245068548` passed all
  jobs: Linux Rust 1.99 (`114796786352`), Linux Rust 1.85 (`114796786642`), and
  Windows Rust 1.99 (`114796786584`). The complete Linux 1.99 log was read.
  This source revision includes the C ownership consumers and corrected raw
  pointer sharing in the scoped-thread test. Later documentation-only updates
  still require current-head CI before merge; this record does not assert an
  unobserved future result.

Per build profile, default tests contain 40 runtime tests and the fixture build
contains 50, including the original 32 and 18 new lifecycle tests. Each mode also
runs one compile-fail image-borrow doctest. Both Debug and Release were executed.
No tests were ignored to obtain the green result.

The lifecycle tests include 10,000 create/release cycles with all three live
object counts returning to baseline, 1,100 rejected session configurations with
no object-count delta, three sessions retaining one model state, and eight
scoped threads each performing 100 immutable descriptions. The counters measure
model handles, model states, and sessions, not RSS, allocator caches, GPU memory,
peak usage, or every possible leak. No process-memory benchmark is claimed.

Linux executes four new C ownership consumers: normal dynamic/static and fixture
dynamic/static. Each fixture C consumer runs 1,000 lifecycle cycles and verifies
that both sessions still work after the model handle is released. The two
existing C ABI query/guard-page consumers and exact export-set checks also pass.
Windows executes Rust/native builds here, not the C consumers.

## Reproduction

Run the commands in [README](../../README.md). The precise C compiler commands
and feature/output directories are in
[the workflow](../../.github/workflows/native-contract.yml). Use
`tests/c/ownership.c` with `-DFY_TEST_FIXTURES -Itests/fixtures` only against the
fixture build; ordinary consumers use `include/facts_yolo.h` and the normal
library. Do not reuse fixture build directories for production artifacts.

Runtime dependencies remain the Rust standard library and platform requirements;
no external Cargo crate, ML framework, pretrained weight, image codec, secret,
package release, or branch-protection change is introduced by this task.

The PR review discussion records the independent review outcome and merge state;
this test record must not be mistaken for independent review or human sign-off.
