# M1 task 5 execution ledger

Plan: docs/superpowers/plans/2026-10-12-native-foundation.md, task 5.
Spec: sections 7-8 of the native contract. Base: 3b8b099 (PR #7 merged first).

The owner requested continuous infrastructure execution, with sequential reviewed
merges. Task 4 was resumed rather than reimplemented: exact-head native CI, the
full Linux log, CodeQL success and independent review were read before merging.

## Environment and implementation rulings

The authoring environment has no Rust compiler or external Git DNS, confirmed
by local commands. Isolated Git trees and a feature branch are authored through
the connected repository API. Rust/C verification executes in GitHub Actions;
no local build or worktree clone is claimed.

The private `execution.rs` module separates state admission from session settings.
Only a shared outer session reference exists before try_lock. An admitted guard
holds the mutex through backend execution, finalization and owned-result creation.
Incomplete admitted work sets explicit POISONED state, including returned fatal
errors with no Rust unwind. Advisory Mutex poisoning is an additional rejection
signal, not the only representation of validity.

Invalid host options/images remain pre-admission failures. A backend must classify
its failure as explicitly Reusable or Fatal. Identical numeric error statuses can
have different state-validity guarantees; unit tests cover both. Invalid model
output is conservatively fatal. No real backend is supplied in this milestone.

The fixture latch is owned by each session, not the shared model. Its controls
exist only under contract-fixtures and in the test header. One boolean resume
permit may precede entry for cleanup; repeated resumes coalesce. Each blocked
call consumes the permit. A 30-second internal fail-safe bounds broken tests,
not production inference, cancellation or a promised deadline.

All 11 production C exports use the existing common entry boundary. It catches
unwinding panics, destroys the first payload in a second catch boundary, and
forgets only the secondary payload if its destructor itself panics. That rare
retention is intentional and not included in ordinary leak-free claims. The
library does not replace the process panic hook. A dedicated child-process test
installs its own hook and confirms exactly two hook invocations and survival.
Abort, invalid foreign pointers, driver crashes, OOM aborts and double panics
while already unwinding are outside this recovery promise.

## Observed red/green evidence

- 64869d6, Native run 38335196123: the added tests compiled; both controlled-overlap
  tests failed and pre-admission validation passed. No implementation was claimed.
- 030efadd, Native run 38335746464, Linux 1.85 job 115095299017: all four private
  gate tests, both overlap tests and all four fault tests failed against the
  rejecting/incomplete behavior. Existing suites remained green. --no-fail-fast
  ensures a failed integration binary does not hide failures in later binaries.
- 23aa7d8, Native run 38336065271: all required runtime/build/clippy steps passed
  for Linux Rust 1.85/1.99 and Windows Rust 1.99. Full Linux 1.99 job 115096293668
  was read: 56 normal or 83 fixture runtime tests, plus one compile-fail image
  lifetime doctest per mode, in both Debug and Release. The totals overlap.
- The same run passed Linux dynamic/static C query and ownership consumers and
  exact 11-production/14-fixture export checks. Windows C consumers are still
  a later task. Rustfmt failed on four files; its exact reported formatting
  changes are applied in this revision, not bypassed.

Final-head native checks, aggregate CodeQL and independent review must be read
again before merge; earlier runs are not a claim about later revisions. No model,
compute dependency, package publication, security suppression, new secret, or
human contributor certification is part of this task. AI-assisted.
