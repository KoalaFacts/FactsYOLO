# M1 task 5 execution ledger

Plan: docs/superpowers/plans/2026-10-12-native-foundation.md, task 5.
Spec: sections 7-8 of the native contract. Base: 3b8b099 (PR #7 merged first).

Owner requested continuous infrastructure execution, with sequential reviewed merges.
Task 4 resumed rather than reimplemented: exact-head native CI, full Linux log,
CodeQL success and independent review were read before merging PR #7.

Ruling: this environment has no Rust compiler or external Git DNS, verified by
local command execution. Author isolated Git trees and a feature branch through
the connected repository API; execute Rust/C tests in existing GitHub Actions.
Do not claim a local build or fabricate a worktree clone.

Task 5 RED: add actual concurrency/failure assertions with rejecting test-control
stubs. Existing model IDs 4-6 still reject execution. No new behavior is claimed.
Wait assertions occur only after latch cleanup and scoped joins, so a failed
assertion cannot strand a worker.

Ruling: invalid backend output poisons conservatively; no adapter guarantees
reusability after this fault. Invalid host input before admission remains recoverable.
Backend errors require an explicit reusable-state classification, not a numeric
status-code guess. A later adapter must supply that guarantee.

Ruling: latch controls belong only to contract-fixtures, never the production API.
A resume permit may precede entry to make test cleanup robust; each blocked call
consumes one permit. The internal fixture wait has a finite fail-safe and is not
production cancellation or a promised inference deadline.

No model, compute dependency, package publication, security suppression, or human
contributor certification is part of this task. AI-assisted.
