# M1 execution ledger

Current checkpoint: tasks 1–7 merged through PR10, `43746283ae66ef768b1f0a34be4180f25bfe4b85`.
Task 8 resource/safety gates are under implementation/review; see [current evidence](m1-evidence.md).
No real detector/backend/model format or SDK release has been selected.

## Historical task-4 execution record

Plan: `docs/superpowers/plans/2026-10-12-native-foundation.md`.
Owner direction: continuously implement infrastructure; integrate each reviewed task before starting the next.

- Tasks 1-3: merged through c3a6723 (PRs 4, 5, 6). Do not reimplement.
- Task 4: active. RED f0fdc5c/run 38252226013/job 114821181024: prior 52 runtime tests pass, all nine finalizer cases fail against the rejecting stub. At 1075dd8/run 38252563436/job 114822337919, all nine finalizer cases and an internal source-index tie test pass; all eight new FFI result cases fail against rejecting stubs. Full logs read.
- Tasks 5-8: pending; no completion claim.

Ruling: author on an isolated remote feature branch and execute CI there. The current container has no Rust compiler and a fresh Git connectivity attempt failed DNS. This is not a claim of local builds or a local git worktree.

Preflight: task 4 consumes Session/Model ownership and ImageView, and produces OwnedResult for task 5. Tasks 6-8 consume the same eleven-function ABI, not independent implementations.

Ruling: inverted candidate edges and nonfinite inverse-transform arithmetic are malformed model output. Clipped zero-area boxes are discarded. Invalid candidates remain errors below threshold or beyond cap.

Ruling: a minimal try_lock admission guard is introduced with synchronous detect so the new entry point does not temporarily allow overlapping execution. The fault-state machine, deterministic overlap tests and fault fixtures remain task 5. Only shared outer references exist before admission. This dependency does not mark task 5 complete.

Ruling: bounded top-k uses O(min(candidate count, cap)) retained records with fallible reservations rather than allocating one record for every candidate before truncation. Model-specific suppression is still not implemented; canonical fixtures are post-suppression.

No detector, weights, parser, third-party runtime, SDK publication, security-policy change or release is authorized by these steps.
