# M1 execution ledger

Plan: `docs/superpowers/plans/2026-10-12-native-foundation.md`.
Owner direction: continuously implement infrastructure; integrate each reviewed task before starting the next.

- Tasks 1-3: merged through c3a6723 (PRs 4, 5, 6). Do not reimplement.
- Task 4: active, first RED checkpoint exercises the pure result finalizer.
- Tasks 5-8: pending; no completion or passing-test claim.

Ruling: author on an isolated remote feature branch and execute CI there. The current container has no Rust compiler and a fresh Git connectivity attempt failed DNS. This is not a claim of local builds or a local git worktree.

Preflight: task 4 consumes task 3 Session/Model ownership and task 2 ImageView, and produces OwnedResult for task 5. Task 5 will own admission/poisoning; task 4 must not introduce mutable references through raw handles. Tasks 6-8 consume the same eleven-function ABI, not independent reimplementations.

Ruling: inverted candidate edges are malformed model output (MODEL_OUTPUT_INVALID); clipped zero-area boxes are discarded. Invalid candidates remain errors even below threshold or beyond the output cap. This resolves an unspecified malformed-input edge without changing the accepted canonical fixture.

No detector, weights, parser, third-party runtime, SDK publication, security-policy change or release is authorized by these implementation steps.
