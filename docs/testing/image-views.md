# Borrowed image views: M1 task 2

Original text: Apache-2.0. Date: 2026-10-12.

## Implemented scope

`facts_yolo::image::ImageView` borrows immutable host pixels. The safe
`from_bytes` constructor ties its view to a slice lifetime; the unsafe
`from_ffi` constructor requires the caller to prove the foreign allocation
remains live, initialized and immutable for the view's lifetime. The descriptor
borrow alone is not proof of pixel lifetime. Both constructors share one layout
validator. Neither constructor allocates or copies pixel storage.

Supported layouts here are top-down packed RGB8, BGR8 and straight RGBA8. Access
returns RGB and ignores alpha, without compositing or unpremultiplication. There
is no decoder, resizing, normalization, color-profile conversion or new C export.
Existing ABI/query validation continues unchanged.

Required storage is `(height - 1) * row_stride + width * channels`, with checked
arithmetic, positive dimensions/limits, valid format, and stride >= row bytes.
Only that prefix is borrowed; final-row padding is not required. The entire
supplied/declared byte length still counts against the byte budget. Inter-row
padding in the retained prefix must be initialized because it is part of a Rust
byte slice. Oversized/overflowing layouts fail before pixel access; a false but
numerically plausible foreign pointer/length is still a caller safety violation.

## Tests and observed execution

The 12 image integration tests cover exact padded-row length, truncation,
channel equivalence, alpha handling, bad dimensions/stride/format/header, explicit
resource limits, overflow-before-access, no-copy borrowing, out-of-range pixel
reads and sharing an immutable view across threads. A fixed-seed `0xFAC7_2026`
corpus checks 10,000 backed descriptors against independent u128 arithmetic.
The final corpus exercises both the slice and FFI constructors. A compile-fail
doctest confirms a safe view cannot outlive its pixel allocation.

The deliberately rejecting constructors at `2d4840e` were actually run in
[run 38234284880](https://github.com/KoalaFacts/FactsYOLO/actions/runs/38234284880),
job 114761015856: existing 20 tests passed, new image tests had 3 passes and
9 failures. The real implementation at `747bca8` passed all 32 Rust tests and the
borrow-lifetime doctest in Debug and Release in
[run 38234944109](https://github.com/KoalaFacts/FactsYOLO/actions/runs/38234944109).
Linux/Windows builds and clippy succeeded; Linux C dynamic/static smoke checks
also passed. That run was not fully green: Rust 1.99 formatting requested two
line collapses in image.rs. Those exact formatting changes are applied alongside
the final corpus parity assertions; current-head CI must pass before merge.

The [workflow](../../.github/workflows/native-contract.yml) remains unchanged.
Final review and current-head results are recorded on PR #5. This is input
infrastructure, not a trained detector, backend, published SDK or completed M1.
