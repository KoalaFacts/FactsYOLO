# Stage B proxy dependency record

This workspace is development-only (`publish = false`) and is not a member of the
root workspace; the deployed library does not depend on it.

- `fyp-core` has no dependencies and builds with Rust 1.85.
- `fyp-compare` uses exactly the direct dependencies already reviewed for the TD1
  experiment: `burn` 0.22.0 (`std`, `flex`, `autodiff`), `candle-core` 0.11.0
  (no default features), `serde` 1.0.229, `serde_json` 1.0.151 and `sha2` 0.10.9.
  See the [TD1 dependency review](../cpu-detector/DEPENDENCIES.md) for licences,
  native components and obligations.
- `Cargo.lock` started from the TD1 experiment's lockfile; its third-party
  package set (name and version) is identical to that lockfile, so no
  unreviewed registry version is introduced.
- The `measure` job of the `TD2 framework proxy` workflow records each
  candidate's selected graph (`cargo tree -p burn` and `-p candle-core`, normal
  and build edges, with declared licences) as an artifact; the summary is in
  [the comparison record](../../docs/testing/td2-framework-comparison.md).

No GPU, BLAS, libtorch or JIT backend is enabled. No external model, image,
annotation or weight is used; generated pixels and proxy weights stay in memory.
