# Experimental dependency review

This independent workspace is development-only and `publish=false`. Root M1
Cargo.lock and the default native library do not depend on it. This records the
experiment's dependency decision, not approval to ship a training binary or to
select a production compute backend.

| Direct dependency | Version / enabled features | Declared license | Upstream |
| --- | --- | --- | --- |
| burn | 0.22.0; default-features=false, std/flex/autodiff | MIT OR Apache-2.0 | https://github.com/tracel-ai/burn/tree/v0.22.0 |
| candle-core | 0.11.0; default-features=false | MIT OR Apache-2.0 | https://github.com/huggingface/candle/tree/0.11.0 |
| sha2 | 0.10.9; defaults | MIT OR Apache-2.0 | https://github.com/RustCrypto/hashes |
| serde | 1.0.229; defaults/derive | MIT OR Apache-2.0 | https://github.com/serde-rs/serde |
| serde_json | 1.0.151; defaults | MIT OR Apache-2.0 | https://github.com/serde-rs/json |

Cargo.lock records exact registry checksums. [dependencies.json](dependencies.json)
records210 selected registry packages on Linux x64, including their declared
licenses, source, repositories, Rust versions, active features, link names and
build scripts. Regenerate with `python3 experiments/cpu-detector/tools/inventory.py`
from the pinned Rust1.99 environment after fetching the locked dependencies.
The full lockfile also contains *unselected* optional GPU/BLAS/libtorch packages;
a lock entry or Cargo fetch does not mean that backend is built or executed.

The experiment deliberately enables no GPU, CUDA, Metal, MKL, BLAS, WGPU, libtorch
or JIT backend. Burn Flex is eager CPU execution. Some generic CubeCL utility
crates are selected transitively; the CubeCL compute backend is not enabled.
Burn's Rust1.95 floor is confined to this Rust1.99 evaluation program. The
portable model crate and its sha2 dependency are tested with Rust1.85. The root
workspace still has no external Cargo dependency.

## Components requiring explicit attention

- Burn tensor depends on **colored3.1.1, MPL-2.0**. This is recorded as an
  experimental development dependency, not silently characterized as MIT/Apache.
  No colored source is modified/vendored and no compiled evaluation binary is
  uploaded. A future redistribution must retain applicable notices and provide
  covered source as required by MPL2.0 §§3.1–3.4. Larger-work and covered-file
  conditions must be considered separately; this experiment does not approve
  binary publication. See [upstream license](https://github.com/colored-rs/colored/blob/v3.1.1/LICENSE)
  and [the MPL text](https://www.mozilla.org/en-US/MPL/2.0/).
- Candle core pulls tokenizers0.22.2 -> onig6.5.3 -> **onig_sys69.9.3**.
  The wrapper declares MIT, while its bundled Oniguruma C implementation has a
  separate two-clause BSD notice (`oniguruma/COPYING` in the exact registry
  package, copyright K.Kosako2002–2021). The C code is compiled by `cc`; include
  that notice in source/binary redistribution as required. This makes the whole
  evaluation binary unsuitable for a blanket “pure Rust/no native dependency”
  claim. No separate system Oniguruma library is requested here.
- esaxx-rs has a build script but its `cpp` feature is disabled in the recorded
  graph, so its C++ compilation path is inactive. Rayon is enabled by upstream
  dependencies; this experiment fixes `RAYON_NUM_THREADS=1`.
- Unicode data crates include Unicode-3.0 notices; fast-float2 uses
  MIT OR Apache-2.0, and other exact declarations are in the inventory. Preserve
  each selected license/copyright notice when distributing covered material.
- Rust toolchain/std, host C runtime and compiler/linker components retain their
  own terms. The inventory is a selected Cargo graph, **not a binary SBOM**.

No third-party implementation is copied into original model, loss, optimizer,
synthetic generator or artifact code. The adapter source calls published library
APIs. No external model, image dataset or pretrained weight is used. Review of
experimental dependency scope is explicit here; production backend, packaging,
license-copy bundles and versioned binary SBOM remain separate decisions.
