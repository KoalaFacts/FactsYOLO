# TD1 v1 compatibility baseline

This gate covers synthetic test vectors only. Its historical baseline is PR13,
commit `a58f6c44beb6b7c0c35452da2d3057de41b6ccb9` (tree
`63b3ff0cdf9afb6cb7f9da03175da01b69c22691`). It preserves the format and native
behavior already implemented there. It is not a stable SDK release, a new model,
a trained-weight distribution, or evidence of natural-image accuracy.

## Separate version contracts

- Artifact schema: `FYTDv001`, exactly 51,284 bytes. Header: magic[8], LE u32
  architecture=1, LE u32 parameter_count=12,807, LE u64 payload_bytes=51,228,
  SHA256[32], then 12,807 LE IEEE754 F32 values. The digest covers bytes0..24
  followed by bytes56..end; it detects corruption, not publisher authenticity.
- Architecture1: 64x64 RGB, 8x8 block means divided by255, row/column/channel
  order,192->64 ReLU->7. Parameter order W1[192,64], b1[64], W2[64,7], b2[7].
  Output0 is background,1/2 map to class0/1; the lowest output index wins exact logit
  ties (background first, then class0, then class1). Background or score<0.5
  produces no detection. Four sigmoid outputs are
  normalized cx/cy/w/h, converted to clipped pixel-edge corners. At most one box.
- Native ABI remains experimental1.0 with descriptor version1 and the same
  eleven exports. Unknown ABI majors, descriptor versions and required flags
  retain their existing status categories.
- Package0.1.0 is separate from schema/architecture/ABI versions. This gate does
  not approve publication or a general backwards-compatible SDK promise.

The current writer must reproduce the historical vectors byte-for-byte and the
current reader must accept them with the pinned predictions and parameter bits.
Unknown schema/architecture/count/length, trailing/truncated data, bad checksum,
nonfinite parameters and abs(weight)>10,000 remain rejected. Both signed zeros
and the exact +/-10,000 endpoints are valid. A future format change must use an
explicit version decision; do not regenerate these baselines to hide drift.

## Independent synthetic vectors

`tests/compat/td1_v1_vectors.rs` constructs bytes from literal v1 dimensions,
header bytes, sparse IEEE754 bit patterns and fixed SHA256 digests. It imports
no current model constants, serializer or decoder. Digests are calculated with
Python standard-library hashlib during baseline capture; the portable tests
independently verify them with SHA-2. Valid baseline vectors were checked against
the PR13 reader/writer before acceptance. No random initialization, training,
downloaded model, pixel dataset, or framework is needed for these vectors.

The vectors exercise signed zero, input-dependent class0/class1 predictions,
exact score0.5 with a positive-class tie, finite parameter endpoints, invalid
numeric values and unsupported headers with otherwise valid checksums. They
are built in test memory; no trained weights or image files are committed.

## Native limits, error mapping and recovery

For valid descriptor storage and an empty output slot, positive model limits
are required. Artifact-byte budget<51,284 returns LIMIT_EXCEEDED. After byte
limits, exact length and magic recognition precede TD1 parser-work checks;
parser_steps<51,284 returns LIMIT_EXCEEDED for recognized artifacts. The exact
51,284 budget succeeds; larger budgets do not change output. Unknown architecture
is checked by the shared parser after the work budget, so a recognized artifact
with both an unsupported architecture and too-small parser budget reports the
budget error first. This records existing precedence, not a new rule.

With adequate budgets, all malformed historical variants return
UNSUPPORTED_MODEL; zero limits are INVALID_ARGUMENT. ABI mismatches are
UNSUPPORTED_ABI and unknown required flags are UNSUPPORTED_FEATURE. Failed
creation preserves an empty owning slot. A subsequent valid model open succeeds.
On a live session, invalid shape/threshold leaves the result slot untouched and
the next valid detection succeeds. Exact threshold0.5 includes a score0.5 box;
the next representable F32 threshold above0.5 filters it out.

Only actually allocated, live and appropriately aligned foreign memory is used.
Dangling pointers and use/release races remain outside the raw C contract.

## Verification and limits

Portable historical tests run on Rust1.85 and the pinned evaluation toolchain.
Native historical tests join the existing TD1 suite, including its Miri run and
Linux/Windows Rust matrix. Existing C/C# JIT/AOT, actual Linux trained-artifact
consumers, resource, layout and full local/CI gates remain mandatory. This does
not add Windows trained-artifact C/C# support or a package-installation claim.

Controlled mutations in disposable source copies demonstrate that these tests
fail for coordinated schema drift, unknown-architecture acceptance, parser-budget
off-by-one rejection and loss of score0.5 inclusion. Such deliberate failures
are regression controls, not claims that PR13 had those defects. Production
parser/network/ABI code remains unchanged unless an actual contract bug is found.
