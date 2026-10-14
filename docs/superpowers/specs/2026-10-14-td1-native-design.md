# TD1 native integration

M2a is merged at81ef46fceffd0760c29c4a51f24b4e02d8590d90 with all17checks and
complete local verification. This M2b slice implements the owner's already
approved next requirement: exercise the trained artifact through existing C and
C# JIT/Native AOT contracts. No new architecture, external weights or framework
selection is introduced. Execute inline within that authorization.

## Build and compatibility

Add an explicit production-candidate `td1` Cargo feature to `facts-yolo`, disabled
by default. It depends only on the shared `fy-tiny-model` crate (Rust1.85) and its
sha2 graph, never Burn/Candle. Keep that crate in its existing independent
workspace and exclude it from root automatic workspace membership. Root default
and `contract-fixtures` builds keep their M1 behavior; `td1` adds actual available
CPU F32 backend ID1 (`FY_BACKEND_TD1_CPU`). A combined td1+contract-fixtures build
advertises both, with TEST_ONLY set only on the private fixture capability.

No function/record/export-set/ABI version changes. Capabilities already negotiate
compiled optional execution; a newly available backend does not change ABI1.0.
No extra public symbols. Default builds cannot accept TD1. All new root lockfile
packages are fetched explicitly before mandatory offline checks. Preserve
Rust1.85 for all native code and existing Linux/Windows gates.

## Model/session execution

`fy_model_open` keeps all existing descriptor/pointer/limit checks. Under td1,
recognize the fixed FYTDv001 format and invoke the existing parser. Its exact
51,284-byte size, checksum, architecture/count/finite-weight limits remain as M2a.
Unknown/corrupt/truncated artifacts return FY_UNSUPPORTED_MODEL and never populate
the output slot. Artifact-byte limits return FY_LIMIT_EXCEEDED. Parser budget
units for TD1 are defined as one unit per artifact byte: require at least51,284
before parsing a recognized artifact; zero remains invalid. No file/network I/O
or caller-storage retention in the ABI. Model identity is the verified32-byte
SHA256 digest stored in the artifact header (header-prefix plus parameter payload).

Sessions retain immutable model state and existing one-at-a-time admission.
TD1 permits CPU device0,F32,one thread,no fallback; other settings preserve
existing precise errors. Require64x64 images before admission so an invalid
shape cannot poison a session. Convert RGB/BGR/straight-RGBA views through the
existing checked `rgb_at` accessor (ignore alpha and stride padding), then call
shared TD1 preprocessing/forward/decode. No duplicated model math in bindings.
TD1's inherent background/0.5 score gate remains; the detect-call threshold can
further filter its at-most-one candidate. Existing finalize and owned-result
metadata/copy/release paths remain authoritative. Results outlive model/session
and input pixel buffers. Fixture2x2 and panic/admission semantics stay unchanged.

## Verification contract

Rust tests use runtime-generated, known-parameter artifacts to prove actual
parameter-controlled class/box/background results and identity; changing input
or parameter bytes changes predictions. Cover successful model release before
session use, result lifetime after session release, paddedRGB/BGR/RGBA, failed
shape followed by successful reuse, parser/byte budget, bad checksum/header,
unknown format, unsupported device/settings and output-slot preservation.
Test default,td1,fixture and combined selections; portability remains1.85.

A new additive native verifier consumes the *actual trained* artifact and held-out
RGB/metrics output from M2a's fixed experiment. It builds td1-only shared/static
native outputs, verifies exact normal exports, executes C consumers against the
shared and static library, and executes the existing managed adapter in both JIT
and Native AOT modes with explicit native paths. Bindings receive six held-out
samples including positives/background and compare class,score,corners to the
portable saved/reloaded reference at abs1e-4+relative1e-4. They also exercise
ownership, invalid artifact/shape, and threshold behavior; no fixture factory
may satisfy this real-model path. No weights/pixels/binaries are uploaded.

Linux CI runs the full real training then C/C# native verifier. Windows retains
all M1 gates and additionally runs native td1 Rust contract tests; this slice
claims real-artifact C/C# integration on Linux x64 only until actual Windows
integration evidence exists. Local root builds/tests,verify,explicit AOT,Miri,
i686,portable+candidate experiment and actual C/C# TD1 path are all mandatory.
Fresh whole-branch review, zero unresolved review threads and passing available
CI precede a normal expected-head merge. No protection bypass, database change,
external model/data, public artifact release or CI-for-local-failure substitution.
