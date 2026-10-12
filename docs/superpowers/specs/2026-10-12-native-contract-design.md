# FactsYOLO native interface, ownership, and session design

Date: 2026-10-12

Status: **Proposed for owner review; not an approved implementation plan.**

Baseline: PR #1 merged as `09d5409a2d677094374e3f91aada20dd4dc7b1e1`.
This document introduces no executable code, released ABI, backend dependency,
model weights, package publication, or supported-platform claim.

## 1. Intent and decision boundary

The owner has selected FactsYOLO, Rust, and a single Apache-2.0 license for
original code and documentation. The product is an independent vision foundation
that C#, Go, and TypeScript products can embed, not a renamed detector wrapper.
See [ADR 0001](../../decisions/0001-project-and-license.md).

This proposal defines the first native contract: resource ownership, input and
result semantics, session concurrency, errors, compatibility, and evidence needed
before publishing an implementation. It preserves one shared Rust implementation.
A first independently implemented YOLO-style detector and its training pipeline
are later work; they are not replaced by a permanent inference-only wrapper.

The proposed first contract is synchronous, single-image, CPU-buffer based, and
limited to axis-aligned detection results. These scope choices are proposals,
not requirements already approved by the owner. Training is a separate component
and must not become a compulsory dependency of an inference distribution.

## 2. Approaches considered

| Approach | Benefit | Cost / decision |
| --- | --- | --- |
| Native Rust core, C ABI, thin bindings | Direct embedding and explicit resource contracts | Recommended; requires careful FFI testing and per-target packaging |
| Mandatory out-of-process service | Crash isolation and fewer host-ABI interactions | Not the baseline; introduces process management and transport/copy overhead |
| WASM-only core for every host | A shared sandbox boundary | Not the baseline; native device integration and browser execution still differ |

Node-API and browser WASM adapters can compile against the same Rust core directly.
They do not have to route through the C ABI. No adapter may independently implement
resize rules, model output decoding, or detection postprocessing. Backend-specific
kernels may differ, but their observable semantics need common tests.

## 3. Component boundaries

The core owns validated image descriptions, coordinate/result types, error
categories, and ownership rules. The detection pipeline owns preprocessing,
model execution coordination, and postprocessing. Backend adapters own tensors,
device contexts, and backend-specific allocations; these types stay private.
The FFI layer validates the foreign-call contract and translates public types.
Bindings own host-language lifetime management and error translation.

The first design does not prescribe separate crates for every conceptual unit.
Start with cohesive modules; split a crate when dependency isolation requires it.
No public `Burn`, `LibTorch`, ONNX Runtime, or GPU-SDK types are permitted.
Burn remains an evaluation candidate, not a selected dependency.

## 4. C ABI representation and compatibility

Use explicitly exported `extern "C"` functions, C-layout boundary records,
fixed-width integer status/tag fields, and opaque resource handles. Do not expose
Rust references, trait objects, `Vec`, `String`, generic types, or Rust enums.
Do not use platform-dependent C `long`, implicit boolean marshalling, bitfields,
variadic functions, or packed records in the interface. Rust's default layout is
not a stable foreign-layout contract [R1, R2].

Use `uint32_t` for dimensions and tags, `int32_t` for statuses, and `size_t` for
buffer lengths/capacities in the target process. Detection coordinates/scores use
32-bit floating point. These are in-process records, not a portable serialization
format. SDKs must map `size_t` to the target pointer width, including WASM32.

Each extensible descriptor begins with `uint32_t struct_size` and
`uint32_t struct_version`. Version 1 fields and offsets are frozen only when the
first ABI is approved and published. A callee reads only the supported prefix,
rejects a prefix too small for that version, and rejects unsupported versions.
Append-only optional fields require documented defaults; new mandatory semantics
require a new descriptor version. Unknown optional tails can be ignored, but
unknown required flags must produce `UNSUPPORTED_FEATURE`. Required flags live
in the existing validated prefix, not solely in an ignored tail.

The initial exported operation families are:

| Family | Contract |
| --- | --- |
| ABI/build query | Report ABI major/minor and build identity without model loading |
| Capabilities | Distinguish compiled backends from devices currently available |
| Model open/release | Consume caller-supplied artifact bytes into an owned model |
| Session create/release | Bind a model to explicit execution settings and limits |
| Detect | Borrow one image until return; produce an owned result on success |
| Result describe/copy/release | Read metadata and copy detections without rerunning inference |

These are the required operations, not a published header. Exact C declarations,
export names, constants, and compiler-generated layout tests belong in the reviewed
implementation plan. The API must not advertise working model loading until the
separate artifact/first-model specification and implementation exist.

ABI major/minor, package version, model architecture version, and artifact schema
version are separate. A major ABI mismatch fails before resource creation. Minor
extensions preserve old call behavior. An old SDK/new library compatibility test
is required before claiming backward binary compatibility.

## 5. Resource ownership and lifetime

| Resource | Owner | Lifetime and release |
| --- | --- | --- |
| Artifact input bytes | Host | Borrowed only during model open; retained bytes are copied or decoded into library ownership |
| Model handle | Library; host holds one owning handle | Release via the matching library operation |
| Session handle | Library | Holds an internal strong reference to required model state |
| Image view and its bytes | Host | Borrowed read-only until the synchronous detection call returns |
| Result handle | Library | Immutable; owns its metadata/detections and outlives model/session handles |
| Copied result array | Host | Host allocates and frees; library writes within validated capacity |
| Diagnostic text buffer | Host | Used only during that call; never stored by the library |

Releasing a model handle does not invalidate sessions already created from it.
Result data does not borrow session scratch memory or host input pixels. Releasing
a session does not invalidate a completed result. Actual device-weight sharing
is an implementation detail; the API does not promise one physical allocation
across all sessions or devices.

The allocating library frees its own resources. The host must never `free` a
library handle. Release operations consume a pointer-to-handle slot, clear a valid
released slot to null, and accept a null value in a valid slot as a no-op.
A null slot pointer is an invalid argument. This does not make stale copied handle
values safe: forged pointers, wrong-kind handles, use-after-free, and concurrent
release/use violate the raw C contract and are not guaranteed recoverable errors.

Safe SDK objects prevent repeated release and keep resources alive for the entire
native call. They serialize or reject a close that races with a call; they must
not free memory underneath native execution. All resources must be gone before
unloading their native library. A process must not exchange handles between
different loaded copies of the core, including statically linked copies.

The first interface does not expose custom allocators or borrowed result pointers.
An owned result plus an optional host copy is preferred to an ambiguous zero-copy
promise. Later zero-copy or device-buffer APIs require a separate lifetime contract.

## 6. Image and result semantics

Accept top-down packed RGB8, BGR8, and RGBA8 host images. Width/height are positive;
row stride is positive and measured in bytes. RGBA8 means straight, not
premultiplied, RGB channels; alpha is ignored, not composited. Negative strides,
planar YUV, encoded JPEG/PNG, device pointers, EXIF rotation, and implicit color
profile conversion are outside this first interface. Hosts must supply oriented,
sRGB-encoded pixels with the declared channel order.

The descriptor carries data pointer, byte length, width, height, row stride, and
pixel-format tag. Before access, validate the enum tag and checked arithmetic:

```text
row_bytes = width * channels
required_bytes = (height - 1) * row_stride + row_bytes
row_stride >= row_bytes
byte_length >= required_bytes
```

Every operation is overflow-checked in the target address space; a derived Rust
slice length must also fit its `isize::MAX` constraint. Explicit image-byte/pixel
limits are applied before allocation or preprocessing. A non-null pointer and
claimed length cannot prove that foreign memory is actually mapped: the caller
must supply a readable allocation covering the declared range. Output and input
regions must not overlap. Numeric validation does not sandbox an unsafe C caller.

The host must not modify, free, or detach borrowed storage before return. When a
GPU backend is eventually added, a returned call must leave no pending access to
host input memory. Borrowing does not promise zero internal copies.

The model artifact defines resize/letterbox, interpolation, rounding, padding,
normalization, tensor layout, class mapping, score computation, and postprocessing.
No SDK may guess those details from a model filename or use its own defaults.

Public boxes use `(x_min, y_min, x_max, y_max)` in the oriented original image's
pixel-edge coordinate system: origin top-left, x rightwards, y downwards, maxima
exclusive. Clip to `[0,width]` and `[0,height]`, then discard zero-area boxes.
Coordinates and scores must be finite. Class IDs are zero-based indices into the
artifact's class table. Score range is `[0,1]`; scores are not promised to be
calibrated probabilities. Invalid model outputs fail with `MODEL_OUTPUT_INVALID`
rather than becoming plausible-looking detections. A valid no-detection result
is success with count zero.

Detection options include a finite score threshold in `[0,1]` and positive
`max_detections`, bounded by the session limit. Return scores at or above the
threshold. Sort by descending score, then class ID, then lexicographic coordinates,
with source candidate index as the final tie-breaker. Apply the result cap after
model-defined suppression; record a truncation flag when that cap removes boxes.
Model-defined suppression is not itself result truncation. Near-threshold or
near-tied results can differ across numerical backends; cross-backend tests use
documented matching/tolerances rather than an unconditional bitwise-equality claim.

Result copy accepts a host destination, capacity, and required-count output.
A null destination with zero capacity is a size query and returns success with
the required count, without copying. Other insufficient-capacity calls return `BUFFER_TOO_SMALL`, reports required count, and writes no detections.
Success writes exactly that many records. Retries copy the existing result, not
rerun the detector. Results include source dimensions, model identity, actual
backend/device/precision, and truncation status.

## 7. Sessions, concurrency, and resource limits

The session state machine is `READY -> RUNNING -> READY`, or
`RUNNING -> POISONED` after a caught panic or an unrecoverable backend fault.
A poisoned session cannot run again and must be released. Invalid input rejected
before execution does not poison the session. An ordinary backend error returns
to READY only when its adapter guarantees that state remains valid.

One session admits one detection call at a time. An overlapping call returns
`BUSY` rather than entering an unbounded queue. Admission must be protected before
forming exclusive references to mutable internals. Distinct sessions may run
concurrently; backend limitations must be explicitly serialized or reported.
Immutable model/result queries can be concurrent while the caller guarantees
that the queried handle is not released. Session configuration is immutable.

Release is not a cancellation mechanism. The caller waits for active calls before
releasing that handle. The first core API provides no asynchronous jobs, callbacks,
cancellation, deadlines, background downloads, or application task scheduler.
Native compute runtimes may have worker threads, but their budgets and lifetimes
must be documented; importing the library alone must not initialize a GPU.

Session creation requires positive limits for accepted image bytes, pixels, and
result count, and an explicit CPU thread budget. Model loading separately limits
artifact byte size and parser complexity. Zero is invalid, not an unlimited
sentinel. These bound library-controlled input/output work, not all process RSS,
GPU-driver allocations, backend caches, or host-retained result objects. Do not
advertise a global memory ceiling without backend-specific enforcement evidence.

Device and precision choice are explicit. A required unavailable device fails.
Fallback happens only under an explicit allowed policy, preserves requested
precision, and is recorded in session/results. Backend-internal mixed placement
must also be disclosed; an adapter unable to enforce requested placement cannot
claim strict GPU-only execution. No silent GPU-to-CPU or precision fallback.

## 8. Errors and panic boundary

Public calls use numeric status values and caller-provided output storage.
Calls returning owned handles require valid output slots that start null. A
nonempty slot is rejected without changing it; other failures leave the initially
empty slot null. Never overwrite an existing resource while reporting an error.
Expected categories include `INVALID_ARGUMENT`, `UNSUPPORTED_ABI`,
`UNSUPPORTED_FEATURE`, `UNSUPPORTED_MODEL`, `DEVICE_UNAVAILABLE`, `BUSY`,
`LIMIT_EXCEEDED`, `BUFFER_TOO_SMALL`, `MODEL_OUTPUT_INVALID`, `BACKEND_ERROR`,
`POISONED`, and `INTERNAL_ERROR`. Unknown future nonzero statuses remain failures
in every SDK. Null checks happen before dereferencing a nullable argument.

Diagnostics are per-call UTF-8 in host-provided storage, never a process-global
last-error string. With positive capacity, always NUL-terminate; report required
byte length excluding NUL and truncate only at a UTF-8 boundary. Zero capacity
permits no text output. The main status remains authoritative when diagnostics
are absent or truncated. Diagnostic truncation must not turn successful inference
into failure. Diagnostics must not contain image bytes, credentials, or full
model contents.

Expected failures use Rust `Result`, not panics. Unwinding panics are contained
inside exported C entry points and translated when safe; use an unwind-enabled
native library build for this behavior. Session invariants must be restored or
marked poisoned. Do not replace the host's global panic hook, logger, or signal
handlers. Foreign C++ exceptions must be translated inside the backend boundary.

`catch_unwind` does not catch aborting panics, and dropping a panic payload can
itself panic [R3]. The implementation must explicitly handle that secondary path.
Fallible allocation is preferred for controlled buffers, but universal recovery
from allocation failure, stack overflow, segmentation faults, driver failure,
or process abort is not promised. Process isolation remains a separate option,
not a safety property supplied by this C ABI.

## 9. Bindings and deployment

C# uses typed P/Invoke declarations, `LibraryImport` where supported, and
`SafeHandle` ownership. Correctly map integer widths and record layouts; pin or
copy buffers for the full call. Native AOT is an acceptance test, not a current
support claim. This direction follows Microsoft's interop guidance [R4].

Go uses cgo, typed wrappers, and explicit idempotent `Close()`. Pass byte storage
that satisfies Go's pointer rules; do not retain Go pointers beyond the call.
A future asynchronous wrapper must use owned/copied native storage unless an
explicitly reviewed pinning contract is added. Finalization is a fallback, not
the primary release mechanism [R5].

Node uses a pinned supported Node-API surface rather than V8 internal APIs.
A future Promise wrapper copies input before queued work, retains session ownership,
and uses bounded admission; it does not borrow a JavaScript buffer across arbitrary
await points. It must release result storage on success, rejection, and shutdown.
Node-API stability does not guarantee stability of external native dependencies
[R6]. Such asynchronous binding behavior is later work, not part of the first
synchronous core implementation.

Browsers use a separate WASM adapter, optionally with `wasm-bindgen` [R7]. Share
Rust semantics where supported; do not promise that desktop GPU libraries compile
to a browser. Bound WASM memory and account for copying or view invalidation on
memory growth. Browser inference/backend selection needs its own validation;
this document does not add it to the first native milestone.

Native releases are per OS/architecture/ABI and disclose required native libraries.
Keep CPU and accelerator dependency choices separable. No Python, .NET, Go, or
JavaScript runtime is required by the Rust core itself; host bindings and selected
backends bring their separately documented requirements. Never promise one universal
binary, a fully static GPU deployment, or no runtime shader/kernel compilation.

## 10. Artifact contract and backend evaluation gate

An eventual model artifact must describe schema and architecture versions,
weight encoding/checksums, input constraints, preprocessing, outputs, class table,
score/suppression semantics, numerical policy, and required operators. Training
provenance and the explicit weight/data license record remain mandatory under
[MODEL_LICENSES.md](../../../MODEL_LICENSES.md).

The wire format and first network architecture are deliberately outside this
contract decision. Until that separate design is approved, foundation tests use
only test-scoped synthetic fixtures; production loaders must not accept undocumented
formats or import another detector's checkpoints. No executable pickle or embedded
custom code is evaluated. No automatic network resolution is allowed. Unknown
required schema/architecture features fail explicitly.

Before choosing a compute framework, demonstrate on the proposed first detector:
required forward/backward operators, loss/gradient correctness, small-dataset
overfit, checkpoint round-trip, independent validation evaluation, CPU/GPU numeric
agreement where claimed, native packaging, and license/dependency inventory.
Measure cold start, preprocessing, inference, postprocessing, p50/p95/p99 latency,
RSS/device memory, and repeated-session resource behavior under pinned conditions.
Acceptance budgets must be set before the comparative experiment, not selected
after seeing results. Do not assume model export exists because import exists.

This gate does not require a new tensor compiler or handwritten GPU kernels in
this milestone. It also does not approve Burn, ONNX Runtime, or any other backend
by naming them as candidates.

## 11. Acceptance evidence and sequencing

Approve this written contract first, then review an implementation plan. Initial
native-contract implementation must include a tiny C consumer and a thin C#
lifetime smoke test; Go and Node checks validate the same contract before those
SDKs are published. The first backend fixture is test-only and is never described
as a trained detector. Real model/training work requires its own specification.

Required implementation evidence:

| Area | Required evidence before the relevant support claim |
| --- | --- |
| ABI | Export listing; C/Rust/SDK size, alignment, offset tests; major mismatch and descriptor-version tests |
| Ownership | Model release with live session; result survival after session release; wrapper double-close protection; failed-create cleanup |
| Inputs | Channel order, padded stride, zero dimensions, short buffer, overflow, invalid tags, and resource-limit cases |
| Results | Original-coordinate mapping, empty result, finite values, deterministic ordering, cap flag, and capacity-query/copy behavior |
| Concurrency | Controlled overlapping calls yield BUSY; independent sessions; wrapper close/use races; poisoned-session rejection |
| Failures | Panic containment with safe cleanup; per-call diagnostic isolation; malformed model-output rejection |
| Resources | Repeated create/run/release with allocation accounting and memory plateau checks; distinguish allocator caches from live-object leaks |
| Bindings | Same approved fixture results and error categories in each supported language; explicit platform/build matrix |
| Provenance | Reviewed dependencies/notices; no Ultralytics code, ports, weights, or derived model artifacts |

Memory/undefined-behavior tools and fuzzing should exercise validly allocated but
malformed descriptors; passing arbitrary dangling pointers to a C API is not an
expected safe error-path test. Record commands, toolchains, fixtures, and actual
results. A docs check or test-only backend cannot establish detector accuracy,
GPU support, benchmark claims, or production readiness.

Owner review is requested for the synchronous scope, owned-result policy,
per-session admission rules, strict device selection, and ABI version discipline.
Approval of this document is not automatic approval of a later implementation
plan, backend, model architecture, or package release.

## 12. Primary technical references

References checked on 2026-10-12. The choices above are FactsYOLO proposals;
these sources document the interoperability constraints, not an existing
FactsYOLO implementation. Original document text uses the root Apache-2.0 license.

- R1: [Rust type layout](https://doc.rust-lang.org/reference/type-layout.html).
- R2: [Rust extern functions](https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier).
- R3: [Rust catch_unwind guarantees and limitations](https://doc.rust-lang.org/std/panic/fn.catch_unwind.html).
- R4: [.NET native interoperability best practices](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices).
- R5: [Go cgo pointer passing](https://pkg.go.dev/cmd/cgo#hdr-Passing_pointers).
- R6: [Node-API and ABI stability](https://nodejs.org/api/n-api.html#implications-of-abi-stability).
- R7: [wasm-bindgen guide](https://wasm-bindgen.github.io/wasm-bindgen/).
