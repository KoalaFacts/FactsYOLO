# FactsYOLO M1 Native Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline execution, or superpowers:subagent-driven-development when separate workers are actually available and selected. Implement task-by-task; steps use checkbox syntax. Do not invent an independent reviewer or a passing command.

**Goal:** Build a testable native ABI and resource-lifetime foundation, exercised by C and C#, without presenting a synthetic fixture as a trained detector.

**Architecture:** One Rust library contains cohesive image, result, session, and FFI modules. An explicitly test-only feature supplies deterministic fixtures and fault controls. A development runner builds isolated normal/test artifacts and runs native consumers; real model loading, training, and compute backends remain separate work.

**Tech Stack:** Rust 2024, standard library only for M1 runtime code, C11 consumers, a .NET 10 C# smoke consumer using LibraryImport and SafeHandle, and GitHub Actions. No ML framework, image codec, async runtime, or third-party detector is selected.

**Spec:** [Native interface, ownership, and session design](../specs/2026-10-12-native-contract-design.md), approved by the owner after review and merged in PR #2 as `592a719b0602263f0f47304c2fd1bfff055b0bb1` on 2026-10-12. Its original proposed-status wording records the submission stage; the PR review/merge records the subsequent acceptance.

**Plan status:** Proposed for owner review. This document does not implement code, approve itself, freeze an ABI, publish a package, or select a backend. The names, constants, and test budgets below are proposed implementation decisions, not features already available.

## Global constraints

- Preserve the independent Rust core and [Apache-2.0/provenance baseline](../../decisions/0001-project-and-license.md). No Ultralytics code, ports, weights, or derived model artifacts.
- From the spec: "The first interface does not expose custom allocators or borrowed result pointers."
- From the spec: "One session admits one detection call at a time." Overlap returns BUSY; admission precedes mutable access.
- From the spec: "A null slot pointer is an invalid argument." A valid release slot containing null is a no-op; stale copied handles are not safe.
- From the spec: "Zero is invalid, not an unlimited sentinel." Image-byte, pixel, result, parser-work, artifact-size, and CPU-thread limits are explicit positive values.
- From the spec: "No silent GPU-to-CPU or precision fallback." M1 implements strict placement only; other policies fail explicitly.
- Host image storage is synchronously borrowed; completed results own their data and survive model/session release. No pending host-buffer access may remain after return.
- Dimensions/tags are uint32_t, statuses int32_t, lengths/capacities size_t, and coordinates/scores float32. No public backend types, C long, Rust enums, implicit booleans, packed records, or cross-library handles.
- First image formats are top-down RGB8/BGR8/RGBA8, straight RGBA with ignored alpha, positive stride, oriented sRGB pixels. No image-file decoding, YUV, GPU pointers, implicit orientation, or color conversion.
- Native unwind-enabled builds contain recoverable unwinding panics; abort, invalid foreign memory, driver crashes, and universal OOM recovery are not guaranteed.
- M1 is not a full detector. Normal builds expose zero executable backends and reject otherwise valid model-open requests with UNSUPPORTED_MODEL. The test fixture is never a production model format.
- Mandatory M1 acceptance targets: Linux x64 GNU and Windows x64 MSVC, including C# JIT and Native AOT smoke runs. These are targets to verify, not current support claims. Go, Node, browser inference, macOS/ARM execution, and real backward-compatibility claims remain gated separately.
- Proposed minimum Rust version: 1.85.0, which supports edition 2024 [P1]. At execution start, select and record exact currently supported stable Rust and .NET 10 SDK patches in toolchain files; test the Rust floor separately. Do not call the floor the latest compiler.
- Keep package publication disabled (`publish = false` for Rust and `IsPackable=false` for C#). Do not add release workflows or change branch protections in this milestone.

## Review focus

1. Physically short descriptor prefixes must be rejected before a full Rust reference or record read is formed; test prefix-only allocations and larger ignored tails in Task 1.
2. A nonempty owning output slot must never be overwritten, even on diagnostic/validation failure; test preservation and allocation counts in Tasks 1 and 3.
3. A .NET Dispose/Detect race must not release native storage during execution; test deterministic barriers and a live-handle lease in Task 7.
4. A panic payload whose destructor panics must not cause a second unwind across C; test the nested cleanup path in an isolated process in Task 5.
5. Fixture feature unification or stale build paths must not leak a fake backend into a normal artifact; test distinct output directories and negative symbol/capability checks in Tasks 6 and 8.

## Scope, files, and execution order

This plan implements the native-contract portion of the spec. It does not implement its later real-model/backend gate. Use eight sequential, reviewable tasks. Prefer one PR per task; review and merge the current task before starting the next. Each task gets its own red/green cycle and a focused commit. Do not create eight empty branches or issues in advance.

Proposed repository files, created only during implementation:

| Path | Responsibility |
| --- | --- |
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `.gitignore` | Two-member workspace, reproducible tools, ignored build output |
| `crates/facts-yolo/Cargo.toml`, `src/lib.rs` beneath it | `facts-yolo` package, `facts_yolo` rlib/cdylib/staticlib; feature defaults empty |
| `crates/facts-yolo/src/ffi/{mod,types,descriptors,diagnostics,handles,exports}.rs` | Narrow unsafe boundary, prefix reads, numeric statuses, exports |
| `crates/facts-yolo/src/{error,image,model,session,result}.rs` | Safe internal value types and single-source behavior |
| `crates/facts-yolo/src/fixtures.rs` | Feature-gated fixture factory, counters, and barriers only |
| `crates/facts-yolo/tests/{abi,images,lifetimes,results,concurrency,failures}.rs` | Rust integration tests; unit tests sit beside private helpers |
| `include/facts_yolo.h` | Hand-maintained C declarations, checked against Rust layouts |
| `tests/c/{abi,contract,threads}.c`, `tests/fixtures/facts_yolo_test.h` | Real C boundary checks and test-only controls |
| `tests/dotnet/FactsYOLO.Smoke/{FactsYOLO.Smoke.csproj,Native,Handles,Program}.cs` | Thin unpublished C# lifetime/ABI consumer; the csproj is the project file, not a .cs file |
| `tests/dotnet/global.json` | Exact .NET 10 SDK pin recorded during execution |
| `xtask/Cargo.toml`, `xtask/src/{main,native,dotnet,resources}.rs` | Standard-library development runner; no production CLI |
| `.github/workflows/native-contract.yml` | Required build/test matrix, no release action |
| `docs/testing/native-contract.md`, `docs/testing/m1-evidence.md` | Reproduction instructions and measured evidence |

All listed modules use the workspace root paths above. Start with modules rather than a separate crate for every concept. `xtask` may invoke installed toolchains but must fail clearly when a required tool is unavailable; it must not silently mark a skipped test as passing.

## Proposed C contract to implement

This section supplies exact boundary decisions missing from the design spec. The initial development ABI reports major 1, minor 0, and descriptor version 1. These values remain provisional until the first explicitly approved ABI release. A future release cannot claim compatibility merely because these draft numbers existed.

### Status and tag values

| Symbol, all prefixed FY_ | Value |
| --- | --- |
| OK | 0 |
| INVALID_ARGUMENT, UNSUPPORTED_ABI, UNSUPPORTED_FEATURE, UNSUPPORTED_MODEL | 1, 2, 3, 4 |
| DEVICE_UNAVAILABLE, BUSY, LIMIT_EXCEEDED, BUFFER_TOO_SMALL | 5, 6, 7, 8 |
| MODEL_OUTPUT_INVALID, BACKEND_ERROR, POISONED, INTERNAL_ERROR | 9, 10, 11, 12 |
| RGB8, BGR8, RGBA8 | 1, 2, 3 |
| DEVICE_CPU, DEVICE_GPU; PRECISION_F32; FALLBACK_NONE | 1, 2; 1; 0 |
| RESULT_TRUNCATED | bit 0 |
| CAP_COMPILED, CAP_AVAILABLE, CAP_TEST_ONLY | bits 0, 1, 2 |
| BACKEND_TEST_FIXTURE | 0xffff0001; test build only |

Use integer constants/macros, not C enums. Unknown nonzero status values are failures in wrappers. All initial required_flags fields accept zero only; unknown bits return UNSUPPORTED_FEATURE.

### Boundary record field order

Include stdint.h and stddef.h, C++ extern-C guards, and a FY_API import/export macro: Windows uses `__declspec(dllexport)` while building the DLL and `__declspec(dllimport)` for dynamic consumers; FY_STATIC removes this decoration. Non-Windows uses normal C linkage and export visibility. Do not apply packing pragmas. `fy_model`, `fy_session`, and `fy_result` are distinct opaque struct types.

In the following table H expands in place to `uint32_t struct_size; uint32_t struct_version; uint32_t required_flags;` in that order. Fields use natural target C layout, including padding.

| Record | Fields after H, in order |
| --- | --- |
| `fy_abi_info` | `uint32_t abi_major, abi_minor; uint8_t build_id[64];` |
| `fy_model_options` | `size_t max_artifact_bytes, max_parser_steps;` |
| `fy_session_options` | `uint32_t backend_id, device_kind, device_index, precision, fallback_policy, cpu_threads; size_t max_image_bytes, max_pixels, max_detections;` |
| `fy_image` | `const uint8_t *data; size_t byte_length; uint32_t width, height; size_t row_stride; uint32_t pixel_format;` |
| `fy_detect_options` | `float score_threshold; size_t max_detections;` |
| `fy_session_info` | `uint32_t backend_id, device_kind, device_index, precision, execution_flags;` |
| `fy_result_info` | `uint32_t source_width, source_height; size_t count; uint32_t result_flags, backend_id, device_kind, device_index, precision, execution_flags; uint8_t model_id[32];` |
| `fy_diagnostic` | `char *data; size_t capacity, required_bytes;` |

Two fixed, non-extensible array records have no H: `fy_detection` has `float x_min, y_min, x_max, y_max, score; uint32_t class_id;`; `fy_capability` has `uint32_t backend_id, device_kind, device_index, flags, precision_mask;`. Do not append fields to array elements in a minor ABI update; use a new operation/record version. F32 precision uses bit 0 in precision_mask. A compiled backend without an available device uses device_index UINT32_MAX and no CAP_AVAILABLE bit. Normal M1 has no capability records.

build_id is NUL-terminated ASCII: the build runner supplies the checked-out Git SHA through FACTSYOLO_BUILD_ID, maximum 63 bytes; local builds without it report `local-unversioned`. model_id is an opaque 32-byte equality token, not an approved checksum algorithm or artifact wire format. Fixtures use the fixture ID in byte zero and zero for remaining bytes. M1 execution_flags is zero; requesting fallback or unsupported precision fails rather than inventing execution metadata.

### Function signatures

Every declaration returns `int32_t` and has FY_API. Every final `fy_diagnostic *diag` is optional. Initial output descriptors are host-initialized with H; fill only the supported prefix and preserve unrecognized tail storage. `const` on resource pointers promises no public mutation, not permission to release them during a call.

```c
fy_abi_query(fy_abi_info *out, fy_diagnostic *diag);
fy_capabilities_copy(fy_capability *dst, size_t capacity,
                     size_t *required, fy_diagnostic *diag);
fy_model_open(uint32_t required_abi_major, const uint8_t *bytes, size_t length,
              const fy_model_options *options, fy_model **out,
              fy_diagnostic *diag);
fy_model_release(fy_model **slot, fy_diagnostic *diag);
fy_session_create(uint32_t required_abi_major, const fy_model *model,
                  const fy_session_options *options, fy_session **out,
                  fy_diagnostic *diag);
fy_session_describe(const fy_session *session, fy_session_info *out,
                    fy_diagnostic *diag);
fy_session_release(fy_session **slot, fy_diagnostic *diag);
fy_detect(fy_session *session, const fy_image *image,
          const fy_detect_options *options, fy_result **out,
          fy_diagnostic *diag);
fy_result_describe(const fy_result *result, fy_result_info *out,
                   fy_diagnostic *diag);
fy_result_copy(const fy_result *result, fy_detection *dst, size_t capacity,
               size_t *required, fy_diagnostic *diag);
fy_result_release(fy_result **slot, fy_diagnostic *diag);
```

These are signature specifications inside a plan, not a compilable or released header. The implementation supplies return types/declarations and all safety documentation. Query functions and SDK startup enforce major compatibility before resource creation; both model/session creation recheck required_abi_major.

### Validation and failure behavior

Validate diagnostic storage first. Null diag disables diagnostics. For non-null diag, require a readable/writable H prefix, supported version/size/flags, a writable advertised buffer when capacity is positive, and disjoint storage. A malformed diagnostic returns its validation status without touching other outputs; do not try to diagnose through invalid storage. For a valid diagnostic, success writes required_bytes=0 and a NUL when capacity>0. Error text is fixed, non-sensitive UTF-8; required_bytes excludes NUL, and truncation respects UTF-8 boundaries. A diagnostic shortage never replaces the operation status with BUFFER_TOO_SMALL.

For other descriptors, require at least the two-word header to be readable before inspecting struct_size. Reject too-small size or bad alignment with INVALID_ARGUMENT; unsupported struct_version returns UNSUPPORTED_ABI; unknown required flags return UNSUPPORTED_FEATURE. Read only verified prefixes through raw-pointer operations before making a typed value. Never form `&*descriptor` over an advertised short prefix [P2, P3]. Larger optional tails are ignored and never overwritten. The caller remains responsible for actual allocation validity, initialized input bytes, synchronization, and nonoverlap.

Creation output slots are writable, valid, and initially null. Reject a nonempty slot unchanged. On all other failures an initially empty slot remains null. Commit successful ownership to the slot only after all fallible work finishes. Query/info outputs remain unchanged on failure; copy operations are the exception: a valid required-count output reports the full count on BUFFER_TOO_SMALL, while the destination remains untouched. Null destination+zero capacity is a successful count query. Null destination with positive capacity is INVALID_ARGUMENT. A non-null destination with zero capacity succeeds only when the result is empty, otherwise BUFFER_TOO_SMALL. Check capacity byte multiplication and alignment before writing.

Default model_open validates ABI, descriptors, null/length rules, positive limits, and byte limits, then returns UNSUPPORTED_MODEL without parsing, allocating a model, evaluating code, or making network calls. A future supported format requires another approved specification. Do not add a production magic-byte shortcut for fixtures.

## Task 1: ABI representation, version queries, and isolated diagnostics

**Files:** Create workspace/toolchain metadata, crate manifest/lib, `ffi/{mod,types,descriptors,diagnostics,exports}.rs`, `error.rs`, `include/facts_yolo.h`, `tests/abi.rs`, and the xtask manifest/main. Establish `default=[]`, `contract-fixtures=[]`, `panic="unwind"`, and `unsafe_op_in_unsafe_fn="deny"`; no dependency crate is needed. Record exact installed stable toolchains, not a guessed future patch.

**Interfaces:** Produce all repr-C record/status definitions above and implement fy_abi_query. Internal `Error { status: i32, message: &'static str }`; `unsafe fn read_input<T: Copy>(ptr: *const T) -> Result<T, Error>` is only for the explicit boundary-record implementations whose H, full size, alignment, and required flags it validates; do not expose it as an unconstrained general-purpose safe API. `unsafe fn write_diagnostic(diag: *mut FyDiagnostic, status: i32, message: &str) -> Result<(), Error>` follows the prevalidation rules. Use type names FyImage, FyDetection, etc. consistently for the corresponding C records.

- [ ] **Red:** Add tests `abi_query_reports_1_0`, `prefix_only_allocation_rejected`, `unknown_version_rejected`, `unknown_required_flag_rejected`, `larger_tail_unchanged`, and `diagnostics_are_per_call`. Assert `(major,minor)==(1,0)`, status 1/2/3 as specified, tail sentinel unchanged, and isolation of simultaneous caller buffers. For message `éX`, capacity 2 writes only NUL and required_bytes=3; capacity 3 writes `é\0`; capacity 0 writes no text. Bad diag leaves an output sentinel untouched.
- [ ] **Confirm red:** `cargo test -p facts-yolo --test abi` must fail on the absent behavior, not a missing toolchain. A minimal scaffold may compile, but do not write successful behavior before the failing assertions.
- [ ] **Implement:** Add narrow prefix-validation/diagnostic helpers and the query export with `#[unsafe(no_mangle)] unsafe extern "C"`. Use explicit unsafe blocks with documented preconditions [P1]. No backend initialization at library load.
- [ ] **Green:** Repeat the ABI test; run `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`. Record commands and output.
- [ ] **Commit/review:** `feat: establish native ABI query and diagnostics`; include manifests, header, helpers, and their tests in the same reviewable change.

## Task 2: Checked image views and channel-order semantics

**Files:** Create `image.rs` and `tests/images.rs`; extend error/descriptor helpers only as needed.

**Interfaces:** Produce `ImageLimits { max_bytes: usize, max_pixels: usize }`, `ImageView<'a>` owning only metadata plus a borrowed byte slice, `unsafe fn ImageView::from_ffi(raw: &FyImage, limits: ImageLimits) -> Result<ImageView<'_>, Error>` with a documented caller allocation/lifetime obligation, and safe `rgb_at(&self, x: u32, y: u32) -> Option<[u8;3]>`. Tie the returned slice lifetime to an explicit caller-owned borrow in internal safe callers; keep raw FFI construction unsafe. Do not return a fabricated unconstrained safe lifetime.

- [ ] **Red:** Assert a 2x2 RGB image with row_stride=8 needs exactly 14 bytes; 13 fails. Test RGB/BGR equivalence, RGBA alpha ignored without unpremultiplication, row padding not interpreted as pixels, zero dimensions/stride, bad format, limit boundaries, checked row/pixel arithmetic, and out-of-bounds rgb_at returning None. Use real allocated buffers; overflow metadata must be rejected before pixel access. A zero limit returns INVALID_ARGUMENT; an exceeded positive limit returns LIMIT_EXCEEDED.
- [ ] **Confirm red:** `cargo test -p facts-yolo --test images` shows the new missing behavior.
- [ ] **Implement:** Check width*channels, `(height-1)*stride+row_bytes`, width*height, address-space conversions, and slice isize::MAX limits before slice construction [P3]. Borrow only the needed accessible range, and reject a declared byte_length exceeding the explicit image-byte budget. No resize, normalize, decoder, or alternate SDK algorithm is introduced.
- [ ] **Green:** Repeat images and ABI tests. Add a deterministic bounded descriptor generator to unit tests; vary dimensions/tags/strides while byte ranges stay backed by allocations. Record its seed and case count.
- [ ] **Commit/review:** `feat: validate borrowed packed image views`.

## Task 3: Ownership, explicit session settings, and fixture isolation

**Files:** Create `model.rs`, the ownership portion of `session.rs`, `ffi/handles.rs`, `fixtures.rs`, and `tests/lifetimes.rs`; implement model/session/capability exports.

**Interfaces:** Internal `Model` is immutable and retained with Arc; `Session` owns an Arc<Model> and immutable validated settings. `fn Session::new(model: Arc<Model>, options: FySessionOptions) -> Result<Session, Error>` checks backend/device/precision and all limits before allocation. Result ownership is introduced in Task 4. Default model_open never succeeds in M1. Only `contract-fixtures` enables `fy_test_model_create(uint32_t required_abi_major, uint32_t fixture_id, fy_model **out, fy_diagnostic *diag)` returning int32_t. This helper is declared solely in `tests/fixtures/facts_yolo_test.h`, not the public header.

Fixture IDs are BASIC=1, EMPTY=2, INVALID_OUTPUT=3, BLOCKING=4, PANIC=5, PAYLOAD_DROP_PANIC=6. Each has two classes and a 2x2 input constraint. The fixture backend is CPU index 0, F32, one execution thread, no fallback. Require cpu_threads=1 for it; other positive thread requests return UNSUPPORTED_FEATURE. Zero thread budgets are INVALID_ARGUMENT. Session tests use max_image_bytes=1024, max_pixels=64, max_detections=16. Fixture IDs are in-process testing choices, never bytes accepted by model_open.

- [ ] **Red:** Test `release_model_with_live_session`, `release_null_value_is_noop`, `null_slot_is_invalid`, `nonempty_output_is_preserved`, `failed_create_has_no_live_object_delta`, `major_mismatch_before_creation`, and `strict_unavailable_device_rejected`. Assert slot clearing, a retained model state after its public handle closes, correct 1/2/3/5 statuses, and zero object-count delta on failure. Unknown nonzero statuses remain failures in test consumers.
- [ ] **Confirm red:** `cargo test -p facts-yolo --features contract-fixtures --test lifetimes` fails for the missing lifetime behavior.
- [ ] **Implement:** Use distinct boxed handle wrappers with strong internal ownership. Release via pointer-to-slot without accepting forged/wrong-kind/stale handles as safe input. Model/session destructors are non-panicking. Normal capabilities are empty; test capabilities have compiled+available+test-only flags, CPU index 0 and F32 bit 0. M1 fallback_policy!=0 returns UNSUPPORTED_FEATURE. Report unavailable required devices rather than falling back.
- [ ] **Green:** Run lifetime/ABI/image suites both with and without the feature where applicable; assert normal model_open is UNSUPPORTED_MODEL and produces no handle. Do not publish fixture helpers.
- [ ] **Commit/review:** `feat: add owned model and session contracts with test fixtures`.

## Task 4: Owned results, coordinate mapping, and bounded result copies

**Files:** Create `result.rs` and `tests/results.rs`; complete the synchronous detect path and result exports, using the ownership types from Task 3.

**Interfaces:** `Candidate { x_min: f32, y_min: f32, x_max: f32, y_max: f32, score: f32, class_id: u32, source_index: usize }`; `ImageTransform { scale_x: f32, scale_y: f32, pad_x: f32, pad_y: f32 }`; `fn finalize(candidates_after_suppression: &[Candidate], transform: ImageTransform, source: (u32,u32), class_count: u32, threshold: f32, max_detections: usize) -> Result<Finalized, Error>`, where `Finalized { detections: Vec<FyDetection>, truncated: bool }`. `OwnedResult` contains those values and a complete FyResultInfo; it borrows neither input nor session. Model-specific suppression is not implemented here; fixture candidates are explicitly post-suppression.

Canonical BASIC fixture: transform=(2,2,1,1), candidates `(class,score,xmin,ymin,xmax,ymax)` are `(1,.9,-1,-1,3,3)`, `(0,.9,1,1,5,5)`, `(0,.5,3,3,5,5)`, `(0,.7,3,3,3,5)`, in source-index order. On a 2x2 image with threshold=.5 and cap=16, expected public outputs are class0/.9/(0,0,2,2), class1/.9/(0,0,1,1), class0/.5/(1,1,2,2); the zero-area candidate disappears. Cap=2 sets truncation. EMPTY returns success with zero records. INVALID_OUTPUT inserts NaN even below threshold and must fail, not silently filter the bad value.

- [ ] **Red:** Assert canonical values, threshold inclusivity, signed-zero normalization for stable ordering, deterministic full-tie source-index ordering, clipping before zero-area removal, and finite transform/output validation. Reject scores outside [0,1], invalid class indices, NaN/Infinity, and nonpositive transform scales. Invalid threshold/cap is INVALID_ARGUMENT; cap exceeding the session limit is LIMIT_EXCEEDED. Test a saved result after both model and session releases.
- [ ] **Confirm red:** `cargo test -p facts-yolo --features contract-fixtures --test results` fails on missing outputs/copy semantics.
- [ ] **Implement:** Invert `(coordinate-padding)/scale`, validate every candidate before filtering, clip and discard zero-area boxes, apply the documented ordering, then cap. Use fallible reservations where practical; do not promise universal allocator-failure recovery. Copy immutable results without invoking the fixture again. Require count-output storage and safe aligned destination capacity; preserve destinations on insufficient capacity.
- [ ] **Green:** Assert null+zero size query succeeds, count=3, capacity=2 reports BUFFER_TOO_SMALL with an untouched sentinel array, exact-capacity copy succeeds, empty copy succeeds, and repeated copies do not increase execution count. Result metadata records source 2x2, fixture identity, actual CPU/F32 settings, and truncation.
- [ ] **Commit/review:** `feat: add owned detection results and bounded copy operations`.

## Task 5: Admission control, poisoning, and panic-safe error paths

**Files:** Complete `session.rs`, add `tests/concurrency.rs` and `tests/failures.rs`, extend FFI guard/fixture test controls.

**Interfaces:** `fn Session::run(&self, image: ImageView<'_>, options: FyDetectOptions) -> Result<OwnedResult, Error>`. Session contains `Mutex<SessionInner>`; only shared outer references exist before try_lock. WouldBlock maps to BUSY, poisoned state to POISONED. A guard marks an admitted execution poisoned unless a valid completion explicitly restores READY. The lock is held through result finalization, not just candidate generation. Error classification distinguishes recoverable validation errors, recoverable backend errors, and unrecoverable execution faults.

Test-only int32-returning controls are `fy_test_session_wait_running(const fy_session*, uint32_t timeout_ms)` and `fy_test_session_resume(const fy_session*)`. They use an independent fixture latch so holding the session execution lock does not deadlock test control. A positive timeout bounds a failed test; timeout returns INTERNAL_ERROR. They are not production cancellation/deadline APIs.

- [ ] **Red:** With a barrier-confirmed BLOCKING call, assert a second detect returns BUSY without starting execution and without allocating a result; resume the first call and get success. Two sessions enter execution independently. Invalid input leaves a ready session usable. PANIC returns INTERNAL_ERROR with a null output; subsequent detect returns POISONED. Test per-call diagnostic isolation concurrently.
- [ ] **Confirm red:** `cargo test -p facts-yolo --features contract-fixtures --test concurrency --test failures` fails on the new admission/fault assertions. Use barrier state, not an arbitrary sleep, to prove overlap.
- [ ] **Implement:** All public C entry points use a common catch_unwind boundary while normal failures use Result. Do not replace the process panic hook. Avoid including panic payloads or pixel/model bytes in diagnostics. Drop a caught payload inside a second catch_unwind; if that destructor panics, forget only the secondary payload and report INTERNAL_ERROR. Explicitly document this exceptional retained payload; normal-path allocation accounting must not hide it [P4]. Double panics during stack unwinding/aborts remain outside the recovery promise.
- [ ] **Green:** Run a subprocess probe for PAYLOAD_DROP_PANIC and assert the child reaches its post-call marker rather than aborting. Verify the session is poisoned and safely releasable. Test cleanup after every failed-create/run path, with no incorrect READY transition after an unrecoverable fault. Native releases must use an unwind-enabled profile.
- [ ] **Commit/review:** `fix: enforce session admission and contain recoverable FFI panics`.

## Task 6: Real C ABI conformance and build separation

**Files:** Create `tests/c/{abi,contract,threads}.c`, test-only header, and `xtask/src/native.rs`; extend xtask command routing.

**Interfaces:** `cargo run -p xtask -- c` builds normal and fixture libraries into `target/normal` and `target/contract-fixtures`, respectively, then compiles/runs C consumers against each exact path. Use the selected target's system C compiler, pthreads on Linux and Win32 threads on Windows for the barrier smoke test. `cargo run -p xtask -- exports` compares defined exported fy_* symbols against an explicit public-symbol list; fixture names must not appear in normal artifacts. No external code generator is necessary for M1.

- [ ] **Red:** C asserts sizeof/_Alignof/offsetof for every public record and field against a Rust-generated line-oriented layout manifest. Include a frozen copy of the initial C declarations within the tests for future compatibility regression use; this is not yet an old released SDK. Use guard bytes around prefix/output buffers and all 11 functions at least once. Test major mismatch, bad tags, count queries, error slots, normal loader rejection, canonical results, ownership, and controlled BUSY.
- [ ] **Confirm red:** Run `cargo run -p xtask -- c`; it must fail on a specific missing/mismatching symbol or assertion, not pass because no C program was executed.
- [ ] **Implement:** Rust layout manifest uses size_of/align_of/offset_of, not handwritten assumptions about 32/64-bit padding. Support dynamic C linking on both required targets and static C linking at least on Linux. Document native runtime/system library link requirements. Cdecl/default target C ABI must match the Rust exports. Do not compare symbol substrings from debug strings with actual exported symbols.
- [ ] **Green:** Run c and exports commands; assert separate normal capabilities count=0 and fixture count=1 with CAP_TEST_ONLY. Neither normal nor fixture model_open accepts a testing byte format. A bad fixture path must fail rather than loading any arbitrary same-named library from the environment.
- [ ] **Commit/review:** `test: verify the native contract from C consumers`.

## Task 7: Thin C# lifetime smoke consumer, including Native AOT

**Files:** Create the files under `tests/dotnet/FactsYOLO.Smoke/` specified above and `tests/dotnet/global.json`; add `xtask/src/dotnet.rs`.

**Interfaces:** Target net10.0 with unsafe code enabled and IsPackable=false. Native methods mirror the exact C signatures with uint/int/nuint/float and explicit Cdecl; avoid SetLastError and bool marshalling. `ModelHandle`, `SessionHandle`, `ResultHandle` derive from SafeHandle with nonthrowing ReleaseHandle calling the corresponding pointer-to-slot native release on a local nint. Allocate an empty managed handle wrapper before native creation, adopt the returned pointer only on success, and clean up on every adoption exception. Keep native handles behind public wrapper objects, not caller-accessible raw IntPtr values.

`SessionLease` is a scoped live-handle lease: under a short wrapper lifecycle lock, reject a closed object, acquire DangerousAddRef, and take the pointer. The lock is not held during Detect. Release with DangerousRelease in finally. Dispose prevents new leases and may defer physical release until admitted calls finish. Apply the same pattern to model/session creation and result queries where needed. This protects memory without serializing all same-session Detect calls and hiding native BUSY [P5]. The wrapper exposes synchronous Detect(ReadOnlySpan<byte> pixels, uint width, uint height, nuint stride, uint format, float threshold, nuint maxDetections), returning an owned Result wrapper. No preprocessing is implemented in C#.

- [ ] **Red:** Assert C# sizeof/offset mapping matches the Rust/C manifest, every native nonzero status translates to failure, and fixture results/errors match the C consumer. Assert double Dispose is harmless, Detect after Dispose throws ObjectDisposedException, and a model disposed while a session lives does not break it.
- [ ] **Confirm red:** `cargo run -p xtask -- dotnet --mode jit` fails on the new assertions, not on a swallowed native-library load failure.
- [ ] **Implement:** Pin image spans only for the synchronous call; lease session handles for the entire native execution and cleanup. A Result wrapper owns its native result until disposal and copies data before exposing managed arrays. Use a resolver bound to the explicit absolute test-library path. Do not rely on GC.KeepAlive alone as a Dispose-race guard. Follow LibraryImport/SafeHandle guidance [P5].
- [ ] **Green:** While BLOCKING Detect is inside native code, call Dispose and force GC; assert the native session stays live until Detect returns, then is released once. Test a second Detect during execution returns BUSY. Run `cargo run -p xtask -- dotnet --mode aot --rid linux-x64` on Linux and the same command with win-x64 on Windows. Execute the published binary; compiling alone is not success [P6].
- [ ] **Commit/review:** `test: exercise native ownership from C# and Native AOT`.

## Task 8: Resource evidence, safety tooling, and merge gates

**Files:** Add `xtask/src/resources.rs`, workflow, testing/evidence documents; update README and THIRD_PARTY_NOTICES only to describe what this implementation actually contains.

**Interfaces:** `cargo run -p xtask -- resources` executes an isolated fixture process with per-kind live counters: model handles, retained model states, sessions, results, plus controlled buffer bytes. `cargo run -p xtask -- verify` orchestrates mandatory local-target checks and returns nonzero if any required check is missing/failing. `cargo run -p xtask -- manifest` writes revision, target, features, toolchains, fixture version, and commands into the evidence directory. No claim is inferred from an unexecuted command in this plan.

- [ ] **Red:** After 1,000 warm-up create/run/release cycles and 10,000 measured cycles with retained-result count fixed at zero, assert live object counts and controlled live buffer bytes return to their baseline after each batch. Sample process resident/private memory with the documented platform API, not cumulative peak memory. Proposed fixture-only regression budget: median of the final 10 batch samples minus median of the first 10 measured samples <=16 MiB, using 100-cycle batches. Report raw samples; this is not a production memory ceiling. Test output-slot errors and repeated count/copy calls in the same workload.
- [ ] **Confirm red:** Introduce a temporary retained-result leak in the test harness and show the live-counter gate fails; remove the injected leak before committing. Separately prove the runner fails on a missing mandatory tool and on a normal build exposing a fixture symbol.
- [ ] **Implement:** Required CI jobs on ubuntu-24.04 x64 and windows-2022 x64 run debug/release Rust tests, C conformance, export checks, C# JIT/AOT, and resource checks. Pin action commit SHAs and exact toolchain versions after verifying upstream sources; use read-only workflow permissions and pull_request/push triggers, not pull_request_target. Add an i686 Linux C/Rust layout probe for pointer-width coverage; it does not establish browser support. Keep all build features and paths explicit.
- [ ] **Green:** Run fmt/clippy/tests plus xtask verify on each required target. Run pure Rust descriptor/ownership tests under a pinned nightly Miri toolchain where supported; the dedicated gate must record the exact command and fail on real findings. It may exclude subprocess/OS-thread/foreign-call cases only with an explicit documented coverage boundary. Add a fixed-seed 10,000-case malformed-descriptor corpus with real backing allocations. Never fuzz by dereferencing arbitrary forged pointers. Any unavailable required job remains open, not green.
- [ ] **Evidence and review:** Record actual commands, exit codes, logs, Rust/C/C# layouts, export lists, memory samples, tested features, and all tool/runtime notices. Preserve the fixture panic-cleanup exception separately from normal-loop leak accounting. Update README to say `native contract tested; no trained detector or published SDK`. Request a fresh whole-branch review, address important findings, and merge only after owner-approved scope and required checks. Commit as `ci: gate native contracts and resource regressions`.

## Coverage and deferred acceptance

| Spec section | This plan's evidence / boundary |
| --- | --- |
| 1-3 Intent/components | Single Rust implementation, no selected backend; all tasks |
| 4 ABI | Tasks 1 and 6; exact declarations, exports, prefix/version/width probes; released backward compatibility still unclaimed |
| 5 Ownership | Tasks 3, 4, 6, 7, 8; unsafe raw preconditions remain explicit |
| 6 Images/results | Tasks 2 and 4; fixture-only transform proof, not real preprocessing or suppression |
| 7 Sessions/resources | Tasks 3, 5, 7, 8; strict-only placement, positive budgets, no async/cancellation |
| 8 Errors/panics | Tasks 1, 3, 5, 7; no universal process-crash/OOM containment claim |
| 9 Bindings/deployment | C and thin unpublished C# tested in Tasks 6-8; Go/Node/browser/macOS/ARM require later plans and evidence |
| 10 Artifact/backend gate | Deliberately deferred: no wire format, real network, weights, training, GPU, or backend dependency is added |
| 11 Acceptance | Tasks 1-8 supply native-contract evidence only; resource counts, numerical fixture semantics, and provenance are separate from detector benchmarks |
| 12 References | Retained in the spec, supplemented below |

Do not mark the whole vision library complete after M1. The next separate design must choose the first independently implemented detector, its artifact/training contract, and predeclared backend comparison budgets. Go/Node adapters may then be planned against the tested contract; each must pass shared fixtures before any SDK publication. Do not create that design while an M1 implementation PR is still awaiting merge.

## Plan self-review and handoff

Self-review checked every spec section against the coverage table, matching type names, all creation/copy failure rules, distinct test/normal builds, scoped lifetime leases, and per-task red/green verification. Proposed numerical budgets and exact ABI details above require plan approval. Acceptance of PR #2 alone is not that approval.

Recommended execution is sequential inline implementation with a fresh external PR review at each gate, since the tasks share ABI, ownership, and error contracts. Separate subagents are optional only when actually available and selected. Do not claim that an automated reviewer has completed a review until its current result is read. Stop here for owner review of this plan; no runtime code or dependency installation is part of submitting it.

## Primary references checked for this plan

These sources document language/tool behavior; the test values, architecture, and proposed API are FactsYOLO decisions.

- P1: [Rust 1.85 and edition 2024](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/).
- P2: [Rust type layout](https://doc.rust-lang.org/reference/type-layout.html).
- P3: [from_raw_parts safety requirements](https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html).
- P4: [catch_unwind limitations, including panic-payload destruction](https://doc.rust-lang.org/std/panic/fn.catch_unwind.html).
- P5: [.NET native interop best practices](https://learn.microsoft.com/en-us/dotnet/standard/native-interop/best-practices).
- P6: [Native AOT deployment](https://learn.microsoft.com/en-us/dotnet/core/deploying/native-aot/).
- P7: [.NET support policy](https://dotnet.microsoft.com/en-us/platform/support/policy/dotnet-core); target .NET 10, and verify the actual supported patch again when executing rather than assuming a cached patch table is current.

Original plan text is Apache-2.0. Prepared with AI assistance; no contributor certification or sign-off is asserted on anyone else's behalf.
