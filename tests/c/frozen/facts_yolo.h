/* SPDX-License-Identifier: Apache-2.0 */
#ifndef FACTS_YOLO_H
#define FACTS_YOLO_H
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && !defined(FY_STATIC)
# if defined(FY_BUILDING_DLL)
#  define FY_API __declspec(dllexport)
# else
#  define FY_API __declspec(dllimport)
# endif
#elif defined(__GNUC__)
# define FY_API __attribute__((visibility("default")))
#else
# define FY_API
#endif
#if defined(_WIN32)
# define FY_CALL __cdecl
#else
# define FY_CALL
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define FY_OK INT32_C(0)
#define FY_INVALID_ARGUMENT INT32_C(1)
#define FY_UNSUPPORTED_ABI INT32_C(2)
#define FY_UNSUPPORTED_FEATURE INT32_C(3)
#define FY_UNSUPPORTED_MODEL INT32_C(4)
#define FY_DEVICE_UNAVAILABLE INT32_C(5)
#define FY_BUSY INT32_C(6)
#define FY_LIMIT_EXCEEDED INT32_C(7)
#define FY_BUFFER_TOO_SMALL INT32_C(8)
#define FY_MODEL_OUTPUT_INVALID INT32_C(9)
#define FY_BACKEND_ERROR INT32_C(10)
#define FY_POISONED INT32_C(11)
#define FY_INTERNAL_ERROR INT32_C(12)
#define FY_ABI_MAJOR UINT32_C(1)
#define FY_ABI_MINOR UINT32_C(0)
#define FY_STRUCT_VERSION UINT32_C(1)
#define FY_RGB8 UINT32_C(1)
#define FY_BGR8 UINT32_C(2)
#define FY_RGBA8 UINT32_C(3)
#define FY_DEVICE_CPU UINT32_C(1)
#define FY_DEVICE_GPU UINT32_C(2)
#define FY_PRECISION_F32 UINT32_C(1)
#define FY_FALLBACK_NONE UINT32_C(0)
#define FY_RESULT_TRUNCATED UINT32_C(1)
#define FY_CAP_COMPILED UINT32_C(1)
#define FY_CAP_AVAILABLE UINT32_C(2)
#define FY_CAP_TEST_ONLY UINT32_C(4)

/* Natural target C layout, never packed. All three prefix fields are inputs.
 * All eleven entry points exist; only explicit test builds have executable
 * fixture models. No trained detector or supported model format is installed. */
typedef struct fy_abi_info {
    uint32_t struct_size, struct_version, required_flags;
    uint32_t abi_major, abi_minor;
    uint8_t build_id[64];
} fy_abi_info;
typedef struct fy_diagnostic {
    uint32_t struct_size, struct_version, required_flags;
    char *data;
    size_t capacity, required_bytes;
} fy_diagnostic;
typedef struct fy_model_options {
    uint32_t struct_size, struct_version, required_flags;
    size_t max_artifact_bytes, max_parser_steps;
} fy_model_options;
typedef struct fy_session_options {
    uint32_t struct_size, struct_version, required_flags;
    uint32_t backend_id, device_kind, device_index, precision;
    uint32_t fallback_policy, cpu_threads;
    size_t max_image_bytes, max_pixels, max_detections;
} fy_session_options;
typedef struct fy_image {
    uint32_t struct_size, struct_version, required_flags;
    const uint8_t *data;
    size_t byte_length;
    uint32_t width, height;
    size_t row_stride;
    uint32_t pixel_format;
} fy_image;
typedef struct fy_detect_options {
    uint32_t struct_size, struct_version, required_flags;
    float score_threshold;
    size_t max_detections;
} fy_detect_options;
typedef struct fy_session_info {
    uint32_t struct_size, struct_version, required_flags;
    uint32_t backend_id, device_kind, device_index, precision, execution_flags;
} fy_session_info;
typedef struct fy_result_info {
    uint32_t struct_size, struct_version, required_flags;
    uint32_t source_width, source_height;
    size_t count;
    uint32_t result_flags, backend_id, device_kind, device_index;
    uint32_t precision, execution_flags;
    uint8_t model_id[32];
} fy_result_info;
/* Fixed-stride array elements: extending these requires a new operation/type. */
typedef struct fy_detection {
    float x_min, y_min, x_max, y_max, score;
    uint32_t class_id;
} fy_detection;
typedef struct fy_capability {
    uint32_t backend_id, device_kind, device_index, flags, precision_mask;
} fy_capability;

/* Opaque owning handles. Never allocate, dereference or free these yourself. */
typedef struct fy_model fy_model;
typedef struct fy_session fy_session;
typedef struct fy_result fy_result;

/* Experimental ABI 1.0; not a stable release.
 * Set struct_size=sizeof(record), struct_version=1, required_flags=0.
 * For info outputs, only the header must be initialized. For diag, also
 * initialize data and capacity; required_bytes is output-only. diag may be NULL.
 * Accepted input options must have all fields initialized, not just the header.
 * Callers provide actually allocated, live, aligned, mutually disjoint and
 * externally synchronized storage. Non-null accepted descriptors advertise
 * real accessible storage. A short descriptor still provides at least two
 * initialized u32 words. Dangling/forged pointers are outside this contract.
 * Unknown versions/required flags fail without changing operation outputs.
 * Larger optional tails and headers are left intact. Malformed diag is checked
 * first and leaves operation outputs and diagnostic storage unchanged.
 * Diagnostic capacity includes a terminating NUL; required_bytes excludes it.
 * Zero capacity is permitted, with NULL data. Positive capacity needs writable
 * data. Diagnostic truncation preserves UTF-8 and the operation's status.
 * There is no global last-error state and no borrowed-buffer retention.
 * Recoverable unwinding is contained, but aborts/invalid foreign memory are not.
 */
FY_API int32_t FY_CALL fy_abi_query(fy_abi_info *out, fy_diagnostic *diag);

/* NULL dst + zero capacity queries count successfully. A non-NULL insufficient
 * destination returns BUFFER_TOO_SMALL, writes required count only, and leaves
 * all records untouched. Other invalid arguments preserve count and records.
 * The normal M1 build reports zero capabilities. Test builds are explicit. */
FY_API int32_t FY_CALL fy_capabilities_copy(
    fy_capability *dst, size_t capacity, size_t *required, fy_diagnostic *diag);

/* Owning output slots must be initialized to NULL. Nonempty slots are rejected
 * without replacing their value. Empty slots remain NULL on creation failure.
 * Model loading validates ABI/limits/buffers, then returns UNSUPPORTED_MODEL:
 * no artifact parser or real backend is installed yet, even in a test build.
 * Positive length and positive limits are mandatory; zero is not unlimited. */
FY_API int32_t FY_CALL fy_model_open(
    uint32_t required_abi_major, const uint8_t *bytes, size_t length,
    const fy_model_options *options, fy_model **out, fy_diagnostic *diag);

/* All non-NULL handles must originate from the same loaded library instance.
 * Release takes a writable initialized slot: NULL slot is INVALID_ARGUMENT;
 * a valid slot containing NULL is a no-op. Success clears the owning slot.
 * Another copy of a released pointer is stale, not an idempotent release token.
 * No use/release race is allowed. Release is not cancellation. Destroy all
 * handles before unloading their library. Only this library frees its handles.
 * Sessions retain model state independently of the public model handle. */
FY_API int32_t FY_CALL fy_model_release(fy_model **slot, fy_diagnostic *diag);
FY_API int32_t FY_CALL fy_session_create(
    uint32_t required_abi_major, const fy_model *model,
    const fy_session_options *options, fy_session **out, fy_diagnostic *diag);
/* Immutable description may run concurrently while the handle stays live.
 * Options are copied at creation. No silent device/precision fallback occurs. */
FY_API int32_t FY_CALL fy_session_describe(
    const fy_session *session, fy_session_info *out, fy_diagnostic *diag);
FY_API int32_t FY_CALL fy_session_release(fy_session **slot, fy_diagnostic *diag);

/* Synchronous host-image borrow: data and inter-row padding are initialized and
 * immutable until return, with no later access. Only RGB8/BGR8/straight RGBA8
 * top-down oriented sRGB pixels are accepted; alpha is ignored. No file decoding.
 * One session admits one execution through finalization; overlap returns BUSY.
 * Threshold is finite [0,1], cap positive and within the session limit.
 * Completed results own their data independently of pixels, models and sessions.
 * Boxes use clipped original-image pixel edges, exclusive maxima. Invalid model
 * output fails rather than being silently filtered. Sort: score descending,
 * class then x_min/y_min/x_max/y_max ascending, source index as final tie-break.
 * Cap applies after model-specific suppression and sets RESULT_TRUNCATED when
 * it removes valid boxes. M1 fixtures contain post-suppression candidates only. */
FY_API int32_t FY_CALL fy_detect(
    fy_session *session, const fy_image *image, const fy_detect_options *options,
    fy_result **out, fy_diagnostic *diag);
FY_API int32_t FY_CALL fy_result_describe(
    const fy_result *result, fy_result_info *out, fy_diagnostic *diag);
/* Copy existing immutable results, never rerun. NULL+zero is a count query.
 * required is mandatory and output-only. Positive capacity requires a valid
 * destination. BUFFER_TOO_SMALL updates required only, never partial records.
 * Exactly count records are written on success; extra destination space stays.
 * Immutable describe/copy calls may overlap while the result remains live. */
FY_API int32_t FY_CALL fy_result_copy(
    const fy_result *result, fy_detection *dst, size_t capacity,
    size_t *required, fy_diagnostic *diag);
FY_API int32_t FY_CALL fy_result_release(fy_result **slot, fy_diagnostic *diag);
#ifdef __cplusplus
}
#endif
#endif
