/* SPDX-License-Identifier: Apache-2.0 */
#ifndef FACTS_YOLO_TEST_H
#define FACTS_YOLO_TEST_H
#include "facts_yolo.h"

/* Test builds only. These symbols are absent from the normal native library.
 * IDs select deterministic in-process fixtures, never trained networks or files. */
#define FY_BACKEND_TEST_FIXTURE UINT32_C(0xffff0001)
#define FY_TEST_BASIC UINT32_C(1)
#define FY_TEST_EMPTY UINT32_C(2)
#define FY_TEST_INVALID_OUTPUT UINT32_C(3)
#define FY_TEST_BLOCKING UINT32_C(4)
#define FY_TEST_PANIC UINT32_C(5)
#define FY_TEST_PAYLOAD_DROP_PANIC UINT32_C(6)
#ifdef __cplusplus
extern "C" {
#endif
FY_API int32_t FY_CALL fy_test_model_create(
    uint32_t required_abi_major, uint32_t fixture_id,
    fy_model **out, fy_diagnostic *diag);

/* Only for FY_TEST_BLOCKING sessions; timeout_ms must be positive.
 * Uses an independent latch, not the execution lock. A timeout is INTERNAL_ERROR.
 * resume grants at most one permit and may precede entry to enable safe cleanup.
 * Neither operation releases/cancels a session or provides production deadlines.
 * All non-null handles must remain live; no concurrent release is permitted. */
FY_API int32_t FY_CALL fy_test_session_wait_running(
    const fy_session *session, uint32_t timeout_ms);
FY_API int32_t FY_CALL fy_test_session_resume(const fy_session *session);

/* Test-only object counts, in order: public model handles, retained model states,
 * sessions, completed results. destination has four live writable size_t slots;
 * capacity must equal four. Invalid arguments leave the destination unchanged.
 * A coherent multi-count snapshot requires quiescent allocations or a test latch.
 * Not RSS, allocator-peak or GPU-memory evidence. Never a production API. */
FY_API int32_t FY_CALL fy_test_resource_counts(size_t *destination, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
