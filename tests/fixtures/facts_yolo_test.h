/* SPDX-License-Identifier: Apache-2.0 */
#ifndef FACTS_YOLO_TEST_H
#define FACTS_YOLO_TEST_H
#include "facts_yolo.h"

/* Test builds only. These symbols are absent from the normal native library.
 * IDs create model metadata for contract tests, not trained networks or model
 * files. Execution/fault behavior is not implemented by the task-3 factory. */
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
#ifdef __cplusplus
}
#endif
#endif
