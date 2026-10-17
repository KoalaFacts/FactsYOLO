/* SPDX-License-Identifier: Apache-2.0 */
/* Loader rejection fixtures only. Never packaged, never used for inference. */
#define FY_STATIC
#include "facts_yolo.h"
#include <string.h>
#ifdef _WIN32
/* Keep FY_STATIC declarations and definitions at the same C linkage. Request
 * only implemented exports from the MSVC linker, including the missing-export
 * negative case, without adding conflicting dllexport redeclarations. */
#pragma comment(linker, "/export:fy_abi_query")
#pragma comment(linker, "/export:fy_capabilities_copy")
#pragma comment(linker, "/export:fy_model_open")
#pragma comment(linker, "/export:fy_model_release")
#pragma comment(linker, "/export:fy_session_create")
#pragma comment(linker, "/export:fy_session_describe")
#pragma comment(linker, "/export:fy_session_release")
#pragma comment(linker, "/export:fy_detect")
#pragma comment(linker, "/export:fy_result_describe")
#pragma comment(linker, "/export:fy_result_copy")
#if !defined(MISSING_EXPORT)
#pragma comment(linker, "/export:fy_result_release")
#endif
#endif
#if defined(WITH_DEPENDENCY)
extern int fy_dependency_probe(void);
#endif
int32_t FY_CALL fy_abi_query(fy_abi_info *out, fy_diagnostic *diag) {
    int32_t status = FY_OK;
#if defined(WITH_DEPENDENCY)
    out->abi_major = (uint32_t)fy_dependency_probe();
#elif defined(DIAGNOSTIC_BOUNDARY)
    size_t count = diag->capacity > 0 ? diag->capacity - 1 : 0;
    if (count > 1024) count = 1024;
    memset(diag->data, 'x', count);
    if (diag->capacity) diag->data[count] = 0;
    diag->required_bytes = 1024;
    (void)out;
    status = FY_UNSUPPORTED_MODEL;
#else
    out->abi_major = 2;
#endif
    (void)diag;
    return status;
}
int32_t FY_CALL fy_capabilities_copy(fy_capability *a,size_t b,size_t *c,fy_diagnostic *d) { (void)a;(void)b;(void)c;(void)d;return 12; }
int32_t FY_CALL fy_model_open(uint32_t a,const uint8_t *b,size_t c,const fy_model_options *d,fy_model **e,fy_diagnostic *f) { (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 12; }
int32_t FY_CALL fy_model_release(fy_model **a,fy_diagnostic *b) { (void)a;(void)b;return 12; }
int32_t FY_CALL fy_session_create(uint32_t a,const fy_model *b,const fy_session_options *c,fy_session **d,fy_diagnostic *e) { (void)a;(void)b;(void)c;(void)d;(void)e;return 12; }
int32_t FY_CALL fy_session_describe(const fy_session *a,fy_session_info *b,fy_diagnostic *c) { (void)a;(void)b;(void)c;return 12; }
int32_t FY_CALL fy_session_release(fy_session **a,fy_diagnostic *b) { (void)a;(void)b;return 12; }
int32_t FY_CALL fy_detect(fy_session *a,const fy_image *b,const fy_detect_options *c,fy_result **d,fy_diagnostic *e) { (void)a;(void)b;(void)c;(void)d;(void)e;return 12; }
int32_t FY_CALL fy_result_describe(const fy_result *a,fy_result_info *b,fy_diagnostic *c) { (void)a;(void)b;(void)c;return 12; }
int32_t FY_CALL fy_result_copy(const fy_result *a,fy_detection *b,size_t c,size_t *d,fy_diagnostic *e) { (void)a;(void)b;(void)c;(void)d;(void)e;return 12; }
#if !defined(MISSING_EXPORT)
int32_t FY_CALL fy_result_release(fy_result **a,fy_diagnostic *b) { (void)a;(void)b;return 12; }
#endif
