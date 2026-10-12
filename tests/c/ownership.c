/* SPDX-License-Identifier: Apache-2.0 */
#include "facts_yolo.h"
#ifdef FY_TEST_FIXTURES
#include "facts_yolo_test.h"
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void capabilities(void) {
    size_t count = SIZE_MAX;
    fy_capability record = {77, 77, 77, 77, 77};
    assert(fy_capabilities_copy(NULL, 0, &count, NULL) == FY_OK);
#ifdef FY_TEST_FIXTURES
    assert(count == 1);
    assert(fy_capabilities_copy(&record, 0, &count, NULL) == FY_BUFFER_TOO_SMALL);
    assert(count == 1 && record.backend_id == 77);
    assert(fy_capabilities_copy(&record, 1, &count, NULL) == FY_OK);
    assert(record.backend_id == FY_BACKEND_TEST_FIXTURE);
    assert(record.device_kind == FY_DEVICE_CPU && record.device_index == 0);
    assert(record.flags == (FY_CAP_COMPILED | FY_CAP_AVAILABLE | FY_CAP_TEST_ONLY));
    assert(record.precision_mask == 1);
#else
    assert(count == 0);
    assert(fy_capabilities_copy(&record, 1, &count, NULL) == FY_OK);
    assert(record.backend_id == 77);
#endif
    count = 77;
    assert(fy_capabilities_copy(NULL, 1, &count, NULL) == FY_INVALID_ARGUMENT);
    assert(count == 77);
}

static void unsupported_loading(void) {
    const uint8_t bytes[] = "FY_TEST_BASIC";
    fy_model_options options = {sizeof(options), 1, 0, 64, 16};
    fy_model *model = NULL;
    fy_session *session = NULL;
    unsigned char text[64];
    memset(text, 0xA5, sizeof(text));
    fy_diagnostic diag = {sizeof(diag), 1, 0, (char *)text, 8, SIZE_MAX};
    assert(fy_model_open(1, bytes, sizeof(bytes), &options, &model, &diag)
        == FY_UNSUPPORTED_MODEL);
    assert(model == NULL && diag.required_bytes > 7 && text[7] == 0);
    for (size_t i = 8; i < sizeof(text); ++i) assert(text[i] == 0xA5);
    assert(fy_model_open(2, bytes, sizeof(bytes), &options, &model, NULL)
        == FY_UNSUPPORTED_ABI);
    _Alignas(max_align_t) uint32_t prefix[2] = {8, 1};
    assert(fy_model_open(1, bytes, sizeof(bytes),
        (const fy_model_options *)(const void *)prefix, &model, NULL)
        == FY_INVALID_ARGUMENT);
    assert(model == NULL);
    assert(fy_model_release(&model, NULL) == FY_OK);
    assert(fy_session_release(&session, NULL) == FY_OK);
    assert(fy_model_release(NULL, NULL) == FY_INVALID_ARGUMENT);
    assert(fy_session_release(NULL, NULL) == FY_INVALID_ARGUMENT);
}

#ifdef FY_TEST_FIXTURES
static void owned_sessions(void) {
    fy_model *model = NULL;
    fy_session *one = NULL;
    fy_session *two = NULL;
    fy_session_options options = {
        sizeof(options), 1, 0, FY_BACKEND_TEST_FIXTURE, FY_DEVICE_CPU, 0,
        FY_PRECISION_F32, FY_FALLBACK_NONE, 1, 1024, 64, 16
    };
    assert(fy_test_model_create(1, FY_TEST_BASIC, &model, NULL) == FY_OK);
    assert(model != NULL);
    options.device_kind = FY_DEVICE_GPU;
    assert(fy_session_create(1, model, &options, &one, NULL) == FY_DEVICE_UNAVAILABLE);
    assert(one == NULL);
    options.device_kind = FY_DEVICE_CPU;
    assert(fy_session_create(1, model, &options, &one, NULL) == FY_OK);
    assert(fy_session_create(1, model, &options, &two, NULL) == FY_OK);
    fy_session *saved = one;
    assert(fy_session_create(1, model, &options, &one, NULL) == FY_INVALID_ARGUMENT);
    assert(one == saved);
    fy_diagnostic bad_diag = {sizeof(bad_diag), 9, 0, NULL, 0, SIZE_MAX};
    fy_model *saved_model = model;
    assert(fy_model_release(&model, &bad_diag) == FY_UNSUPPORTED_ABI);
    assert(model == saved_model && bad_diag.required_bytes == SIZE_MAX);
    assert(fy_model_release(&model, NULL) == FY_OK);
    assert(model == NULL);
    options.device_index = 99;
    fy_session_info info; /* Only input header initialized, payload is output. */
    info.struct_size = sizeof(info);
    info.struct_version = 1;
    info.required_flags = 0;
    assert(fy_session_describe(one, &info, NULL) == FY_OK);
    assert(info.backend_id == FY_BACKEND_TEST_FIXTURE);
    assert(info.device_kind == FY_DEVICE_CPU && info.device_index == 0);
    assert(info.precision == FY_PRECISION_F32 && info.execution_flags == 0);
    assert(fy_session_release(&one, NULL) == FY_OK && one == NULL);
    assert(fy_session_describe(two, &info, NULL) == FY_OK);
    assert(fy_session_release(&two, NULL) == FY_OK && two == NULL);
    assert(fy_session_release(&two, NULL) == FY_OK);
}
#endif

int main(void) {
    capabilities();
    unsupported_loading();
#ifdef FY_TEST_FIXTURES
    for (size_t i = 0; i < 1000; ++i) owned_sessions();
    puts("C fixture ownership: 1000 model/session lifecycle cycles passed");
#else
    puts("C normal build: zero backends, rejected artifacts, safe output slots passed");
#endif
    return 0;
}
