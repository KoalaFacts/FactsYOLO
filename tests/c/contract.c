/* SPDX-License-Identifier: Apache-2.0 */
#include <facts_yolo.h>
#ifdef FY_TEST_FIXTURES
#include "facts_yolo_test.h"
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>

#define HEADER(T) (uint32_t)sizeof(T), FY_STRUCT_VERSION, 0

static void common_contract(void) {
    struct extended_info { fy_abi_info info; unsigned char tail[16]; } extended;
    unsigned char text[8];
    fy_diagnostic diag = { HEADER(fy_diagnostic), (char *)text, sizeof(text), 99 };
    fy_model_options options = { HEADER(fy_model_options), 128, 1000 };
    unsigned char bytes[4] = {1, 2, 3, 4};
    fy_model *model = NULL;
    fy_session *session = NULL;
    fy_result *result = NULL;
    size_t count = 99;
    _Alignas(fy_model_options) uint32_t prefix[2] = {8, FY_STRUCT_VERSION};
    memset(&extended, 0xA5, sizeof(extended));
    memset(text, 0xA5, sizeof(text));
    extended.info.struct_size = (uint32_t)sizeof(extended);
    extended.info.struct_version = FY_STRUCT_VERSION;
    extended.info.required_flags = 0;
    assert(fy_abi_query(&extended.info, &diag) == FY_OK);
    assert(extended.info.abi_major == 1 && extended.info.abi_minor == 0);
    assert(extended.info.struct_size == sizeof(extended));
    assert(diag.required_bytes == 0 && text[0] == 0);
    for (size_t i = 0; i < sizeof(extended.tail); ++i) assert(extended.tail[i] == 0xA5);
    assert(fy_model_open(2, bytes, sizeof(bytes), &options, &model, NULL) == FY_UNSUPPORTED_ABI);
    assert(fy_model_open(1, bytes, sizeof(bytes), (const fy_model_options *)prefix,
                         &model, NULL) == FY_INVALID_ARGUMENT);
    options.max_artifact_bytes = 3;
    assert(fy_model_open(1, bytes, sizeof(bytes), &options, &model, NULL) == FY_LIMIT_EXCEEDED);
    options.max_artifact_bytes = 128;
    assert(fy_model_open(1, bytes, sizeof(bytes), &options, &model, &diag) == FY_UNSUPPORTED_MODEL);
    assert(model == NULL && diag.required_bytes > sizeof(text) && text[7] == 0);
    diag.struct_version = 99;
    assert(fy_capabilities_copy(NULL, 0, &count, &diag) == FY_UNSUPPORTED_ABI);
    assert(count == 99);
    assert(fy_capabilities_copy(NULL, 0, &count, NULL) == FY_OK);
#ifdef FY_TEST_FIXTURES
    assert(count == 1);
    fy_capability capability;
    assert(fy_capabilities_copy(&capability, 1, &count, NULL) == FY_OK);
    assert(capability.backend_id == FY_BACKEND_TEST_FIXTURE);
    assert((capability.flags & FY_CAP_TEST_ONLY) != 0);
#else
    assert(count == 0);
#endif
    assert(fy_session_create(1, NULL, NULL, &session, NULL) == FY_INVALID_ARGUMENT);
    assert(fy_session_describe(NULL, NULL, NULL) == FY_INVALID_ARGUMENT);
    assert(fy_detect(NULL, NULL, NULL, &result, NULL) == FY_INVALID_ARGUMENT);
    assert(fy_result_describe(NULL, NULL, NULL) == FY_INVALID_ARGUMENT);
    assert(fy_result_copy(NULL, NULL, 0, &count, NULL) == FY_INVALID_ARGUMENT);
    assert(fy_model_release(&model, NULL) == FY_OK);
    assert(fy_session_release(&session, NULL) == FY_OK);
    assert(fy_result_release(&result, NULL) == FY_OK);
    assert(fy_result_release(NULL, NULL) == FY_INVALID_ARGUMENT);
}

#ifdef FY_TEST_FIXTURES
static fy_session *session_for(uint32_t fixture) {
    fy_model *model = NULL;
    fy_session *session = NULL;
    fy_session_options options = {
        HEADER(fy_session_options), FY_BACKEND_TEST_FIXTURE,
        FY_DEVICE_CPU, 0, FY_PRECISION_F32, FY_FALLBACK_NONE, 1, 1024, 64, 16
    };
    assert(fy_test_model_create(1, fixture, &model, NULL) == FY_OK);
    assert(fy_session_create(1, model, &options, &session, NULL) == FY_OK);
    assert(fy_model_release(&model, NULL) == FY_OK && model == NULL);
    return session;
}

static void result_contract(void) {
    fy_result *result = NULL;
    fy_session *session = session_for(FY_TEST_BASIC);
    fy_session_info actual = { HEADER(fy_session_info), 0, 0, 0, 0, 0 };
    assert(fy_session_describe(session, &actual, NULL) == FY_OK);
    assert(actual.device_kind == FY_DEVICE_CPU && actual.precision == FY_PRECISION_F32);
    {
        unsigned char pixels[12] = {0};
        fy_image image = { HEADER(fy_image), pixels, sizeof(pixels), 2, 2, 6, FY_RGB8 };
        fy_detect_options options = { HEADER(fy_detect_options), 0.5f, 16 };
        assert(fy_detect(session, &image, &options, &result, NULL) == FY_OK);
        fy_result *saved = result;
        assert(fy_detect(session, &image, &options, &result, NULL) == FY_INVALID_ARGUMENT);
        assert(result == saved);
        memset(pixels, 0xA5, sizeof(pixels));
    }
    assert(fy_session_release(&session, NULL) == FY_OK && session == NULL);
    struct extended_result { fy_result_info info; unsigned char tail[16]; } output;
    memset(&output, 0xA5, sizeof(output));
    output.info.struct_size = (uint32_t)sizeof(output);
    output.info.struct_version = FY_STRUCT_VERSION;
    output.info.required_flags = 0;
    assert(fy_result_describe(result, &output.info, NULL) == FY_OK);
    assert(output.info.count == 3 && output.info.model_id[0] == FY_TEST_BASIC);
    assert(output.info.source_width == 2 && output.info.source_height == 2);
    for (size_t i = 0; i < sizeof(output.tail); ++i) assert(output.tail[i] == 0xA5);
    fy_detection records[4], untouched[4];
    memset(records, 0xA5, sizeof(records));
    memcpy(untouched, records, sizeof(records));
    size_t count = 99;
    assert(fy_result_copy(result, NULL, 0, &count, NULL) == FY_OK && count == 3);
    assert(fy_result_copy(result, records, 2, &count, NULL) == FY_BUFFER_TOO_SMALL);
    assert(count == 3 && memcmp(records, untouched, sizeof(records)) == 0);
    assert(fy_result_copy(result, records, 4, &count, NULL) == FY_OK && count == 3);
    assert(memcmp(&records[3], &untouched[3], sizeof(records[3])) == 0);
    assert(records[0].class_id == 0 && records[0].score == 0.9f);
    assert(records[0].x_min == 0 && records[0].y_min == 0);
    assert(records[0].x_max == 2 && records[0].y_max == 2);
    assert(records[1].class_id == 1 && records[1].x_max == 1);
    assert(records[2].score == 0.5f && records[2].x_min == 1);
    assert(fy_result_release(&result, NULL) == FY_OK && result == NULL);
    assert(fy_result_release(&result, NULL) == FY_OK);
}

static void failure_contract(uint32_t fixture, int32_t expected) {
    fy_session *session = session_for(fixture);
    unsigned char pixels[12] = {0};
    fy_image image = { HEADER(fy_image), pixels, sizeof(pixels), 2, 2, 6, FY_RGB8 };
    fy_detect_options options = { HEADER(fy_detect_options), 0.5f, 16 };
    fy_result *result = NULL;
    assert(fy_detect(session, &image, &options, &result, NULL) == expected);
    assert(result == NULL);
    assert(fy_detect(session, &image, &options, &result, NULL) == FY_POISONED);
    assert(result == NULL && fy_session_release(&session, NULL) == FY_OK);
}
#endif

int main(void) {
    common_contract();
#ifdef FY_TEST_FIXTURES
    result_contract();
    failure_contract(FY_TEST_INVALID_OUTPUT, FY_MODEL_OUTPUT_INVALID);
    failure_contract(FY_TEST_PANIC, FY_INTERNAL_ERROR);
    failure_contract(FY_TEST_PAYLOAD_DROP_PANIC, FY_INTERNAL_ERROR);
    puts("C contract: owned results, bounded copies and poisoned execution passed");
#else
    puts("C contract: normal build has no executable model or fixture backend");
#endif
    return 0;
}
