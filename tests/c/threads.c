/* SPDX-License-Identifier: Apache-2.0 */
#include "facts_yolo_test.h"
#include <assert.h>
#include <stdio.h>
#ifdef NDEBUG
#error Native conformance requires active assertions
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif
#define HEADER(T) (uint32_t)sizeof(T), FY_STRUCT_VERSION, 0

struct call { fy_session *session; int32_t status; size_t count; };
static void execute(struct call *call) {
    unsigned char pixels[12] = {0};
    fy_image image = { HEADER(fy_image), pixels, sizeof(pixels), 2, 2, 6, FY_RGB8 };
    fy_detect_options options = { HEADER(fy_detect_options), 0.5f, 16 };
    fy_result *result = NULL;
    call->status = fy_detect(call->session, &image, &options, &result, NULL);
    if (call->status == FY_OK) {
        int32_t copy = fy_result_copy(result, NULL, 0, &call->count, NULL);
        int32_t release = fy_result_release(&result, NULL);
        if (copy != FY_OK || release != FY_OK) call->status = FY_INTERNAL_ERROR;
    } else if (result != NULL) {
        call->status = FY_INTERNAL_ERROR;
    }
}
#ifdef _WIN32
static DWORD WINAPI worker(LPVOID argument) { execute((struct call *)argument); return 0; }
#else
static void *worker(void *argument) { execute((struct call *)argument); return NULL; }
#endif

int main(void) {
    fy_model *model = NULL;
    fy_session *session = NULL;
    fy_session_options options = {
        HEADER(fy_session_options), FY_BACKEND_TEST_FIXTURE,
        FY_DEVICE_CPU, 0, FY_PRECISION_F32, FY_FALLBACK_NONE, 1, 1024, 64, 16
    };
    assert(fy_test_model_create(1, FY_TEST_BLOCKING, &model, NULL) == FY_OK);
    assert(fy_session_create(1, model, &options, &session, NULL) == FY_OK);
    assert(fy_model_release(&model, NULL) == FY_OK);
    for (int i = 0; i < 50; ++i) {
        struct call first = { session, -1, 0 }, second = { session, -1, 0 };
#ifdef _WIN32
        HANDLE thread = CreateThread(NULL, 0, worker, &first, 0, NULL);
        assert(thread != NULL);
#else
        pthread_t thread;
        assert(pthread_create(&thread, NULL, worker, &first) == 0);
#endif
        int32_t waiting = fy_test_session_wait_running(session, 3000);
        if (waiting == FY_OK) execute(&second);
        fy_session_info info = { HEADER(fy_session_info), 0, 0, 0, 0, 0 };
        int32_t described = fy_session_describe(session, &info, NULL);
        /* Cleanup before assertions: a failed test must not strand its worker. */
        int32_t resumed = fy_test_session_resume(session);
#ifdef _WIN32
        assert(WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0);
        assert(CloseHandle(thread) != 0);
#else
        assert(pthread_join(thread, NULL) == 0);
#endif
        assert(waiting == FY_OK && resumed == FY_OK && described == FY_OK);
        assert(info.device_kind == FY_DEVICE_CPU);
        assert(second.status == FY_BUSY && second.count == 0);
        assert(first.status == FY_OK && first.count == 3);
    }
    assert(fy_session_release(&session, NULL) == FY_OK && session == NULL);
    puts("C threads: 50 barrier-confirmed BUSY cycles passed");
    return 0;
}
