/* SPDX-License-Identifier: Apache-2.0 */
#include <facts_yolo.h>
#include <stdio.h>

#define RECORD(T) printf("T %s %zu %zu\n", #T, sizeof(T), (size_t)_Alignof(T))
#define FIELD(T, F) printf("F %s %s %zu\n", #T, #F, offsetof(T, F))
#define HEADER(T) FIELD(T, struct_size); FIELD(T, struct_version); FIELD(T, required_flags)
#define CONSTANT(K) printf("K %s %llu\n", #K, (unsigned long long)(K))

int main(void) {
    printf("P pointer %zu\n", sizeof(size_t));
    RECORD(fy_abi_info); HEADER(fy_abi_info);
    FIELD(fy_abi_info, abi_major); FIELD(fy_abi_info, abi_minor); FIELD(fy_abi_info, build_id);
    RECORD(fy_diagnostic); HEADER(fy_diagnostic);
    FIELD(fy_diagnostic, data); FIELD(fy_diagnostic, capacity); FIELD(fy_diagnostic, required_bytes);
    RECORD(fy_model_options); HEADER(fy_model_options);
    FIELD(fy_model_options, max_artifact_bytes); FIELD(fy_model_options, max_parser_steps);
    RECORD(fy_session_options); HEADER(fy_session_options);
    FIELD(fy_session_options, backend_id); FIELD(fy_session_options, device_kind);
    FIELD(fy_session_options, device_index); FIELD(fy_session_options, precision);
    FIELD(fy_session_options, fallback_policy); FIELD(fy_session_options, cpu_threads);
    FIELD(fy_session_options, max_image_bytes); FIELD(fy_session_options, max_pixels);
    FIELD(fy_session_options, max_detections);
    RECORD(fy_image); HEADER(fy_image);
    FIELD(fy_image, data); FIELD(fy_image, byte_length); FIELD(fy_image, width);
    FIELD(fy_image, height); FIELD(fy_image, row_stride); FIELD(fy_image, pixel_format);
    RECORD(fy_detect_options); HEADER(fy_detect_options);
    FIELD(fy_detect_options, score_threshold); FIELD(fy_detect_options, max_detections);
    RECORD(fy_session_info); HEADER(fy_session_info);
    FIELD(fy_session_info, backend_id); FIELD(fy_session_info, device_kind);
    FIELD(fy_session_info, device_index); FIELD(fy_session_info, precision);
    FIELD(fy_session_info, execution_flags);
    RECORD(fy_result_info); HEADER(fy_result_info);
    FIELD(fy_result_info, source_width); FIELD(fy_result_info, source_height);
    FIELD(fy_result_info, count); FIELD(fy_result_info, result_flags);
    FIELD(fy_result_info, backend_id); FIELD(fy_result_info, device_kind);
    FIELD(fy_result_info, device_index); FIELD(fy_result_info, precision);
    FIELD(fy_result_info, execution_flags); FIELD(fy_result_info, model_id);
    RECORD(fy_detection);
    FIELD(fy_detection, x_min); FIELD(fy_detection, y_min); FIELD(fy_detection, x_max);
    FIELD(fy_detection, y_max); FIELD(fy_detection, score); FIELD(fy_detection, class_id);
    RECORD(fy_capability);
    FIELD(fy_capability, backend_id); FIELD(fy_capability, device_kind);
    FIELD(fy_capability, device_index); FIELD(fy_capability, flags);
    FIELD(fy_capability, precision_mask);
    CONSTANT(FY_OK); CONSTANT(FY_INVALID_ARGUMENT); CONSTANT(FY_UNSUPPORTED_ABI);
    CONSTANT(FY_UNSUPPORTED_FEATURE); CONSTANT(FY_UNSUPPORTED_MODEL);
    CONSTANT(FY_DEVICE_UNAVAILABLE); CONSTANT(FY_BUSY); CONSTANT(FY_LIMIT_EXCEEDED);
    CONSTANT(FY_BUFFER_TOO_SMALL); CONSTANT(FY_MODEL_OUTPUT_INVALID);
    CONSTANT(FY_BACKEND_ERROR); CONSTANT(FY_POISONED); CONSTANT(FY_INTERNAL_ERROR);
    CONSTANT(FY_ABI_MAJOR); CONSTANT(FY_ABI_MINOR); CONSTANT(FY_STRUCT_VERSION);
    CONSTANT(FY_RGB8); CONSTANT(FY_BGR8); CONSTANT(FY_RGBA8);
    CONSTANT(FY_DEVICE_CPU); CONSTANT(FY_DEVICE_GPU); CONSTANT(FY_PRECISION_F32);
    CONSTANT(FY_FALLBACK_NONE); CONSTANT(FY_RESULT_TRUNCATED);
    CONSTANT(FY_CAP_COMPILED); CONSTANT(FY_CAP_AVAILABLE); CONSTANT(FY_CAP_TEST_ONLY);
    return 0;
}
