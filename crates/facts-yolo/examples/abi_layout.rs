// SPDX-License-Identifier: Apache-2.0
//! Development evidence: target compiler layouts, not a public library export.
use facts_yolo::ffi::types::*;
use std::mem::{align_of, offset_of, size_of};

macro_rules! record {
    ($rust:ident, $c:literal; $($field:ident),+ $(,)?) => {{
        // Exhaustive pattern: adding an unlisted Rust field fails compilation.
        fn fields_are_exhaustive(value: $rust) {
            let $rust { $($field: _,)+ } = value;
        }
        let _ = fields_are_exhaustive as fn($rust);
        println!("T {} {} {}", $c, size_of::<$rust>(), align_of::<$rust>());
        $(println!("F {} {} {}", $c, stringify!($field), offset_of!($rust, $field));)+
    }};
}
macro_rules! constants {
    ($($name:ident),+ $(,)?) => {
        $(println!("K {} {}", stringify!($name), $name);)+
    };
}

fn main() {
    println!("P pointer {}", size_of::<usize>());
    record!(FyAbiInfo, "fy_abi_info";
        struct_size, struct_version, required_flags, abi_major, abi_minor, build_id);
    record!(FyDiagnostic, "fy_diagnostic";
        struct_size, struct_version, required_flags, data, capacity, required_bytes);
    record!(FyModelOptions, "fy_model_options";
        struct_size, struct_version, required_flags, max_artifact_bytes, max_parser_steps);
    record!(FySessionOptions, "fy_session_options";
        struct_size, struct_version, required_flags, backend_id, device_kind, device_index,
        precision, fallback_policy, cpu_threads, max_image_bytes, max_pixels, max_detections);
    record!(FyImage, "fy_image";
        struct_size, struct_version, required_flags, data, byte_length,
        width, height, row_stride, pixel_format);
    record!(FyDetectOptions, "fy_detect_options";
        struct_size, struct_version, required_flags, score_threshold, max_detections);
    record!(FySessionInfo, "fy_session_info";
        struct_size, struct_version, required_flags, backend_id,
        device_kind, device_index, precision, execution_flags);
    record!(FyResultInfo, "fy_result_info";
        struct_size, struct_version, required_flags, source_width, source_height, count,
        result_flags, backend_id, device_kind, device_index, precision, execution_flags, model_id);
    record!(FyDetection, "fy_detection"; x_min, y_min, x_max, y_max, score, class_id);
    record!(FyCapability, "fy_capability";
        backend_id, device_kind, device_index, flags, precision_mask);
    constants!(
        FY_OK,
        FY_INVALID_ARGUMENT,
        FY_UNSUPPORTED_ABI,
        FY_UNSUPPORTED_FEATURE,
        FY_UNSUPPORTED_MODEL,
        FY_DEVICE_UNAVAILABLE,
        FY_BUSY,
        FY_LIMIT_EXCEEDED,
        FY_BUFFER_TOO_SMALL,
        FY_MODEL_OUTPUT_INVALID,
        FY_BACKEND_ERROR,
        FY_POISONED,
        FY_INTERNAL_ERROR,
        FY_ABI_MAJOR,
        FY_ABI_MINOR,
        FY_STRUCT_VERSION,
        FY_RGB8,
        FY_BGR8,
        FY_RGBA8,
        FY_DEVICE_CPU,
        FY_DEVICE_GPU,
        FY_PRECISION_F32,
        FY_FALLBACK_NONE,
        FY_RESULT_TRUNCATED,
        FY_CAP_COMPILED,
        FY_CAP_AVAILABLE,
        FY_CAP_TEST_ONLY,
    );
}
