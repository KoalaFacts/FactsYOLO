// SPDX-License-Identifier: Apache-2.0
//! Natural C layout; never pack these records or append to array elements.
use std::ffi::c_char;

pub const FY_OK: i32 = 0;
pub const FY_INVALID_ARGUMENT: i32 = 1;
pub const FY_UNSUPPORTED_ABI: i32 = 2;
pub const FY_UNSUPPORTED_FEATURE: i32 = 3;
pub const FY_UNSUPPORTED_MODEL: i32 = 4;
pub const FY_DEVICE_UNAVAILABLE: i32 = 5;
pub const FY_BUSY: i32 = 6;
pub const FY_LIMIT_EXCEEDED: i32 = 7;
pub const FY_BUFFER_TOO_SMALL: i32 = 8;
pub const FY_MODEL_OUTPUT_INVALID: i32 = 9;
pub const FY_BACKEND_ERROR: i32 = 10;
pub const FY_POISONED: i32 = 11;
pub const FY_INTERNAL_ERROR: i32 = 12;
pub const FY_ABI_MAJOR: u32 = 1;
pub const FY_ABI_MINOR: u32 = 0;
pub const FY_STRUCT_VERSION: u32 = 1;
pub const FY_RGB8: u32 = 1;
pub const FY_BGR8: u32 = 2;
pub const FY_RGBA8: u32 = 3;
pub const FY_BACKEND_TD1_CPU: u32 = 1;
pub const FY_DEVICE_CPU: u32 = 1;
pub const FY_DEVICE_GPU: u32 = 2;
pub const FY_PRECISION_F32: u32 = 1;
pub const FY_FALLBACK_NONE: u32 = 0;
pub const FY_RESULT_TRUNCATED: u32 = 1;
pub const FY_CAP_COMPILED: u32 = 1;
pub const FY_CAP_AVAILABLE: u32 = 2;
pub const FY_CAP_TEST_ONLY: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FyAbiInfo {
    pub struct_size: u32,
    pub struct_version: u32,
    pub required_flags: u32,
    pub abi_major: u32,
    pub abi_minor: u32,
    pub build_id: [u8; 64],
}

impl Default for FyAbiInfo {
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            struct_version: FY_STRUCT_VERSION,
            required_flags: 0,
            abi_major: 0,
            abi_minor: 0,
            build_id: [0; 64],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FyDiagnostic {
    pub struct_size: u32,
    pub struct_version: u32,
    pub required_flags: u32,
    pub data: *mut c_char,
    pub capacity: usize,
    pub required_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FyModelOptions {
    pub struct_size: u32,
    pub struct_version: u32,
    pub required_flags: u32,
    pub max_artifact_bytes: usize,
    pub max_parser_steps: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FySessionOptions {
    pub struct_size: u32,
    pub struct_version: u32,
    pub required_flags: u32,
    pub backend_id: u32,
    pub device_kind: u32,
    pub device_index: u32,
    pub precision: u32,
    pub fallback_policy: u32,
    pub cpu_threads: u32,
    pub max_image_bytes: usize,
    pub max_pixels: usize,
    pub max_detections: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FyImage {
    pub struct_size: u32,
    pub struct_version: u32,
    pub required_flags: u32,
    pub data: *const u8,
    pub byte_length: usize,
    pub width: u32,
    pub height: u32,
    pub row_stride: usize,
    pub pixel_format: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FyDetectOptions {
    pub struct_size: u32,
    pub struct_version: u32,
    pub required_flags: u32,
    pub score_threshold: f32,
    pub max_detections: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FySessionInfo {
    pub struct_size: u32,
    pub struct_version: u32,
    pub required_flags: u32,
    pub backend_id: u32,
    pub device_kind: u32,
    pub device_index: u32,
    pub precision: u32,
    pub execution_flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FyResultInfo {
    pub struct_size: u32,
    pub struct_version: u32,
    pub required_flags: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub count: usize,
    pub result_flags: u32,
    pub backend_id: u32,
    pub device_kind: u32,
    pub device_index: u32,
    pub precision: u32,
    pub execution_flags: u32,
    pub model_id: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FyDetection {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub score: f32,
    pub class_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FyCapability {
    pub backend_id: u32,
    pub device_kind: u32,
    pub device_index: u32,
    pub flags: u32,
    pub precision_mask: u32,
}
