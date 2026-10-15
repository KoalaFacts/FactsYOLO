// SPDX-License-Identifier: Apache-2.0
using System.Runtime.InteropServices;
namespace FactsYOLO;

// Natural target layout. No packing or managed references cross the C ABI.
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct FyAbiInfo
{
    public uint struct_size, struct_version, required_flags, abi_major, abi_minor;
    public fixed byte build_id[64];
}
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct FyDiagnostic
{
    public uint struct_size, struct_version, required_flags;
    public byte* data;
    public nuint capacity, required_bytes;
}
[StructLayout(LayoutKind.Sequential)]
internal struct FyModelOptions
{
    public uint struct_size, struct_version, required_flags;
    public nuint max_artifact_bytes, max_parser_steps;
}
[StructLayout(LayoutKind.Sequential)]
internal struct FySessionOptions
{
    public uint struct_size, struct_version, required_flags;
    public uint backend_id, device_kind, device_index, precision, fallback_policy, cpu_threads;
    public nuint max_image_bytes, max_pixels, max_detections;
}
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct FyImage
{
    public uint struct_size, struct_version, required_flags;
    public byte* data;
    public nuint byte_length;
    public uint width, height;
    public nuint row_stride;
    public uint pixel_format;
}
[StructLayout(LayoutKind.Sequential)]
internal struct FyDetectOptions
{
    public uint struct_size, struct_version, required_flags;
    public float score_threshold;
    public nuint max_detections;
}
[StructLayout(LayoutKind.Sequential)]
internal struct FySessionInfo
{
    public uint struct_size, struct_version, required_flags;
    public uint backend_id, device_kind, device_index, precision, execution_flags;
}
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct FyResultInfo
{
    public uint struct_size, struct_version, required_flags, source_width, source_height;
    public nuint count;
    public uint result_flags, backend_id, device_kind, device_index, precision, execution_flags;
    public fixed byte model_id[32];
}
[StructLayout(LayoutKind.Sequential)]
internal struct FyDetection
{
    public float x_min, y_min, x_max, y_max, score;
    public uint class_id;
}
[StructLayout(LayoutKind.Sequential)]
internal struct FyCapability
{
    public uint backend_id, device_kind, device_index, flags, precision_mask;
}

internal static class Constants
{
    internal const int FY_OK = 0, FY_INVALID_ARGUMENT = 1, FY_UNSUPPORTED_ABI = 2;
    internal const int FY_UNSUPPORTED_FEATURE = 3, FY_UNSUPPORTED_MODEL = 4;
    internal const int FY_DEVICE_UNAVAILABLE = 5, FY_BUSY = 6, FY_LIMIT_EXCEEDED = 7;
    internal const int FY_BUFFER_TOO_SMALL = 8, FY_MODEL_OUTPUT_INVALID = 9;
    internal const int FY_BACKEND_ERROR = 10, FY_POISONED = 11, FY_INTERNAL_ERROR = 12;
    internal const uint FY_ABI_MAJOR = 1, FY_ABI_MINOR = 0, FY_STRUCT_VERSION = 1;
    internal const uint FY_RGB8 = 1, FY_BGR8 = 2, FY_RGBA8 = 3;
    internal const uint FY_DEVICE_CPU = 1, FY_DEVICE_GPU = 2, FY_PRECISION_F32 = 1;
    internal const uint FY_FALLBACK_NONE = 0, FY_RESULT_TRUNCATED = 1;
    internal const uint FY_CAP_COMPILED = 1, FY_CAP_AVAILABLE = 2, FY_CAP_TEST_ONLY = 4;
}
