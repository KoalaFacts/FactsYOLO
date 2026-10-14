// SPDX-License-Identifier: Apache-2.0
using System.Runtime.InteropServices;
using static FactsYOLO.Smoke.Constants;
namespace FactsYOLO.Smoke;

// Compiler-derived pointer offsets, no reflection-generated native layout.
internal static unsafe class Layout
{
    [StructLayout(LayoutKind.Sequential)]
    private struct Probe<T> where T : unmanaged { public byte prefix; public T value; }
    private static void T<TValue>(List<string> lines, string name) where TValue : unmanaged
    {
        Probe<TValue> probe = new() { prefix = 0, value = default };
        long alignment = (byte*)&probe.value - (byte*)&probe;
        lines.Add($"T {name} {sizeof(TValue)} {alignment}");
    }
    private static void F(List<string> lines, string type, string field, void* record, void* value) =>
        lines.Add($"F {type} {field} {(byte*)value - (byte*)record}");
    private static void K(List<string> lines, string name, long value) => lines.Add($"K {name} {value}");
    internal static void Verify(string reference)
    {
        if (!File.Exists(reference) || new FileInfo(reference).Length > 65536)
            throw new InvalidOperationException("Rust layout evidence absent or excessive.");
        List<string> lines = [$"P pointer {sizeof(nuint)}"];
        FyAbiInfo a = default;
        T<FyAbiInfo>(lines, "fy_abi_info");
        F(lines, "fy_abi_info", "struct_size", &a, &a.struct_size);
        F(lines, "fy_abi_info", "struct_version", &a, &a.struct_version);
        F(lines, "fy_abi_info", "required_flags", &a, &a.required_flags);
        F(lines, "fy_abi_info", "abi_major", &a, &a.abi_major);
        F(lines, "fy_abi_info", "abi_minor", &a, &a.abi_minor);
        F(lines, "fy_abi_info", "build_id", &a, a.build_id);
        FyDiagnostic d = default;
        T<FyDiagnostic>(lines, "fy_diagnostic");
        F(lines, "fy_diagnostic", "struct_size", &d, &d.struct_size);
        F(lines, "fy_diagnostic", "struct_version", &d, &d.struct_version);
        F(lines, "fy_diagnostic", "required_flags", &d, &d.required_flags);
        F(lines, "fy_diagnostic", "data", &d, &d.data);
        F(lines, "fy_diagnostic", "capacity", &d, &d.capacity);
        F(lines, "fy_diagnostic", "required_bytes", &d, &d.required_bytes);
        FyModelOptions m = default;
        T<FyModelOptions>(lines, "fy_model_options");
        F(lines, "fy_model_options", "struct_size", &m, &m.struct_size);
        F(lines, "fy_model_options", "struct_version", &m, &m.struct_version);
        F(lines, "fy_model_options", "required_flags", &m, &m.required_flags);
        F(lines, "fy_model_options", "max_artifact_bytes", &m, &m.max_artifact_bytes);
        F(lines, "fy_model_options", "max_parser_steps", &m, &m.max_parser_steps);
        FySessionOptions s = default;
        T<FySessionOptions>(lines, "fy_session_options");
        F(lines, "fy_session_options", "struct_size", &s, &s.struct_size);
        F(lines, "fy_session_options", "struct_version", &s, &s.struct_version);
        F(lines, "fy_session_options", "required_flags", &s, &s.required_flags);
        F(lines, "fy_session_options", "backend_id", &s, &s.backend_id);
        F(lines, "fy_session_options", "device_kind", &s, &s.device_kind);
        F(lines, "fy_session_options", "device_index", &s, &s.device_index);
        F(lines, "fy_session_options", "precision", &s, &s.precision);
        F(lines, "fy_session_options", "fallback_policy", &s, &s.fallback_policy);
        F(lines, "fy_session_options", "cpu_threads", &s, &s.cpu_threads);
        F(lines, "fy_session_options", "max_image_bytes", &s, &s.max_image_bytes);
        F(lines, "fy_session_options", "max_pixels", &s, &s.max_pixels);
        F(lines, "fy_session_options", "max_detections", &s, &s.max_detections);
        FyImage i = default;
        T<FyImage>(lines, "fy_image");
        F(lines, "fy_image", "struct_size", &i, &i.struct_size);
        F(lines, "fy_image", "struct_version", &i, &i.struct_version);
        F(lines, "fy_image", "required_flags", &i, &i.required_flags);
        F(lines, "fy_image", "data", &i, &i.data);
        F(lines, "fy_image", "byte_length", &i, &i.byte_length);
        F(lines, "fy_image", "width", &i, &i.width);
        F(lines, "fy_image", "height", &i, &i.height);
        F(lines, "fy_image", "row_stride", &i, &i.row_stride);
        F(lines, "fy_image", "pixel_format", &i, &i.pixel_format);
        FyDetectOptions o = default;
        T<FyDetectOptions>(lines, "fy_detect_options");
        F(lines, "fy_detect_options", "struct_size", &o, &o.struct_size);
        F(lines, "fy_detect_options", "struct_version", &o, &o.struct_version);
        F(lines, "fy_detect_options", "required_flags", &o, &o.required_flags);
        F(lines, "fy_detect_options", "score_threshold", &o, &o.score_threshold);
        F(lines, "fy_detect_options", "max_detections", &o, &o.max_detections);
        FySessionInfo si = default;
        T<FySessionInfo>(lines, "fy_session_info");
        F(lines, "fy_session_info", "struct_size", &si, &si.struct_size);
        F(lines, "fy_session_info", "struct_version", &si, &si.struct_version);
        F(lines, "fy_session_info", "required_flags", &si, &si.required_flags);
        F(lines, "fy_session_info", "backend_id", &si, &si.backend_id);
        F(lines, "fy_session_info", "device_kind", &si, &si.device_kind);
        F(lines, "fy_session_info", "device_index", &si, &si.device_index);
        F(lines, "fy_session_info", "precision", &si, &si.precision);
        F(lines, "fy_session_info", "execution_flags", &si, &si.execution_flags);
        FyResultInfo r = default;
        T<FyResultInfo>(lines, "fy_result_info");
        F(lines, "fy_result_info", "struct_size", &r, &r.struct_size);
        F(lines, "fy_result_info", "struct_version", &r, &r.struct_version);
        F(lines, "fy_result_info", "required_flags", &r, &r.required_flags);
        F(lines, "fy_result_info", "source_width", &r, &r.source_width);
        F(lines, "fy_result_info", "source_height", &r, &r.source_height);
        F(lines, "fy_result_info", "count", &r, &r.count);
        F(lines, "fy_result_info", "result_flags", &r, &r.result_flags);
        F(lines, "fy_result_info", "backend_id", &r, &r.backend_id);
        F(lines, "fy_result_info", "device_kind", &r, &r.device_kind);
        F(lines, "fy_result_info", "device_index", &r, &r.device_index);
        F(lines, "fy_result_info", "precision", &r, &r.precision);
        F(lines, "fy_result_info", "execution_flags", &r, &r.execution_flags);
        F(lines, "fy_result_info", "model_id", &r, r.model_id);
        FyDetection box = default;
        T<FyDetection>(lines, "fy_detection");
        F(lines, "fy_detection", "x_min", &box, &box.x_min);
        F(lines, "fy_detection", "y_min", &box, &box.y_min);
        F(lines, "fy_detection", "x_max", &box, &box.x_max);
        F(lines, "fy_detection", "y_max", &box, &box.y_max);
        F(lines, "fy_detection", "score", &box, &box.score);
        F(lines, "fy_detection", "class_id", &box, &box.class_id);
        FyCapability c = default;
        T<FyCapability>(lines, "fy_capability");
        F(lines, "fy_capability", "backend_id", &c, &c.backend_id);
        F(lines, "fy_capability", "device_kind", &c, &c.device_kind);
        F(lines, "fy_capability", "device_index", &c, &c.device_index);
        F(lines, "fy_capability", "flags", &c, &c.flags);
        F(lines, "fy_capability", "precision_mask", &c, &c.precision_mask);
        K(lines, nameof(FY_OK), FY_OK);
        K(lines, nameof(FY_INVALID_ARGUMENT), FY_INVALID_ARGUMENT);
        K(lines, nameof(FY_UNSUPPORTED_ABI), FY_UNSUPPORTED_ABI);
        K(lines, nameof(FY_UNSUPPORTED_FEATURE), FY_UNSUPPORTED_FEATURE);
        K(lines, nameof(FY_UNSUPPORTED_MODEL), FY_UNSUPPORTED_MODEL);
        K(lines, nameof(FY_DEVICE_UNAVAILABLE), FY_DEVICE_UNAVAILABLE);
        K(lines, nameof(FY_BUSY), FY_BUSY);
        K(lines, nameof(FY_LIMIT_EXCEEDED), FY_LIMIT_EXCEEDED);
        K(lines, nameof(FY_BUFFER_TOO_SMALL), FY_BUFFER_TOO_SMALL);
        K(lines, nameof(FY_MODEL_OUTPUT_INVALID), FY_MODEL_OUTPUT_INVALID);
        K(lines, nameof(FY_BACKEND_ERROR), FY_BACKEND_ERROR);
        K(lines, nameof(FY_POISONED), FY_POISONED);
        K(lines, nameof(FY_INTERNAL_ERROR), FY_INTERNAL_ERROR);
        K(lines, nameof(FY_ABI_MAJOR), FY_ABI_MAJOR);
        K(lines, nameof(FY_ABI_MINOR), FY_ABI_MINOR);
        K(lines, nameof(FY_STRUCT_VERSION), FY_STRUCT_VERSION);
        K(lines, nameof(FY_RGB8), FY_RGB8);
        K(lines, nameof(FY_BGR8), FY_BGR8);
        K(lines, nameof(FY_RGBA8), FY_RGBA8);
        K(lines, nameof(FY_DEVICE_CPU), FY_DEVICE_CPU);
        K(lines, nameof(FY_DEVICE_GPU), FY_DEVICE_GPU);
        K(lines, nameof(FY_PRECISION_F32), FY_PRECISION_F32);
        K(lines, nameof(FY_FALLBACK_NONE), FY_FALLBACK_NONE);
        K(lines, nameof(FY_RESULT_TRUNCATED), FY_RESULT_TRUNCATED);
        K(lines, nameof(FY_CAP_COMPILED), FY_CAP_COMPILED);
        K(lines, nameof(FY_CAP_AVAILABLE), FY_CAP_AVAILABLE);
        K(lines, nameof(FY_CAP_TEST_ONLY), FY_CAP_TEST_ONLY);
        string[] expected = File.ReadAllLines(reference);
        if (expected.Length != 113 || lines.Count != 113 ||
            !expected.Order(StringComparer.Ordinal).SequenceEqual(lines.Order(StringComparer.Ordinal)))
            throw new InvalidOperationException("C# compiler layouts differ from Rust/C: " + string.Join("; ", lines.Except(expected)));
        Console.WriteLine("All 113 C#/Rust compiler layout and constant records agree.");
    }
}
