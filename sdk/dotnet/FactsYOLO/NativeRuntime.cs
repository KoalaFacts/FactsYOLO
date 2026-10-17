// SPDX-License-Identifier: Apache-2.0
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
namespace FactsYOLO;

/// <summary>Loads the application's installed native asset, once per process.</summary>
public static unsafe class NativeRuntime
{
    private static readonly Lazy<RuntimeInfo> runtime = new(Load);
    // Cached imports and SafeHandle finalizers can outlive all public wrappers.
    // Keep the validated native module loaded for process lifetime.
    private static nint module;
    public static RuntimeInfo Query() => runtime.Value;
    internal static void EnsureLoaded() => _ = runtime.Value;

    private static RuntimeInfo Load()
    {
        string architecture = RuntimeInformation.ProcessArchitecture.ToString();
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64 ||
            (!OperatingSystem.IsLinux() && !OperatingSystem.IsWindows()))
            throw new PlatformNotSupportedException($"FactsYOLO supports Linux/Windows x64 processes; process={architecture}, runtime={RuntimeInformation.RuntimeIdentifier}.");
        string rid = OperatingSystem.IsWindows() ? "win-x64" : "linux-x64";
        string file = OperatingSystem.IsWindows() ? "facts_yolo.dll" : "libfacts_yolo.so";
        string[] paths = [Path.Combine(AppContext.BaseDirectory, file),
            Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", file)];
        string? selected = paths.FirstOrDefault(File.Exists);
        string context = $"FactsYOLO RID={rid}, process={architecture}; attempted: {string.Join(", ", paths)}";
        if (selected is null) throw new DllNotFoundException($"{context}. Native package asset is absent.");
        nint loaded = 0;
        try
        {
            // An existing but broken first asset is an error, not a reason to fall back.
            loaded = NativeLibrary.Load(selected);
            foreach (string name in new[] { "fy_abi_query", "fy_capabilities_copy", "fy_model_open",
                "fy_model_release", "fy_session_create", "fy_session_describe", "fy_session_release",
                "fy_detect", "fy_result_describe", "fy_result_copy", "fy_result_release" })
                _ = NativeLibrary.GetExport(loaded, name);
            var query = (delegate* unmanaged[Cdecl]<FyAbiInfo*, FyDiagnostic*, int>)NativeLibrary.GetExport(loaded, "fy_abi_query");
            FyAbiInfo abi = new() { struct_size = (uint)sizeof(FyAbiInfo), struct_version = 1 };
            byte* buffer = stackalloc byte[Diagnostics.Capacity];
            FyDiagnostic diagnostic = Diagnostics.Create(buffer);
            Diagnostics.Check(query(&abi, &diagnostic), &diagnostic);
            if (abi.abi_major != 1) throw new InvalidOperationException($"Incompatible ABI major {abi.abi_major}; expected 1.");
            ReadOnlySpan<byte> id = new(abi.build_id, 64);
            int end = id.IndexOf((byte)0);
            if (end < 0) throw new InvalidOperationException("Native build ID is not terminated.");
            string build = Encoding.UTF8.GetString(id[..end]);
            // Query directly before registration so every failure can free the module safely.
            var caps = (delegate* unmanaged[Cdecl]<FyCapability*, nuint, nuint*, FyDiagnostic*, int>)NativeLibrary.GetExport(loaded, "fy_capabilities_copy");
            nuint count = 0;
            Diagnostics.Check(caps(null, 0, &count, &diagnostic), &diagnostic);
            if (count > 1024) throw new InvalidOperationException("Native capability count exceeds SDK bound.");
            FyCapability[] values = new FyCapability[(int)count];
            fixed (FyCapability* pointer = values)
                Diagnostics.Check(caps(pointer, count, &count, &diagnostic), &diagnostic);
            if (count != (nuint)values.Length) throw new InvalidOperationException("Native capability count changed.");
            RuntimeInfo info = new(abi.abi_major, abi.abi_minor, build, rid, selected,
                Array.AsReadOnly(values.Select(v => new Capability(v.backend_id, v.device_kind, v.device_index, v.flags, v.precision_mask)).ToArray()));
            NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
            module = loaded;
            return info;
        }
        catch (Exception error)
        {
            if (loaded != 0) NativeLibrary.Free(loaded);
            throw new DllNotFoundException($"{context}. Loading '{selected}' failed: {error.Message}", error);
        }
    }
    private static nint Resolve(string name, Assembly assembly, DllImportSearchPath? search) =>
        name == Native.Library && module != 0 ? module : throw new DllNotFoundException(name);
}
