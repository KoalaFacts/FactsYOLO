// SPDX-License-Identifier: Apache-2.0
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
[assembly: DisableRuntimeMarshalling]
namespace FactsYOLO.Smoke;

internal static unsafe partial class Native
{
    internal const string Library = "facts_yolo";
    // This test process loads one explicit library for its whole lifetime.
    // It is deliberately not unloaded while cached imports/finalizers may exist.
    private static nint module;
    internal static void ValidatePath(string path)
    {
        if (!Path.IsPathFullyQualified(path)) throw new ArgumentException("Absolute library path required.");
        if (!File.Exists(path)) throw new FileNotFoundException("Native library is absent.", path);
    }
    internal static void Initialize(string path)
    {
        ValidatePath(path);
        if (module != 0) throw new InvalidOperationException("Library already selected.");
        nint loaded = NativeLibrary.Load(path);
        try { NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve); }
        catch { NativeLibrary.Free(loaded); throw; }
        module = loaded;
        FyAbiInfo info = new() { struct_size = (uint)sizeof(FyAbiInfo), struct_version = 1 };
        int status = AbiQuery(&info, null);
        if (status != 0 || info.abi_major != 1) throw new InvalidOperationException("Incompatible native ABI.");
    }
    private static nint Resolve(string name, Assembly assembly, DllImportSearchPath? search) =>
        name == Library && module != 0 ? module : throw new DllNotFoundException(name);
    internal static bool HasExport(string name) => NativeLibrary.TryGetExport(module, name, out _);

    [LibraryImport(Library, EntryPoint = "fy_abi_query")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int AbiQuery(FyAbiInfo* output, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_capabilities_copy")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int Capabilities(FyCapability* destination, nuint capacity, nuint* required, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_model_open")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int ModelOpen(uint major, byte* bytes, nuint length, FyModelOptions* options, nint* output, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_model_release")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int ModelRelease(nint* slot, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_session_create")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int SessionCreate(uint major, nint model, FySessionOptions* options, nint* output, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_session_describe")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int SessionDescribe(nint session, FySessionInfo* output, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_session_release")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int SessionRelease(nint* slot, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_detect")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int Detect(nint session, FyImage* image, FyDetectOptions* options, nint* output, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_result_describe")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int ResultDescribe(nint result, FyResultInfo* output, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_result_copy")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int ResultCopy(nint result, FyDetection* destination, nuint capacity, nuint* required, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_result_release")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int ResultRelease(nint* slot, FyDiagnostic* diagnostic);

    // The following imports are test controls, never a production model API.
    [LibraryImport(Library, EntryPoint = "fy_test_model_create")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int TestModel(uint major, uint fixture, nint* output, FyDiagnostic* diagnostic);
    [LibraryImport(Library, EntryPoint = "fy_test_session_wait_running")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int WaitRunning(nint session, uint milliseconds);
    [LibraryImport(Library, EntryPoint = "fy_test_session_resume")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int Resume(nint session);
    [LibraryImport(Library, EntryPoint = "fy_test_resource_counts")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int ResourceCounts(nuint* destination, nuint capacity);
}

internal sealed class NativeFailure(int status) : Exception($"FactsYOLO status {status}.")
{
    internal int Status { get; } = status;
}
internal static class Status
{
    // RED checkpoint: negative status handling is deliberately not implemented.
    internal static void Check(int value) { }
}
