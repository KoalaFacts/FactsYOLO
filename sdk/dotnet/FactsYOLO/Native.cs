// SPDX-License-Identifier: Apache-2.0
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
[assembly: DisableRuntimeMarshalling]
namespace FactsYOLO;

internal static unsafe partial class Native
{
    internal const string Library = "facts_yolo";
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

}
