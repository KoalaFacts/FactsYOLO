// SPDX-License-Identifier: Apache-2.0
namespace FactsYOLO;

/// <summary>Immutable native-owned result, independent of model, session and input pixels.</summary>
public sealed unsafe class Result : IDisposable
{
    private readonly Lifetime<ResultHandle> lifetime = new(new ResultHandle());
    internal Result() { }
    internal void Adopt(ref nint raw) => lifetime.Handle.Adopt(ref raw);
    public ResultInfo Info()
    {
        using Lease lease = lifetime.Acquire();
        byte* buffer = stackalloc byte[Diagnostics.Capacity];
        FyDiagnostic diagnostic = Diagnostics.Create(buffer);
        FyResultInfo info = new() { struct_size = (uint)sizeof(FyResultInfo), struct_version = 1 };
        Diagnostics.Check(Native.ResultDescribe(lease.Pointer, &info, &diagnostic), &diagnostic);
        return new(info.source_width, info.source_height, info.count, info.result_flags, info.backend_id,
            info.device_kind, info.device_index, info.precision, info.execution_flags,
            Convert.ToHexString(new ReadOnlySpan<byte>(info.model_id, 32)).ToLowerInvariant());
    }
    public Detection[] Copy()
    {
        using Lease lease = lifetime.Acquire();
        byte* buffer = stackalloc byte[Diagnostics.Capacity];
        FyDiagnostic diagnostic = Diagnostics.Create(buffer);
        nuint required = 0;
        Diagnostics.Check(Native.ResultCopy(lease.Pointer, null, 0, &required, &diagnostic), &diagnostic);
        if (required > (nuint)Array.MaxLength) throw new InvalidOperationException("Result exceeds managed array limit.");
        Detection[] output = new Detection[(int)required];
        CopyWithLease(lease.Pointer, output, &diagnostic);
        return output;
    }
    public void CopyTo(Span<Detection> destination)
    {
        using Lease lease = lifetime.Acquire();
        byte* buffer = stackalloc byte[Diagnostics.Capacity];
        FyDiagnostic diagnostic = Diagnostics.Create(buffer);
        nuint required = 0;
        Diagnostics.Check(Native.ResultCopy(lease.Pointer, null, 0, &required, &diagnostic), &diagnostic);
        // The immutable result count bounds scratch storage, even for a large caller span.
        // A short span still reaches the native copy and preserves its original error.
        int capacity = required < (nuint)destination.Length ? (int)required : destination.Length;
        CopyWithLease(lease.Pointer, destination[..capacity], &diagnostic);
    }
    private static void CopyWithLease(nint result, Span<Detection> destination, FyDiagnostic* diagnostic)
    {
        nuint required = 0;
        FyDetection[] values = new FyDetection[destination.Length];
        FyDetection emptyDestination = default;
        fixed (FyDetection* pointer = values)
            Diagnostics.Check(Native.ResultCopy(result, values.Length == 0 ? &emptyDestination : pointer,
                (nuint)values.Length, &required, diagnostic), diagnostic);
        for (int i = 0; i < (int)required; i++)
        {
            FyDetection v = values[i];
            destination[i] = new(v.x_min, v.y_min, v.x_max, v.y_max, v.score, v.class_id);
        }
    }
    public void Dispose() => lifetime.Dispose();
}
