// SPDX-License-Identifier: Apache-2.0
namespace FactsYOLO;

/// <summary>A native session; concurrent Detect calls retain the native BUSY contract.</summary>
public sealed unsafe class Session : IDisposable
{
    private readonly Lifetime<SessionHandle> lifetime = new(new SessionHandle());
    private Session() { }
    internal static Session Create(nint model, SessionOptions options)
    {
        Session owner = new();
        nint raw = 0;
        byte* buffer = stackalloc byte[Diagnostics.Capacity];
        FyDiagnostic diagnostic = Diagnostics.Create(buffer);
        try
        {
            FySessionOptions native = new() { struct_size = (uint)sizeof(FySessionOptions), struct_version = 1,
                backend_id = options.BackendId, device_kind = options.DeviceKind, device_index = options.DeviceIndex,
                precision = options.Precision, fallback_policy = options.FallbackPolicy, cpu_threads = options.CpuThreads,
                max_image_bytes = options.MaxImageBytes, max_pixels = options.MaxPixels, max_detections = options.MaxDetections };
            Diagnostics.Check(Native.SessionCreate(1, model, &native, &raw, &diagnostic), &diagnostic);
            owner.lifetime.Handle.Adopt(ref raw);
            return owner;
        }
        catch { owner.Dispose(); throw; }
        finally { if (raw != 0) Native.SessionRelease(&raw, null); }
    }
    public SessionInfo Info()
    {
        using Lease lease = lifetime.Acquire();
        byte* buffer = stackalloc byte[Diagnostics.Capacity];
        FyDiagnostic diagnostic = Diagnostics.Create(buffer);
        FySessionInfo info = new() { struct_size = (uint)sizeof(FySessionInfo), struct_version = 1 };
        Diagnostics.Check(Native.SessionDescribe(lease.Pointer, &info, &diagnostic), &diagnostic);
        return new(info.backend_id, info.device_kind, info.device_index, info.precision, info.execution_flags);
    }
    /// <summary>Borrows pixels only during this call. Coordinates and all vision logic come from Rust.</summary>
    public Result Detect(ReadOnlySpan<byte> pixels, uint width, uint height, nuint rowStride,
        PixelFormat format, DetectOptions? options = null)
    {
        using Lease lease = lifetime.Acquire();
        options ??= new();
        Result owner = new();
        nint raw = 0;
        byte* buffer = stackalloc byte[Diagnostics.Capacity];
        FyDiagnostic diagnostic = Diagnostics.Create(buffer);
        try
        {
            FyDetectOptions native = new() { struct_size = (uint)sizeof(FyDetectOptions), struct_version = 1,
                score_threshold = options.ScoreThreshold, max_detections = options.MaxDetections };
            fixed (byte* pointer = pixels)
            {
                FyImage image = new() { struct_size = (uint)sizeof(FyImage), struct_version = 1,
                    data = pointer, byte_length = (nuint)pixels.Length, width = width, height = height,
                    row_stride = rowStride, pixel_format = (uint)format };
                Diagnostics.Check(Native.Detect(lease.Pointer, &image, &native, &raw, &diagnostic), &diagnostic);
            }
            owner.Adopt(ref raw);
            return owner;
        }
        catch { owner.Dispose(); throw; }
        finally { if (raw != 0) Native.ResultRelease(&raw, null); }
    }
    public void Dispose() => lifetime.Dispose();
}
