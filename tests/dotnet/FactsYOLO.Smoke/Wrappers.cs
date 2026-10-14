// SPDX-License-Identifier: Apache-2.0
namespace FactsYOLO.Smoke;

// Unpublished test adapters. All vision behavior remains inside Rust.
internal sealed unsafe class Model : IDisposable
{
    private readonly Lifetime<ModelHandle> lifetime = new(new ModelHandle());
    private Model() { }
    internal static Model Open(ReadOnlySpan<byte> bytes)
    {
        Model owner = new();
        nint raw = 0;
        try
        {
            FyModelOptions options = new()
            {
                struct_size = (uint)sizeof(FyModelOptions), struct_version = 1,
                max_artifact_bytes = 1024 * 1024, max_parser_steps = 65536
            };
            fixed (byte* pointer = bytes)
                Status.Check(Native.ModelOpen(1, pointer, (nuint)bytes.Length, &options, &raw, null));
            owner.lifetime.Handle.Adopt(ref raw);
            return owner;
        }
        catch { owner.Dispose(); throw; }
        finally { if (raw != 0) Native.ModelRelease(&raw, null); }
    }
    internal static Model Fixture(uint id)
    {
        Model owner = new();
        nint raw = 0;
        try
        {
            Status.Check(Native.TestModel(1, id, &raw, null));
            owner.lifetime.Handle.Adopt(ref raw);
            return owner;
        }
        catch { owner.Dispose(); throw; }
        finally { if (raw != 0) Native.ModelRelease(&raw, null); }
    }
    internal Session CreateSession(FySessionOptions options)
    {
        using Lease lease = lifetime.Acquire();
        return Session.Create(lease.Pointer, options);
    }
    public void Dispose() => lifetime.Dispose();
}

internal sealed unsafe class Session : IDisposable
{
    private readonly Lifetime<SessionHandle> lifetime = new(new SessionHandle());
    private Session() { }
    // The caller holds a model lease until this method returns.
    internal static Session Create(nint model, FySessionOptions options)
    {
        Session owner = new();
        nint raw = 0;
        try
        {
            Status.Check(Native.SessionCreate(1, model, &options, &raw, null));
            owner.lifetime.Handle.Adopt(ref raw);
            return owner;
        }
        catch { owner.Dispose(); throw; }
        finally { if (raw != 0) Native.SessionRelease(&raw, null); }
    }
    internal FySessionInfo Info()
    {
        using Lease lease = lifetime.Acquire();
        FySessionInfo info = new() { struct_size = (uint)sizeof(FySessionInfo), struct_version = 1 };
        Status.Check(Native.SessionDescribe(lease.Pointer, &info, null));
        return info;
    }
    internal Result Detect(ReadOnlySpan<byte> pixels, uint width = 2, uint height = 2,
        nuint stride = 6, uint format = 1, float threshold = .5f, nuint cap = 16)
    {
        using Lease lease = lifetime.Acquire();
        // Both managed ownership objects exist before native result allocation.
        Result owner = new();
        nint raw = 0;
        try
        {
            FyDetectOptions options = new()
            {
                struct_size = (uint)sizeof(FyDetectOptions), struct_version = 1,
                score_threshold = threshold, max_detections = cap
            };
            fixed (byte* pointer = pixels)
            {
                FyImage image = new()
                {
                    struct_size = (uint)sizeof(FyImage), struct_version = 1,
                    data = pointer, byte_length = (nuint)pixels.Length,
                    width = width, height = height, row_stride = stride, pixel_format = format
                };
                Status.Check(Native.Detect(lease.Pointer, &image, &options, &raw, null));
            }
            owner.Adopt(ref raw);
            return owner;
        }
        catch { owner.Dispose(); throw; }
        finally { if (raw != 0) Native.ResultRelease(&raw, null); }
    }
    internal int WaitRunningForTest(uint timeoutMilliseconds)
    {
        using Lease lease = lifetime.Acquire();
        return Native.WaitRunning(lease.Pointer, timeoutMilliseconds);
    }
    // Only fixture control may retain a disposed but still-leased SafeHandle.
    // DangerousAddRef atomically retains it or rejects an already closed handle;
    // a past native barrier alone is not a lifetime guarantee. Ordinary methods
    // continue to use lifetime.Acquire(), which rejects all disposed admission.
    internal Lease AcquireControlForTest() => new(lifetime.Handle);
    internal int ReleaseCalls => lifetime.Handle.ReleaseCalls;
    public void Dispose() => lifetime.Dispose();
}

internal sealed unsafe class Result : IDisposable
{
    private readonly Lifetime<ResultHandle> lifetime = new(new ResultHandle());
    internal void Adopt(ref nint raw) => lifetime.Handle.Adopt(ref raw);
    internal FyResultInfo Info()
    {
        using Lease lease = lifetime.Acquire();
        FyResultInfo info = new() { struct_size = (uint)sizeof(FyResultInfo), struct_version = 1 };
        Status.Check(Native.ResultDescribe(lease.Pointer, &info, null));
        return info;
    }
    internal FyDetection[] Copy()
    {
        using Lease lease = lifetime.Acquire();
        nuint required = 0;
        Status.Check(Native.ResultCopy(lease.Pointer, null, 0, &required, null));
        if (required > (nuint)Array.MaxLength) throw new InvalidOperationException("Result exceeds managed array limit.");
        FyDetection[] values = new FyDetection[(int)required];
        fixed (FyDetection* pointer = values)
            Status.Check(Native.ResultCopy(lease.Pointer, pointer, (nuint)values.Length, &required, null));
        if (required != (nuint)values.Length) throw new InvalidOperationException("Immutable result count changed.");
        return values;
    }
    internal void CopyTo(Span<FyDetection> destination)
    {
        using Lease lease = lifetime.Acquire();
        nuint required = 0;
        fixed (FyDetection* pointer = destination)
            Status.Check(Native.ResultCopy(lease.Pointer, pointer, (nuint)destination.Length, &required, null));
        // A null/empty span becomes the native count query, not a successful copy
        // of nonempty results. Preserve the wrapper's destination-capacity contract.
        if (required > (nuint)destination.Length) throw new NativeFailure(8);
    }
    public void Dispose() => lifetime.Dispose();
}
