// SPDX-License-Identifier: Apache-2.0
using System.Runtime.InteropServices;
namespace FactsYOLO.Smoke;

internal abstract unsafe class ResourceHandle : SafeHandle
{
    private int releaseCalls;
    protected ResourceHandle() : base(0, true) { }
    public override bool IsInvalid => handle == 0;
    internal int ReleaseCalls => Volatile.Read(ref releaseCalls);
    internal void Adopt(ref nint raw)
    {
        if (raw == 0 || !IsInvalid || IsClosed) throw new InvalidOperationException("Invalid ownership transfer.");
        SetHandle(raw);
        raw = 0; // No fallible work between adoption and clearing the local owner.
    }
    protected abstract int ReleaseNative(nint* slot);
    protected sealed override bool ReleaseHandle()
    {
        nint slot = handle;
        try
        {
            int status = ReleaseNative(&slot);
            if (status != 0 || slot != 0) return false;
            SetHandle(0);
            return true;
        }
        catch (Exception) { return false; } // Finalization must never throw.
        finally { Interlocked.Increment(ref releaseCalls); }
    }
}
internal sealed unsafe class ModelHandle : ResourceHandle
{
    protected override int ReleaseNative(nint* slot) => Native.ModelRelease(slot, null);
}
internal sealed unsafe class SessionHandle : ResourceHandle
{
    protected override int ReleaseNative(nint* slot) => Native.SessionRelease(slot, null);
}
internal sealed unsafe class ResultHandle : ResourceHandle
{
    protected override int ReleaseNative(nint* slot) => Native.ResultRelease(slot, null);
}

// A managed wrapper owns one SafeHandle. Its short lock orders Dispose against
// lease admission, never against the duration of an expensive native operation.
internal sealed class Lifetime<T>(T handle) : IDisposable where T : ResourceHandle
{
    private readonly object gate = new();
    private bool disposed;
    internal T Handle { get; } = handle;
    internal Lease Acquire()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return new Lease(Handle);
        }
    }
    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
        }
        Handle.Dispose(); // Physical release waits for outstanding DangerousAddRef leases.
    }
}

internal sealed class Lease : IDisposable
{
    private SafeHandle? owner;
    internal nint Pointer { get; }
    internal Lease(SafeHandle handle)
    {
        bool added = false;
        try
        {
            handle.DangerousAddRef(ref added);
            if (!added) throw new ObjectDisposedException(nameof(handle));
            Pointer = handle.DangerousGetHandle();
            if (Pointer == 0) throw new InvalidOperationException("Empty resource handle.");
            owner = handle;
        }
        catch
        {
            if (added) handle.DangerousRelease();
            throw;
        }
    }
    public void Dispose() => Interlocked.Exchange(ref owner, null)?.DangerousRelease();
}
