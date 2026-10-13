// SPDX-License-Identifier: Apache-2.0
namespace FactsYOLO.Smoke;

// Safe rejecting scaffolding for the red checkpoint. No native ownership is
// transferred here. The executable tests must fail before implementation.
internal sealed class Model : IDisposable
{
    internal static Model Open(ReadOnlySpan<byte> bytes) => throw new NotImplementedException();
    internal static Model Fixture(uint id) => throw new NotImplementedException();
    internal Session CreateSession(FySessionOptions options) => throw new NotImplementedException();
    public void Dispose() { }
}
internal sealed class Session : IDisposable
{
    internal FySessionInfo Info() => throw new NotImplementedException();
    internal Result Detect(ReadOnlySpan<byte> pixels, uint width = 2, uint height = 2,
        nuint stride = 6, uint format = 1, float threshold = .5f, nuint cap = 16) => throw new NotImplementedException();
    internal nint AddressForTest() => throw new NotImplementedException();
    internal int ReleaseCalls => 0;
    public void Dispose() { }
}
internal sealed class Result : IDisposable
{
    internal FyResultInfo Info() => throw new NotImplementedException();
    internal FyDetection[] Copy() => throw new NotImplementedException();
    internal void CopyTo(Span<FyDetection> destination) => throw new NotImplementedException();
    public void Dispose() { }
}
internal static class Layout
{
    internal static void Verify(string reference) => throw new NotImplementedException();
}
