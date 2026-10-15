// SPDX-License-Identifier: Apache-2.0
namespace FactsYOLO;

/// <summary>Owns a model opened by the existing native backend.</summary>
public sealed unsafe class Model : IDisposable
{
    private readonly Lifetime<ModelHandle> lifetime = new(new ModelHandle());
    private Model() { }
    public static Model Open(ReadOnlySpan<byte> bytes, ModelOptions? options = null)
    {
        NativeRuntime.EnsureLoaded();
        options ??= new();
        Model owner = new();
        nint raw = 0;
        byte* buffer = stackalloc byte[Diagnostics.Capacity];
        FyDiagnostic diagnostic = Diagnostics.Create(buffer);
        try
        {
            FyModelOptions native = new() { struct_size = (uint)sizeof(FyModelOptions), struct_version = 1,
                max_artifact_bytes = options.MaxArtifactBytes, max_parser_steps = options.MaxParserSteps };
            fixed (byte* pointer = bytes)
                Diagnostics.Check(Native.ModelOpen(1, pointer, (nuint)bytes.Length, &native, &raw, &diagnostic), &diagnostic);
            owner.lifetime.Handle.Adopt(ref raw);
            return owner;
        }
        catch { owner.Dispose(); throw; }
        finally { if (raw != 0) Native.ModelRelease(&raw, null); }
    }
    public Session CreateSession(SessionOptions? options = null)
    {
        using Lease lease = lifetime.Acquire();
        return Session.Create(lease.Pointer, options ?? new());
    }
    public void Dispose() => lifetime.Dispose();
}
