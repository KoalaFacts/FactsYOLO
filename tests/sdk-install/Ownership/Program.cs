// SPDX-License-Identifier: Apache-2.0
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using FactsYOLO;

internal static unsafe class Program
{
    private static void Require(bool condition, string text)
    {
        if (!condition) throw new InvalidOperationException(text);
    }
    private sealed class CountedModel : ResourceHandle
    {
        internal static int Releases;
        protected override int ReleaseNative(nint* slot)
        {
            Interlocked.Increment(ref Releases);
            return Native.ModelRelease(slot, null);
        }
    }
    private static CountedModel Open(byte[] artifact)
    {
        CountedModel owner = new();
        nint raw = 0;
        try
        {
            FyModelOptions options = new() { struct_size = (uint)sizeof(FyModelOptions), struct_version = 1,
                max_artifact_bytes = 1024 * 1024, max_parser_steps = 65536 };
            fixed (byte* pointer = artifact)
                Require(Native.ModelOpen(1, pointer, (nuint)artifact.Length, &options, &raw, null) == 0, "Actual model failed to open");
            owner.Adopt(ref raw); return owner;
        }
        catch { owner.Dispose(); throw; }
        finally { if (raw != 0) Native.ModelRelease(&raw, null); }
    }
    private static int Main(string[] args)
    {
        try
        {
            Require(args.Length == 3, "Expected native path, actual model, jit|aot");
            Require(RuntimeFeature.IsDynamicCodeSupported == (args[2] == "jit"), "Wrong execution mode");
            nint module = NativeLibrary.Load(Path.GetFullPath(args[0]));
            NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, (name, _, _) => name == Native.Library ? module : throw new DllNotFoundException(name));
            using Lifetime<CountedModel> lifetime = new(Open(File.ReadAllBytes(args[1])));
            using ManualResetEventSlim admitted = new(false), resume = new(false);
            Task operation = Task.Run(() => {
                using Lease lease = lifetime.Acquire();
                admitted.Set();
                Require(resume.Wait(TimeSpan.FromSeconds(10)), "Resume timeout");
                // Do not dereference a freed handle even in the deliberate
                // no-retain regression mutant used to prove this test.
                Require(Volatile.Read(ref CountedModel.Releases) == 0, "Model freed while admitted lease is live");
                FySessionOptions options = new() { struct_size = (uint)sizeof(FySessionOptions), struct_version = 1,
                    backend_id = 1, device_kind = 1, precision = 1, cpu_threads = 1,
                    max_image_bytes = 65536, max_pixels = 4096, max_detections = 16 };
                nint session = 0;
                try { Require(Native.SessionCreate(1, lease.Pointer, &options, &session, null) == 0, "Retained actual model was unusable after Dispose"); }
                finally { if (session != 0) Require(Native.SessionRelease(&session, null) == 0, "Native session release failed"); }
            });
            Require(admitted.Wait(TimeSpan.FromSeconds(10)), "Lease was never admitted");
            try
            {
                lifetime.Dispose(); lifetime.Dispose();
                Require(Volatile.Read(ref CountedModel.Releases) == 0, "Dispose released an admitted handle");
                bool rejected = false;
                try { using Lease unexpected = lifetime.Acquire(); }
                catch (ObjectDisposedException) { rejected = true; }
                Require(rejected, "Disposed lifetime admitted a new operation");
            }
            finally { resume.Set(); }
            operation.GetAwaiter().GetResult();
            Require(Volatile.Read(ref CountedModel.Releases) == 1, "Expected exactly one physical model release");
            lifetime.Dispose();
            Require(Volatile.Read(ref CountedModel.Releases) == 1, "Double Dispose released twice");
            Console.WriteLine($"PASS {args[2]} admitted SDK lease survives Dispose; real native session created after Dispose; exactly one model release");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
