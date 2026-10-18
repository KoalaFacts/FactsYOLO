// SPDX-License-Identifier: Apache-2.0
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using FactsYOLO;

internal static unsafe class Program
{
    internal const string PrematureRelease = "Result physically released before admitted CopyTo completed";
    internal static void Require(bool condition, string text)
    {
        if (!condition) throw new InvalidOperationException(text);
    }

    private static void Check(string mode, string phase)
    {
        Require(RuntimeFeature.IsDynamicCodeSupported == (mode == "jit"), "Wrong execution mode");
        Require(phase is "query" or "copy", "Unknown pause phase");
        using ManualResetEventSlim admitted = new(false), resume = new(false);
        Native.Configure(phase, admitted, resume);
        using Result result = new();
        nint raw = Native.CreateResult();
        result.Adopt(ref raw);
        Require(raw == 0, "Result adoption did not clear the local owner");
        Detection[] expected = [new(1, 2, 3, 4, .75f, 7), new(5, 6, 8, 9, .5f, 11)];
        Detection sentinel = new(-1, -2, -3, -4, -5, uint.MaxValue);
        Detection[] storage = Enumerable.Repeat(sentinel, 6).ToArray();
        Task operation = Task.Run(() => result.CopyTo(storage.AsSpan(1, 4)));
        Exception? failure = null;
        try
        {
            Require(admitted.Wait(TimeSpan.FromSeconds(10)), "CopyTo never reached the requested native boundary");
            Require(!operation.IsCompleted, "CopyTo did not pause at the native boundary");
            result.Dispose(); result.Dispose();
            Require(Native.PhysicalFrees == 0 && Native.ReleaseCalls == 0, PrematureRelease);
            int queries = Native.Queries, copies = Native.Copies;
            bool rejected = false;
            try { result.CopyTo(new Detection[4]); }
            catch (ObjectDisposedException) { rejected = true; }
            Require(rejected, "Disposed Result admitted a new CopyTo");
            Require(Native.Queries == queries && Native.Copies == copies, "Rejected CopyTo reached the native boundary");
        }
        catch (Exception error) { failure = error; }
        finally { resume.Set(); }
        // Always join before disposing the barriers, including deliberate mutants.
        try { operation.GetAwaiter().GetResult(); }
        catch (Exception error) { failure ??= error; }
        if (failure is not null) throw failure;
        Require(storage.AsSpan(1, expected.Length).SequenceEqual(expected), "Admitted CopyTo changed detection data");
        Require(storage[0] == sentinel && storage.AsSpan(3).ToArray().All(value => value == sentinel),
            "CopyTo changed the destination tail or surrounding canaries");
        Require(Native.Queries == 1 && Native.Copies == 1, "Expected one count query and one copy");
        Require(Native.PhysicalFrees == 1 && Native.ReleaseCalls == 1, "Expected exactly one physical result release");
        result.Dispose();
        Require(Native.PhysicalFrees == 1 && Native.ReleaseCalls == 1, "Repeated Dispose released the result twice");
        Console.WriteLine($"PASS {mode} source-linked CopyTo {phase}: double Dispose rejects new calls; admitted data and tail preserved; exactly one physical result release");
    }

    private static int Main(string[] args)
    {
        try
        {
            Require(args.Length == 2, "Expected jit|aot and query|copy");
            Check(args[0], args[1]);
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}

namespace FactsYOLO
{
    // A controlled boundary, not a detector or installed native library. The
    // actual Result and SafeHandle/lease sources above own this unmanaged memory.
    internal static unsafe class Native
    {
        private static readonly object Gate = new();
        private static ManualResetEventSlim admitted = null!, resume = null!;
        private static string pause = "";
        private static nint memory;
        private static int queries, copies, releaseCalls, physicalFrees;
        internal static int Queries => Volatile.Read(ref queries);
        internal static int Copies => Volatile.Read(ref copies);
        internal static int ReleaseCalls => Volatile.Read(ref releaseCalls);
        internal static int PhysicalFrees => Volatile.Read(ref physicalFrees);
        internal static void Configure(string phase, ManualResetEventSlim entered, ManualResetEventSlim continuation)
        {
            pause = phase; admitted = entered; resume = continuation;
        }
        internal static nint CreateResult()
        {
            memory = (nint)NativeMemory.Alloc((nuint)(2 * sizeof(FyDetection)));
            FyDetection* values = (FyDetection*)memory;
            values[0] = new() { x_min = 1, y_min = 2, x_max = 3, y_max = 4, score = .75f, class_id = 7 };
            values[1] = new() { x_min = 5, y_min = 6, x_max = 8, y_max = 9, score = .5f, class_id = 11 };
            return memory;
        }
        internal static int ResultCopy(nint result, FyDetection* destination, nuint capacity, nuint* required, FyDiagnostic* diagnostic)
        {
            string phase = destination == null ? "query" : "copy";
            if (destination == null) Interlocked.Increment(ref queries);
            else Interlocked.Increment(ref copies);
            if (phase == pause)
            {
                admitted.Set();
                Program.Require(resume.Wait(TimeSpan.FromSeconds(10)), "Native boundary resume timeout");
            }
            lock (Gate)
            {
                // Fail before dereferencing freed memory in the negative control.
                Program.Require(memory != 0 && result == memory && physicalFrees == 0, Program.PrematureRelease);
                *required = 2;
                if (destination != null)
                {
                    Program.Require(capacity >= 2, "Unexpected short test destination");
                    new ReadOnlySpan<FyDetection>((void*)memory, 2).CopyTo(new Span<FyDetection>(destination, 2));
                }
                return 0;
            }
        }
        internal static int ResultRelease(nint* slot, FyDiagnostic* diagnostic)
        {
            Interlocked.Increment(ref releaseCalls);
            lock (Gate)
            {
                if (memory == 0 || *slot != memory) return 1;
                NativeMemory.Free((void*)memory);
                memory = 0; *slot = 0;
                Interlocked.Increment(ref physicalFrees);
                return 0;
            }
        }
        internal static int ModelRelease(nint* slot, FyDiagnostic* diagnostic) => throw new NotSupportedException();
        internal static int SessionRelease(nint* slot, FyDiagnostic* diagnostic) => throw new NotSupportedException();
        internal static int ResultDescribe(nint result, FyResultInfo* output, FyDiagnostic* diagnostic) => throw new NotSupportedException();
    }
}
