// SPDX-License-Identifier: Apache-2.0
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using FactsYOLO;

internal static unsafe class Program
{
    internal const string LostBusy = "Concurrent Detect did not preserve native BUSY status 6";
    internal const string Serialized = "Concurrent Detect did not complete while the first call remained paused";
    internal const string Retried = "Concurrent Detect retried native BUSY";
    private static readonly byte[] Pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    internal static void Require(bool condition, string text)
    {
        if (!condition) throw new InvalidOperationException(text);
    }
    private static Result Detect(Session session) => session.Detect(Pixels, 2, 2, 6, PixelFormat.Rgb8);
    private static Task<T> Worker<T>(Func<T> operation) => Task.Factory.StartNew(operation,
        CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
    private static void CheckDiagnostic(FactsYoloException error)
    {
        Require(error.Status == 6 && error.NativeDiagnostic == Native.BusyText &&
            error.RequiredDiagnosticBytes == Native.RequiredBytes &&
            error.IsDiagnosticTruncated == Native.Truncated &&
            error.Message == $"FactsYOLO status 6: {Native.BusyText}", "BUSY diagnostic was changed or lost");
    }
    private static void Check(string mode, bool truncated)
    {
        Require(RuntimeFeature.IsDynamicCodeSupported == (mode == "jit"), "Wrong execution mode");
        using ManualResetEventSlim admitted = new(false), resume = new(false), secondStarted = new(false);
        Native.Configure(admitted, resume, truncated);
        using Session session = Session.Create(17, new SessionOptions());
        Task<Result> first = Worker(() => Detect(session));
        Task<Exception?>? second = null;
        Exception? failure = null;
        FactsYoloException? busy = null;
        Result? firstResult = null;
        try
        {
            Require(admitted.Wait(TimeSpan.FromSeconds(10)), "First Detect did not reach the native boundary");
            second = Worker<Exception?>(() =>
            {
                secondStarted.Set();
                try { using Result unexpected = Detect(session); return null; }
                catch (Exception error) { return error; }
            });
            Require(secondStarted.Wait(TimeSpan.FromSeconds(10)), "Concurrent worker did not start");
            // Completion must precede resuming the first call. This watchdog
            // detects serialization; it is not a latency benchmark.
            Require(second.Wait(TimeSpan.FromSeconds(10)), Serialized);
            Require(!first.IsCompleted && !resume.IsSet, "First call escaped the controlled overlap");
            Require(Native.DetectCalls == 2 && Native.BusyCalls == 1, Retried);
            busy = second.GetAwaiter().GetResult() as FactsYoloException;
            Require(busy is { Status: 6 }, LostBusy);
            CheckDiagnostic(busy!);
            Require(Native.ResultAllocations == 0 && Native.ResultReleases == 0, "BUSY allocated or released a result");
        }
        catch (Exception error) { failure = error; }
        finally
        {
            resume.Set();
            // Mutants also join both dedicated workers before disposing the
            // session or barriers. The boundary rejects repeated BUSY calls.
            try { firstResult = first.GetAwaiter().GetResult(); }
            catch (Exception error) { failure ??= error; }
            if (second is not null)
            {
                try { _ = second.GetAwaiter().GetResult(); }
                catch (Exception error) { failure ??= error; }
            }
        }
        using (firstResult)
        {
            if (failure is not null) throw failure;
            Require(firstResult is not null, "First Detect lost its result");
            Require(firstResult!.Copy().SequenceEqual(new[] { Native.Expected(7) }), "First result is unusable after BUSY");
            Require(firstResult.Info().Count == 1, "First result metadata is unusable");
            Require(session.Info() == new SessionInfo(1, 1, 0, 1, 0), "Session is unusable after BUSY");
            using (Result later = Detect(session))
                Require(later.Copy().SequenceEqual(new[] { Native.Expected(11) }), "Subsequent Detect failed after BUSY");
            Require(firstResult.Copy().SequenceEqual(new[] { Native.Expected(7) }), "Later Detect changed the first result");
            CheckDiagnostic(busy!);
            Require(Native.DetectCalls == 3 && Native.BusyCalls == 1, Retried);
        }
        session.Dispose(); session.Dispose();
        Require(Native.ResultAllocations == 2 && Native.ResultReleases == 2 && Native.SessionReleases == 1,
            "Successful results or session were leaked or released twice");
        Console.WriteLine($"PASS {mode} source-linked Session BUSY {(truncated ? "truncated" : "full")}: status 6 and diagnostic preserved without retry; first result and later Detect usable");
    }
    private static int Main(string[] args)
    {
        try
        {
            Require(args.Length == 2 && args[0] is "jit" or "aot" && args[1] is "full" or "truncated",
                "Expected jit|aot and full|truncated");
            Check(args[0], args[1] == "truncated");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}

namespace FactsYOLO
{
    // Controlled native boundary, not an inference backend. Production Session,
    // diagnostics and Result/lease sources manage these test-owned handles.
    internal static unsafe class Native
    {
        private static readonly object Gate = new();
        private static readonly Dictionary<nint, uint> Results = new();
        private static ManualResetEventSlim admitted = null!, resume = null!;
        private static nint sessionMemory;
        private static int active, detectCalls, busyCalls, successfulCalls, allocations, resultReleases, sessionReleases;
        internal static int DetectCalls => Volatile.Read(ref detectCalls);
        internal static int BusyCalls => Volatile.Read(ref busyCalls);
        internal static int ResultAllocations => Volatile.Read(ref allocations);
        internal static int ResultReleases => Volatile.Read(ref resultReleases);
        internal static int SessionReleases => Volatile.Read(ref sessionReleases);
        internal static string BusyText { get; private set; } = "";
        internal static nuint RequiredBytes { get; private set; }
        internal static bool Truncated { get; private set; }
        internal static Detection Expected(uint id) => new(1, 2, 3, 4, .75f, id);
        internal static void Configure(ManualResetEventSlim entered, ManualResetEventSlim continuation, bool truncated)
        {
            admitted = entered; resume = continuation; Truncated = truncated;
            const string prefix = "BUSY: session occupied — Ω ";
            BusyText = truncated ? prefix + new string('b', Diagnostics.Capacity - 1 - Encoding.UTF8.GetByteCount(prefix)) : prefix;
            RequiredBytes = truncated ? (nuint)(Diagnostics.Capacity + 17) : (nuint)Encoding.UTF8.GetByteCount(BusyText);
        }
        internal static int SessionCreate(uint abi, nint model, FySessionOptions* options, nint* output, FyDiagnostic* diagnostic)
        {
            Program.Require(abi == 1 && model == 17 && *output == 0, "Unexpected session creation");
            sessionMemory = (nint)NativeMemory.Alloc(1); *output = sessionMemory;
            return 0;
        }
        internal static int Detect(nint session, FyImage* image, FyDetectOptions* options, nint* output, FyDiagnostic* diagnostic)
        {
            Program.Require(session == sessionMemory && session != 0 && *output == 0, "Invalid session or result slot");
            Program.Require(image->width == 2 && image->height == 2 && image->row_stride == 6 &&
                image->pixel_format == 1 && image->byte_length == 12 && image->data[0] == 10 && image->data[11] == 120,
                "Borrowed image changed");
            Program.Require(diagnostic->capacity == Diagnostics.Capacity, "Unexpected diagnostic buffer");
            int call = Interlocked.Increment(ref detectCalls);
            if (call != 1 && Volatile.Read(ref active) != 0)
            {
                // A retrying mutant must fail promptly, including an unbounded
                // retry loop, so cleanup can always resume and join both calls.
                Program.Require(Interlocked.Increment(ref busyCalls) == 1, Program.Retried);
                Program.Require(diagnostic->required_bytes == 0 && diagnostic->data[0] == 0, "BUSY diagnostic buffer was not fresh");
                byte[] bytes = Encoding.UTF8.GetBytes(BusyText);
                bytes.CopyTo(new Span<byte>(diagnostic->data, bytes.Length));
                diagnostic->data[bytes.Length] = 0; diagnostic->required_bytes = RequiredBytes;
                return 6;
            }
            Program.Require(diagnostic->required_bytes == 0 && diagnostic->data[0] == 0, "Successful call reused another diagnostic");
            if (call == 1)
            {
                Volatile.Write(ref active, 1); admitted.Set();
                Program.Require(resume.Wait(TimeSpan.FromSeconds(30)), "First native call resume watchdog expired");
            }
            try
            {
                lock (Gate)
                {
                    uint id = ++successfulCalls == 1 ? 7u : 11u;
                    nint memory = (nint)NativeMemory.Alloc((nuint)sizeof(FyDetection));
                    *(FyDetection*)memory = new() { x_min = 1, y_min = 2, x_max = 3, y_max = 4, score = .75f, class_id = id };
                    Results.Add(memory, id); Interlocked.Increment(ref allocations); *output = memory;
                }
                return 0;
            }
            finally { if (call == 1) Volatile.Write(ref active, 0); }
        }
        internal static int SessionDescribe(nint session, FySessionInfo* output, FyDiagnostic* diagnostic)
        {
            Program.Require(session == sessionMemory && session != 0, "Session already released");
            output->backend_id = 1; output->device_kind = 1; output->precision = 1;
            return 0;
        }
        internal static int ResultDescribe(nint result, FyResultInfo* output, FyDiagnostic* diagnostic)
        {
            lock (Gate)
            {
                Program.Require(Results.ContainsKey(result), "Result already released");
                output->source_width = 2; output->source_height = 2; output->count = 1;
                output->backend_id = 1; output->device_kind = 1; output->precision = 1;
                return 0;
            }
        }
        internal static int ResultCopy(nint result, FyDetection* output, nuint capacity, nuint* required, FyDiagnostic* diagnostic)
        {
            lock (Gate)
            {
                Program.Require(Results.ContainsKey(result), "Result already released");
                *required = 1;
                if (output != null)
                {
                    Program.Require(capacity >= 1, "Unexpected short result destination");
                    *output = *(FyDetection*)result;
                }
                return 0;
            }
        }
        internal static int ResultRelease(nint* slot, FyDiagnostic* diagnostic)
        {
            lock (Gate)
            {
                Program.Require(Results.Remove(*slot), "Result released twice");
                NativeMemory.Free((void*)*slot); *slot = 0; Interlocked.Increment(ref resultReleases);
                return 0;
            }
        }
        internal static int SessionRelease(nint* slot, FyDiagnostic* diagnostic)
        {
            Program.Require(*slot == sessionMemory && sessionMemory != 0 && Volatile.Read(ref active) == 0,
                "Session released during an operation or twice");
            NativeMemory.Free((void*)*slot); sessionMemory = 0; *slot = 0;
            Interlocked.Increment(ref sessionReleases);
            return 0;
        }
        internal static int ModelRelease(nint* slot, FyDiagnostic* diagnostic) => throw new NotSupportedException();
    }
}
