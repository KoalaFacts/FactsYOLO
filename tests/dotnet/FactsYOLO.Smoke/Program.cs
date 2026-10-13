// SPDX-License-Identifier: Apache-2.0
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
namespace FactsYOLO.Smoke;

internal static unsafe class Program
{
    private static readonly byte[] Pixels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
    private static T Throws<T>(Action action) where T : Exception
    {
        try { action(); }
        catch (T exception) { return exception; }
        throw new InvalidOperationException($"Expected {typeof(T).Name}.");
    }
    private static void Error(int expected, Action action) =>
        Require(Throws<NativeFailure>(action).Status == expected, "Wrong native status.");
    private static nuint[] Counts()
    {
        nuint[] values = new nuint[4];
        fixed (nuint* pointer = values) Status.Check(Native.ResourceCounts(pointer, 4));
        return values;
    }
    private static void ZeroCounts() => Require(Counts().All(value => value == 0), "Live native resources remain.");
    internal static FySessionOptions Options() => new()
    {
        struct_size = (uint)sizeof(FySessionOptions), struct_version = 1,
        backend_id = 0xffff0001, device_kind = 1, precision = 1, cpu_threads = 1,
        max_image_bytes = 1024, max_pixels = 64, max_detections = 16
    };
    private static void Canonical(FyDetection[] values)
    {
        Require(values.Length == 3, "Expected three fixture detections.");
        Require(values[0].class_id == 0 && values[0].score == .9f &&
            values[0].x_min == 0 && values[0].y_min == 0 && values[0].x_max == 2 && values[0].y_max == 2, "First box differs.");
        Require(values[1].class_id == 1 && values[1].score == .9f &&
            values[1].x_min == 0 && values[1].y_min == 0 && values[1].x_max == 1 && values[1].y_max == 1, "Second box differs.");
        Require(values[2].class_id == 0 && values[2].score == .5f &&
            values[2].x_min == 1 && values[2].y_min == 1 && values[2].x_max == 2 && values[2].y_max == 2, "Third box differs.");
    }
    private static void Statuses()
    {
        Status.Check(0);
        foreach (int value in Enumerable.Range(1, 12).Concat([-1, int.MinValue, int.MaxValue]))
            Error(value, () => Status.Check(value));
    }
    private static void CounterContract()
    {
        ZeroCounts();
        Require(Native.ResourceCounts(null, 4) == 1, "Null counter buffer accepted.");
        nuint[] sentinel = [77, 77, 77, 77];
        fixed (nuint* pointer = sentinel)
            Require(Native.ResourceCounts(pointer, 3) == 1, "Short counter buffer accepted.");
        Require(sentinel.All(value => value == 77), "Failed counter copy wrote partially.");
    }
    private static void Lifetimes()
    {
        ZeroCounts();
        using Model model = Model.Fixture(1);
        using Session session = model.CreateSession(Options());
        model.Dispose(); model.Dispose();
        Require(Counts().SequenceEqual(new nuint[] { 0, 1, 1, 0 }), "Session did not retain model state.");
        Require(session.Info().precision == 1, "Actual precision differs.");
        using Result result = session.Detect(Pixels);
        session.Dispose(); session.Dispose();
        Require(Counts().SequenceEqual(new nuint[] { 0, 0, 0, 1 }), "Result is not independently owned.");
        Canonical(result.Copy());
        FyResultInfo info = result.Info();
        Require(info.source_width == 2 && info.source_height == 2 && info.count == 3 &&
            info.backend_id == 0xffff0001 && info.model_id[0] == 1, "Result metadata differs.");
        result.Dispose(); result.Dispose(); ZeroCounts();
        Throws<ObjectDisposedException>(() => session.Detect(Pixels));
        Throws<ObjectDisposedException>(() => result.Copy());
        Throws<ObjectDisposedException>(() => model.CreateSession(Options()));
    }
    private static void Copies()
    {
        using Model model = Model.Fixture(1);
        using Session session = model.CreateSession(Options());
        using Result result = session.Detect(Pixels);
        FyDetection[] small = [new() { score = 77 }, new() { score = 77 }];
        Error(8, () => result.CopyTo(small));
        Require(small.All(value => value.score == 77), "Short destination was changed.");
        FyDetection[] copy = result.Copy(); copy[0].score = 42;
        Canonical(result.Copy());
        using Result capped = session.Detect(Pixels, cap: 2);
        Require(capped.Copy().Length == 2 && capped.Info().result_flags == 1, "Truncation differs.");
        Error(1, () => session.Detect(Pixels, threshold: float.NaN));
        Error(7, () => session.Detect(Pixels, cap: 17));
        Canonical(result.Copy());
    }
    private static void Faults()
    {
        foreach ((uint fixture, int error) in new (uint, int)[] { (3, 9), (5, 12), (6, 12) })
        {
            using Model model = Model.Fixture(fixture);
            using Session session = model.CreateSession(Options());
            Error(error, () => session.Detect(Pixels));
            Error(11, () => session.Detect(Pixels));
        }
        ZeroCounts();
        using Model empty = Model.Fixture(2);
        using Session sessionEmpty = empty.CreateSession(Options());
        using Result result = sessionEmpty.Detect(Pixels);
        Require(result.Copy().Length == 0 && result.Info().count == 0, "Empty result differs.");
        result.CopyTo(Span<FyDetection>.Empty);
    }
    private static void FailedCreation()
    {
        ZeroCounts();
        using Model model = Model.Fixture(1);
        nuint[] before = Counts();
        FySessionOptions options = Options(); options.device_kind = 2;
        Error(5, () => model.CreateSession(options));
        Require(Counts().SequenceEqual(before), "Failed creation leaked native resources.");
        Error(4, () => Model.Open(Pixels));
        Require(Counts().SequenceEqual(before), "Unsupported model created a handle.");
    }
    private static void DisposeRace()
    {
        ZeroCounts();
        using Model model = Model.Fixture(4);
        using Session session = model.CreateSession(Options());
        model.Dispose();
        Task<Result> work = Task.Run(() => session.Detect(Pixels));
        int running = session.WaitRunningForTest(5000);
        // Failure terminates this test process: never attempt cleanup through a
        // possibly stale pointer. All normal assertions occur after safe resume.
        if (running != 0) Environment.FailFast("Fixture did not enter the native call.");
        bool busy = false;
        try { using Result unexpected = session.Detect(Pixels); }
        catch (NativeFailure error) { busy = error.Status == 6; }
        session.Dispose();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        nuint[] live = Counts();
        if (live[1] != 1 || live[2] != 1 || session.ReleaseCalls != 0)
            Environment.FailFast("Native resource released while its call was in flight.");
        bool closed = false;
        try { using Result unexpected = session.Detect(Pixels); }
        catch (ObjectDisposedException) { closed = true; }
        // Acquire only after proving Detect's own lease retained the session.
        // Resume can let Detect finish before Resume itself returns. This
        // independent lease protects control, or safely rejects a timed-out call.
        using Lease control = session.AcquireControlForTest();
        if (Native.Resume(control.Pointer) != 0 || !work.Wait(TimeSpan.FromSeconds(10)))
            Environment.FailFast("Blocked native call could not be safely joined.");
        using Result result = work.GetAwaiter().GetResult();
        Require(session.ReleaseCalls == 0,
            "Fixture control lifetime ended before the control operation completed.");
        Require(Counts().SequenceEqual(new nuint[] { 0, 1, 1, 1 }),
            "Fixture control did not retain the session after Detect completed.");
        control.Dispose();
        Require(busy && closed, "BUSY or disposed admission was hidden.");
        Require(session.ReleaseCalls == 1, "Physical session release was not exactly once.");
        Throws<ObjectDisposedException>(() => { using Lease unexpected = session.AcquireControlForTest(); });
        Throws<ObjectDisposedException>(() => session.WaitRunningForTest(1));
        Require(Counts().SequenceEqual(new nuint[] { 0, 0, 0, 1 }), "Lease did not release after return.");
        Canonical(result.Copy());
    }
    private static void Cycles()
    {
        ZeroCounts();
        for (int i = 0; i < 1000; ++i)
        {
            using (Model model = Model.Fixture(1))
            using (Session session = model.CreateSession(Options()))
            using (Result result = session.Detect(Pixels)) { Canonical(result.Copy()); }
        }
        ZeroCounts();
    }
    private static void Finalization()
    {
        ZeroCounts();
        ForgetOwners();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        ZeroCounts();
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ForgetOwners()
    {
        Model model = Model.Fixture(1);
        Session session = model.CreateSession(Options());
        Result result = session.Detect(Pixels);
        Canonical(result.Copy());
        // Deliberately omit Dispose to exercise SafeHandle finalization.
    }
    private static int Main(string[] args)
    {
        if (args.Length != 4 || (args[2] != "normal" && args[2] != "fixture") ||
            (args[3] != "jit" && args[3] != "aot")) return 2;
        Console.WriteLine($"{RuntimeInformation.FrameworkDescription}; {RuntimeInformation.ProcessArchitecture}; mode={args[3]}");
        try { Native.Initialize(args[0]); }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
        bool fixture = args[2] == "fixture";
        List<(string, Action)> tests =
        [
            ("execution-mode", () => Require(RuntimeFeature.IsDynamicCodeSupported == (args[3] == "jit"), "Wrong execution mode.")),
            ("compiler-layouts", () => Layout.Verify(args[1])),
            ("nonzero-statuses", Statuses),
            ("missing-library", () => Throws<FileNotFoundException>(() => Native.ValidatePath(args[0] + ".missing"))),
            ("capability-isolation", () => {
                nuint count = 99;
                Require(Native.Capabilities(null, 0, &count, null) == 0 && count == (fixture ? 1u : 0u), "Wrong backend count.");
                Require(Native.HasExport("fy_test_model_create") == fixture, "Test factory leaked.");
                Require(Native.HasExport("fy_test_resource_counts") == fixture, "Test counters leaked or absent.");
            }),
            ("unsupported-artifact", () => Error(4, () => Model.Open(Pixels)))
        ];
        if (fixture) tests.AddRange([
            ("counter-contract", CounterContract), ("independent-lifetimes", Lifetimes),
            ("copies-and-errors", Copies), ("native-faults", Faults),
            ("failed-creation", FailedCreation), ("dispose-during-detect", DisposeRace),
            ("1000-lifecycle-cycles", Cycles), ("safehandle-finalization", Finalization)
        ]);
        int failed = 0;
        foreach ((string name, Action test) in tests)
        {
            try { test(); Console.WriteLine($"PASS {name}"); }
            catch (Exception error) { ++failed; Console.Error.WriteLine($"FAIL {name}: {error}"); }
        }
        Console.WriteLine($"{args[2]} {args[3]}: {tests.Count - failed} passed, {failed} failed");
        return failed == 0 ? 0 : 1;
    }
}
