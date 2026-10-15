// SPDX-License-Identifier: Apache-2.0
using System.Globalization;
using System.Security.Cryptography;
using FactsYOLO;

static void Require(bool value, string message)
{
    if (!value) throw new InvalidOperationException(message);
}
static T Throws<T>(Action action) where T : Exception
{
    try { action(); }
    catch (T error) { return error; }
    throw new InvalidOperationException($"Expected {typeof(T).Name}");
}
static FactsYoloException Error(int status, Action action)
{
    FactsYoloException error = Throws<FactsYoloException>(action);
    Require(error.Status == status, $"Wrong status: {error}");
    Require(error.NativeDiagnostic.Length > 0 && error.NativeDiagnostic.Length < 1024,
        "Missing or unbounded diagnostic");
    Require(error.RequiredDiagnosticBytes > 0, "Missing diagnostic byte count");
    return error;
}
static bool Close(float a, float b) => float.IsFinite(a) && float.IsFinite(b) &&
    MathF.Abs(a - b) <= .0001f + .0001f * MathF.Abs(b);
static void Compare(Detection[] actual, string file)
{
    string[] fields = File.ReadAllText(file).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    int count = int.Parse(fields[0], CultureInfo.InvariantCulture);
    Require(count is 0 or 1 && actual.Length == count, "Detection count differs");
    if (count == 0) { Require(fields.Length == 1, "Invalid empty reference"); return; }
    Require(fields.Length == 7, "Invalid reference");
    float[] expected = fields.Skip(2).Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
    Detection a = actual[0];
    Require(a.ClassId == uint.Parse(fields[1], CultureInfo.InvariantCulture) &&
        Close(a.Score, expected[0]) && Close(a.XMin, expected[1]) && Close(a.YMin, expected[2]) &&
        Close(a.XMax, expected[3]) && Close(a.YMax, expected[4]), "Prediction differs");
}

try
{
    RuntimeInfo runtime = NativeRuntime.Query();
    Console.WriteLine($"RUNTIME rid={runtime.RuntimeIdentifier} abi={runtime.AbiMajor}.{runtime.AbiMinor} build={runtime.BuildId} path={runtime.NativeLibraryPath}");
    if (args.Length == 1 && args[0] == "probe") return 0;
    Require(args.Length == 4, "Expected model, cases, native build ID, native SHA256");
    Require(runtime.AbiMajor == 1 && runtime.BuildId == args[2], "ABI/source differs");
    string hash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(runtime.NativeLibraryPath))).ToLowerInvariant();
    Require(hash == args[3], "Loaded native hash differs from package");
    Require(runtime.Capabilities.Count == 1 && runtime.Capabilities[0].BackendId == 1 &&
        runtime.Capabilities[0].Flags == 3, "Not the real non-fixture TD1 backend");
    byte[] artifact = File.ReadAllBytes(args[0]);
    byte[] corrupt = (byte[])artifact.Clone(); corrupt[^1] ^= 1;
    Error(4, () => { using Model unexpected = Model.Open(corrupt); });
    Error(4, () => { using Model unexpected = Model.Open(artifact.AsSpan(0, 100)); });
    Error(7, () => { using Model unexpected = Model.Open(artifact, new ModelOptions { MaxArtifactBytes = 100 }); });
    Error(7, () => { using Model unexpected = Model.Open(artifact, new ModelOptions { MaxParserSteps = 100 }); });
    for (int index = 0; index < 6; index++)
    {
        byte[] pixels = File.ReadAllBytes(Path.Combine(args[1], $"sample-{index}.rgb"));
        string expected = Path.Combine(args[1], $"expected-{index}.txt");
        using Model model = Model.Open(artifact);
        Error(3, () => { using Session unexpected = model.CreateSession(new SessionOptions { BackendId = 999 }); });
        using Session session = model.CreateSession();
        model.Dispose(); model.Dispose();
        Throws<ObjectDisposedException>(() => model.CreateSession());
        SessionInfo info = session.Info();
        Require(info.BackendId == 1 && info.DeviceKind == 1 && info.Precision == 1 && info.ExecutionFlags == 0, "Wrong session backend");
        Error(1, () => { using Result unexpected = session.Detect(pixels, 63, 64, 192, PixelFormat.Rgb8); });
        Error(1, () => { using Result unexpected = session.Detect(pixels, 64, 64, 192, PixelFormat.Rgb8, new DetectOptions { ScoreThreshold = float.NaN }); });
        Error(7, () => { using Result unexpected = session.Detect(pixels, 64, 64, 192, PixelFormat.Rgb8, new DetectOptions { MaxDetections = 17 }); });
        using Result result = session.Detect(pixels, 64, 64, 192, PixelFormat.Rgb8);
        Compare(result.Copy(), expected);
        Detection[] copied = new Detection[result.Copy().Length]; result.CopyTo(copied); Compare(copied, expected);
        if (copied.Length > 0)
        {
            FactsYoloException shortage = Error(8, () => result.CopyTo(Span<Detection>.Empty));
            Require(shortage.NativeDiagnostic == "destination capacity is too small" &&
                shortage.RequiredDiagnosticBytes == 33, "Empty copy lost the original native diagnostic");
        }
        foreach (PixelFormat format in new[] { PixelFormat.Bgr8, PixelFormat.Rgba8 })
        {
            int channels = format == PixelFormat.Rgba8 ? 4 : 3, stride = 64 * channels + 7;
            byte[] alternate = Enumerable.Repeat((byte)199, stride * 64).ToArray();
            for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++)
            {
                int p = (y * 64 + x) * 3, q = y * stride + x * channels;
                alternate[q] = pixels[p + (format == PixelFormat.Bgr8 ? 2 : 0)];
                alternate[q + 1] = pixels[p + 1];
                alternate[q + 2] = pixels[p + (format == PixelFormat.Bgr8 ? 0 : 2)];
            }
            using Result converted = session.Detect(alternate, 64, 64, (nuint)stride, format);
            Array.Clear(alternate); Compare(converted.Copy(), expected);
        }
        using (Result filtered = session.Detect(pixels, 64, 64, 192, PixelFormat.Rgb8, new DetectOptions { ScoreThreshold = 1 }))
            Require(filtered.Copy().Length == result.Copy().Count(d => d.Score >= 1), "Threshold differs");
        session.Dispose(); session.Dispose(); Array.Clear(pixels); GC.Collect();
        Throws<ObjectDisposedException>(() => session.Info());
        Throws<ObjectDisposedException>(() => session.Detect(pixels, 64, 64, 192, PixelFormat.Rgb8));
        Compare(result.Copy(), expected);
        ResultInfo ri = result.Info();
        Require(ri.SourceWidth == 64 && ri.SourceHeight == 64 && ri.Count == (nuint)copied.Length &&
            ri.BackendId == 1 && ri.ModelId == Convert.ToHexString(artifact.AsSpan(24, 32)).ToLowerInvariant(), "Owned metadata differs");
        result.Dispose(); result.Dispose();
        Throws<ObjectDisposedException>(() => result.Copy());
        Throws<ObjectDisposedException>(() => result.Info());
        Throws<ObjectDisposedException>(() => result.CopyTo(new Detection[1]));
        Console.WriteLine($"PASS public SDK sample {index}: formats/errors/independent lifetimes");
    }
    byte[] racePixels = File.ReadAllBytes(Path.Combine(args[1], "sample-0.rgb"));
    for (int iteration = 0; iteration < 32; iteration++)
    {
        using Model model = Model.Open(artifact);
        using Session session = model.CreateSession();
        Task call = Task.Run(() => {
            try { using Result result = session.Detect(racePixels, 64, 64, 192, PixelFormat.Rgb8); _ = result.Copy(); }
            catch (ObjectDisposedException) { }
        });
        session.Dispose(); model.Dispose(); call.GetAwaiter().GetResult();
    }
    Console.WriteLine("PASS public SDK: 32 operation/dispose races and repeated ownership cycles");
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine(error.ToString());
    return 1;
}
