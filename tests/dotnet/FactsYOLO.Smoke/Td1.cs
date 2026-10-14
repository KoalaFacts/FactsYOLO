// SPDX-License-Identifier: Apache-2.0
using System.Globalization;
namespace FactsYOLO.Smoke;

// Acceptance consumer only: no preprocessing or model implementation in C#.
internal static unsafe class Td1
{
    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
    private static void Error(int status, Action action)
    {
        try { action(); }
        catch (NativeFailure error) { Require(error.Status == status, "Wrong native error."); return; }
        throw new InvalidOperationException("Expected native failure.");
    }
    private static float Number(string text) => float.Parse(text, CultureInfo.InvariantCulture);
    private static FyDetection[] Expected(string path)
    {
        string[] fields = File.ReadAllText(path).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        int count = int.Parse(fields[0], CultureInfo.InvariantCulture);
        Require(count is 0 or 1 && fields.Length == (count == 0 ? 1 : 7), "Invalid reference file.");
        if (count == 0) return [];
        return [new FyDetection {
            class_id = uint.Parse(fields[1], CultureInfo.InvariantCulture), score = Number(fields[2]),
            x_min = Number(fields[3]), y_min = Number(fields[4]), x_max = Number(fields[5]), y_max = Number(fields[6])
        }];
    }
    private static bool Close(float a, float b) => float.IsFinite(a) && float.IsFinite(b) && MathF.Abs(a-b) <= .0001f + .0001f*MathF.Abs(b);
    private static void Compare(FyDetection[] actual, FyDetection[] expected)
    {
        Require(actual.Length == expected.Length, "Detection count differs from trained reference.");
        for (int i=0; i<actual.Length; ++i)
        {
            FyDetection a=actual[i], b=expected[i];
            Require(a.class_id == b.class_id && Close(a.score,b.score) && Close(a.x_min,b.x_min) &&
                Close(a.y_min,b.y_min) && Close(a.x_max,b.x_max) && Close(a.y_max,b.y_max), "Detection differs from trained reference.");
        }
    }
    internal static void Run(string artifactPath, string casesDirectory)
    {
        byte[] artifact = File.ReadAllBytes(artifactPath);
        Require(artifact.Length == 51284, "Wrong trained artifact length.");
        byte[] corrupt = (byte[])artifact.Clone(); corrupt[^1] ^= 1;
        Error(4, () => { using Model unexpected = Model.Open(corrupt); });
        Error(4, () => { using Model unexpected = Model.Open(artifact.AsSpan(0,100)); });
        for (int index=0; index<6; ++index)
        {
            byte[] pixels = File.ReadAllBytes(Path.Combine(casesDirectory,$"sample-{index}.rgb"));
            Require(pixels.Length == 12288, "Wrong held-out RGB length.");
            FyDetection[] expected = Expected(Path.Combine(casesDirectory,$"expected-{index}.txt"));
            using Model model = Model.Open(artifact);
            FySessionOptions options = Program.Options(); options.backend_id=1; options.max_image_bytes=65536; options.max_pixels=4096;
            using Session session = model.CreateSession(options);
            model.Dispose();
            FySessionInfo sessionInfo=session.Info();
            Require(sessionInfo.backend_id==1 && sessionInfo.device_kind==1 && sessionInfo.precision==1 && sessionInfo.execution_flags==0, "Wrong actual backend.");
            Error(1, () => { using Result unexpected = session.Detect(pixels,width:63,height:64,stride:192); });
            using Result result = session.Detect(pixels,width:64,height:64,stride:192);
            Compare(result.Copy(),expected);
            foreach (uint format in new uint[] { 2, 3 })
            {
                int channels=format==3?4:3, stride=64*channels+7;
                byte[] alternate=Enumerable.Repeat((byte)199,stride*64).ToArray();
                for (int y=0;y<64;++y) for (int x=0;x<64;++x)
                {
                    int p=(y*64+x)*3,q=y*stride+x*channels;
                    alternate[q]=pixels[p+(format==2?2:0)];alternate[q+1]=pixels[p+1];alternate[q+2]=pixels[p+(format==2?0:2)];
                }
                using Result converted=session.Detect(alternate,width:64,height:64,stride:(nuint)stride,format:format);
                Array.Clear(alternate); Compare(converted.Copy(),expected);
            }
            using (Result filtered=session.Detect(pixels,width:64,height:64,stride:192,threshold:1.0f))
                Require(filtered.Copy().Length == expected.Count(d=>d.score>=1.0f), "Score threshold differs.");
            session.Dispose(); Array.Clear(pixels); GC.Collect();
            Compare(result.Copy(),expected);
            FyResultInfo info=result.Info();
            Require(info.source_width==64 && info.source_height==64 && info.backend_id==1 && info.count==(nuint)expected.Length, "Owned metadata differs.");
            for (int i=0;i<32;++i) Require(info.model_id[i]==artifact[24+i], "Model identity differs.");
            Console.WriteLine($"PASS TD1 held-out sample {index}: {expected.Length} detections; ownership/formats/errors");
        }
    }
}
