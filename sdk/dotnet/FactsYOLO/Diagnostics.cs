// SPDX-License-Identifier: Apache-2.0
using System.Text;
namespace FactsYOLO;

/// <summary>A primary native status and its bounded, per-call UTF-8 diagnostic.</summary>
public sealed class FactsYoloException : Exception
{
    public int Status { get; }
    public string NativeDiagnostic { get; }
    public nuint RequiredDiagnosticBytes { get; }
    public bool IsDiagnosticTruncated { get; }
    internal FactsYoloException(int status, string diagnostic, nuint required, bool truncated)
        : base($"FactsYOLO status {status}: {diagnostic}")
    {
        Status = status; NativeDiagnostic = diagnostic;
        RequiredDiagnosticBytes = required; IsDiagnosticTruncated = truncated;
    }
}

internal static unsafe class Diagnostics
{
    internal const int Capacity = 1024;
    internal static FyDiagnostic Create(byte* buffer)
    {
        new Span<byte>(buffer, Capacity).Clear();
        return new FyDiagnostic { struct_size = (uint)sizeof(FyDiagnostic), struct_version = 1,
            data = buffer, capacity = Capacity };
    }
    internal static void Check(int status, FyDiagnostic* diagnostic)
    {
        if (status == 0) return;
        ReadOnlySpan<byte> bytes = new(diagnostic->data, Capacity);
        int end = bytes.IndexOf((byte)0);
        string message = Encoding.UTF8.GetString(bytes[..(end < 0 ? Capacity : end)]);
        throw new FactsYoloException(status, message, diagnostic->required_bytes,
            diagnostic->required_bytes >= Capacity);
    }
}
