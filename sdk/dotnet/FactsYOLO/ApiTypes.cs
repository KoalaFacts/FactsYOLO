// SPDX-License-Identifier: Apache-2.0
namespace FactsYOLO;

/// <summary>Byte order in the borrowed, synchronous image view.</summary>
public enum PixelFormat : uint { Rgb8 = 1, Bgr8 = 2, Rgba8 = 3 }

public sealed record ModelOptions
{
    public nuint MaxArtifactBytes { get; init; } = 1024 * 1024;
    public nuint MaxParserSteps { get; init; } = 65536;
}
public sealed record SessionOptions
{
    public uint BackendId { get; init; } = 1;
    public uint DeviceKind { get; init; } = 1;
    public uint DeviceIndex { get; init; }
    public uint Precision { get; init; } = 1;
    public uint FallbackPolicy { get; init; }
    public uint CpuThreads { get; init; } = 1;
    public nuint MaxImageBytes { get; init; } = 65536;
    public nuint MaxPixels { get; init; } = 4096;
    public nuint MaxDetections { get; init; } = 16;
}
public sealed record DetectOptions
{
    public float ScoreThreshold { get; init; } = .5f;
    public nuint MaxDetections { get; init; } = 16;
}
/// <summary>Original-image pixel edge coordinates; maximum edges are exclusive.</summary>
public readonly record struct Detection(float XMin, float YMin, float XMax, float YMax, float Score, uint ClassId);
public sealed record Capability(uint BackendId, uint DeviceKind, uint DeviceIndex, uint Flags, uint PrecisionMask);
public sealed record SessionInfo(uint BackendId, uint DeviceKind, uint DeviceIndex, uint Precision, uint ExecutionFlags);
public sealed record ResultInfo(uint SourceWidth, uint SourceHeight, nuint Count, uint ResultFlags,
    uint BackendId, uint DeviceKind, uint DeviceIndex, uint Precision, uint ExecutionFlags, string ModelId);
public sealed record RuntimeInfo(uint AbiMajor, uint AbiMinor, string BuildId, string RuntimeIdentifier,
    string NativeLibraryPath, IReadOnlyList<Capability> Capabilities);
