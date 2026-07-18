using System.Collections.Generic;
using System.Text.Json.Serialization;
using TorchSharp;

public class SafetensorHeader
{
    [JsonExtensionData]
    public Dictionary<string, System.Text.Json.JsonElement>? Tensors { get; set; }
}

public class TensorInfo
{
    [JsonPropertyName("dtype")]
    public string DType { get; set; } = String.Empty;

    [JsonPropertyName("shape")]
    public long[] Shape { get; set; } = [];

    [JsonPropertyName("data_offsets")]
    public long[] DataOffsets { get; set; } = [];
}

public enum SafetensorsDType
{
    F64, F32, F16, BF16, FP8_E4M3, FP8_E5M2,
    I64, I32, I16, I8, I4,
    U8, BOOL,
    Complex64, Complex32
}

public static class SafetensorsDTypeExtensions
{
    public static SafetensorsDType Parse(string dtype)
    {
        return dtype switch
        {
            "F64" => SafetensorsDType.F64,
            "F32" => SafetensorsDType.F32,
            "F16" => SafetensorsDType.F16,
            "BF16" => SafetensorsDType.BF16,
            "FP8_E4M3" or "F8_E4M3" => SafetensorsDType.FP8_E4M3,
            "FP8_E5M2" or "F8_E5M2" => SafetensorsDType.FP8_E5M2,
            "I64" => SafetensorsDType.I64,
            "I32" => SafetensorsDType.I32,
            "I16" => SafetensorsDType.I16,
            "I8" => SafetensorsDType.I8,
            "I4" => SafetensorsDType.I4,
            "U8" => SafetensorsDType.U8,
            "BOOL" => SafetensorsDType.BOOL,
            "Complex64" => SafetensorsDType.Complex64,
            "Complex32" => SafetensorsDType.Complex32,
            _ => throw new NotSupportedException($"Data type '{dtype}' is not supported.")
        };
    }

    public static string ToSafetensorsString(this SafetensorsDType dtype)
    {
        return dtype switch
        {
            SafetensorsDType.F64 => "F64",
            SafetensorsDType.F32 => "F32",
            SafetensorsDType.F16 => "F16",
            SafetensorsDType.BF16 => "BF16",
            SafetensorsDType.FP8_E4M3 => "FP8_E4M3",
            SafetensorsDType.FP8_E5M2 => "FP8_E5M2",
            SafetensorsDType.I64 => "I64",
            SafetensorsDType.I32 => "I32",
            SafetensorsDType.I16 => "I16",
            SafetensorsDType.I8 => "I8",
            SafetensorsDType.I4 => "I4",
            SafetensorsDType.U8 => "U8",
            SafetensorsDType.BOOL => "BOOL",
            SafetensorsDType.Complex64 => "Complex64",
            SafetensorsDType.Complex32 => "Complex32",
            _ => throw new NotSupportedException($"Data type '{dtype}' is not supported.")
        };
    }

    public static int ByteSize(this SafetensorsDType dtype)
    {
        return dtype switch
        {
            SafetensorsDType.F64 or SafetensorsDType.Complex64 or SafetensorsDType.I64 => 8,
            SafetensorsDType.F32 or SafetensorsDType.I32 or SafetensorsDType.Complex32 => 4,
            SafetensorsDType.F16 or SafetensorsDType.BF16 or SafetensorsDType.I16 => 2,
            SafetensorsDType.FP8_E4M3 or SafetensorsDType.FP8_E5M2 or SafetensorsDType.I8 or SafetensorsDType.U8 or SafetensorsDType.I4 => 1,
            SafetensorsDType.BOOL => 1,
            _ => throw new NotSupportedException($"Data type '{dtype}' has no defined byte size.")
        };
    }
}