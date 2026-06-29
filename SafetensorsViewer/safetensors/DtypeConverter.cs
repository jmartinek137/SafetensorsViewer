using System.Text.Json;
using System.Text.Json.Serialization;
using TorchSharp;

public class DTypeConverter : JsonConverter<torch.ScalarType>
{
    public override torch.ScalarType Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        string? value = reader.GetString()?.ToLower();

        return value switch
        {
            "q32" or "qint32" => torch.ScalarType.QInt32,
            "q8" or "qint8" => torch.ScalarType.QInt8,
            "qu8" or "quint8" => torch.ScalarType.QUInt8,
            "f64" or "float64" => torch.ScalarType.Float64,
            "f32" or "float32" => torch.ScalarType.Float32,
            "f16" or "float16" => torch.ScalarType.Float16,
            "bf16" or "bfloat16" => torch.ScalarType.BFloat16,
            "i32" or "int32" => torch.ScalarType.Int32,
            "i64" or "int64" => torch.ScalarType.Int64,
            "i16" or "int16" => torch.ScalarType.Int16,
            "i8" or "int8" => torch.ScalarType.Int8,
            "u8" or "uint8" => torch.ScalarType.Byte,
            "bool" => torch.ScalarType.Bool,
            "complex32" => torch.ScalarType.ComplexFloat32,
            "complex64" => torch.ScalarType.ComplexFloat64,
            _ => throw new JsonException($"Data type '{value}' is not supported in torchsharp.")
        };
    }

    public override void Write(Utf8JsonWriter writer, torch.ScalarType value, JsonSerializerOptions options)
    {
        // Pokud byste někdy chtěl safetensors ukládat (Write), převede enum zpět na text
        string strValue = value switch
        {
            torch.ScalarType.Float32 => "F32",
            torch.ScalarType.Float16 => "F16",
            torch.ScalarType.BFloat16 => "BF16",
            torch.ScalarType.Int32 => "I32",
            torch.ScalarType.Int64 => "I64",
            torch.ScalarType.Int16 => "I16",
            torch.ScalarType.Int8 => "I8",
            torch.ScalarType.Byte => "U8",
            torch.ScalarType.Bool => "BOOL",
            torch.ScalarType.ComplexFloat32 => "COMPLEX32",
            torch.ScalarType.ComplexFloat64 => "COMPLEX64",
            torch.ScalarType.Float64 => "F64",
            torch.ScalarType.QInt32 => "Q32",
            torch.ScalarType.QInt8 => "Q8",
            torch.ScalarType.QUInt8 => "QU8",
            _ => value.ToString()
        };
        writer.WriteStringValue(strValue);
    }
}