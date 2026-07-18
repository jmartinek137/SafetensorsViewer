using System.Buffers.Binary;
using System.IO;
using System.Text;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Utils;

public class SafetensorsFileWriter
{
    public static void SaveTensor(string filePath, string key, torch.Tensor tensor, SafetensorsDType originalDType)
    {
        byte[] data = originalDType switch
        {
            SafetensorsDType.FP8_E4M3 => ConvertToFP8(tensor, FP8Converters.DoubleToE4M3),
            SafetensorsDType.FP8_E5M2 => ConvertToFP8(tensor, FP8Converters.DoubleToE5M2),
            _ => tensor.to_type(MapToTorchSharpType(originalDType)).bytes.ToArray()
        };

        long dataStart = 8;

        TensorInfo info = new()
        {
            DType = originalDType.ToSafetensorsString(),
            Shape = tensor.shape.Select(x => (long)x).ToArray(),
            DataOffsets = [dataStart, dataStart + data.Length]
        };

        Dictionary<string, object> header = new()
        {
            [key] = info
        };

        string json = JsonSerializer.Serialize(header);
        byte[] headerBytes = Encoding.UTF8.GetBytes(json);

        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        Span<byte> lengthBytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(lengthBytes, (ulong)headerBytes.Length);
        fs.Write(lengthBytes);
        fs.Write(headerBytes, 0, headerBytes.Length);
        fs.Write(data, 0, data.Length);
    }

    static byte[] ConvertToFP8(torch.Tensor tensor, Func<double, byte> converter)
    {
        torch.Tensor flatTensor = tensor.to_type(torch.ScalarType.Float64).flatten();
        long count = flatTensor.numel();
        byte[] result = new byte[count];
        using (TensorAccessor<double> accessor = flatTensor.data<double>())
        {
            for (int i = 0; i < count; i++)
            {
                result[i] = converter(accessor[i]);
            }
        }
        return result;
    }

    static torch.ScalarType MapToTorchSharpType(SafetensorsDType dtype)
    {
        return dtype switch
        {
            SafetensorsDType.F64 => torch.ScalarType.Float64,
            SafetensorsDType.F32 => torch.ScalarType.Float32,
            SafetensorsDType.F16 => torch.ScalarType.Float16,
            SafetensorsDType.BF16 => torch.ScalarType.BFloat16,
            SafetensorsDType.I64 => torch.ScalarType.Int64,
            SafetensorsDType.I32 => torch.ScalarType.Int32,
            SafetensorsDType.I16 => torch.ScalarType.Int16,
            SafetensorsDType.I8 => torch.ScalarType.Int8,
            SafetensorsDType.U8 => torch.ScalarType.Byte,
            SafetensorsDType.BOOL => torch.ScalarType.Bool,
            SafetensorsDType.Complex64 => torch.ScalarType.ComplexFloat64,
            SafetensorsDType.Complex32 => torch.ScalarType.ComplexFloat32,
            _ => throw new NotSupportedException($"Data type '{dtype}' is not supported for writing.")
        };
    }
}
