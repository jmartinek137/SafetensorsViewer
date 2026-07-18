using System.Buffers.Binary;
using System.IO;
using System.Text;
using System.Text.Json;
using TorchSharp;

public class SafetensorsFileWriter
{
    public static void SaveTensor(string filePath, string key, torch.Tensor tensor)
    {
        // Safetensors file layout:
        // [8 bytes: header length as little-endian ulong]
        // [header length bytes: UTF-8 JSON header]
        // [tensor binary data]

        // Flatten tensor to row-major bytes.
        byte[] data = tensor.to_type(torch.ScalarType.Float64).bytes.ToArray();
        long dataStart = 8;

        TensorInfo info = new()
        {
            DType = "F64",
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
}
