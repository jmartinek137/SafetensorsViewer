using System.Buffers.Binary;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Utils;

public class SafetensorsFileReader
{
    private Dictionary<string, TensorInfo> _tensorRegistry = new();
    private string _currentFilePath = string.Empty;
    private long _binaryDataStartOffset;

    public Dictionary<string, TensorInfo> TensorRegistry => _tensorRegistry;

    public SafetensorsFileReader(string filePath)
    {
        _currentFilePath = filePath;
        _tensorRegistry.Clear();

        byte[] headerLengthBytes = new byte[8];
        ulong headerLength;
        byte[] headerBytes;
        using (FileStream fs = new(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            fs.ReadExactly(headerLengthBytes, 0, 8);
            headerLength = BitConverter.ToUInt64(headerLengthBytes, 0);

            headerBytes = new byte[headerLength];
            fs.ReadExactly(headerBytes, 0, (int)headerLength);
        }
        string jsonText = Encoding.UTF8.GetString(headerBytes);

        _binaryDataStartOffset = 8 + (long)headerLength;

        using JsonDocument doc = JsonDocument.Parse(jsonText);
        foreach (JsonProperty property in doc.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__") continue;

            TensorInfo info = JsonSerializer.Deserialize<TensorInfo>(property.Value.GetRawText())!;

            if (info.DataOffsets != null && info.DataOffsets.Length >= 2)
            {
                info.DataOffsets[0] += _binaryDataStartOffset;
                info.DataOffsets[1] += _binaryDataStartOffset;
            }

            _tensorRegistry.Add(property.Name, info);
        }
    }

    public string[] Keys { get { return _tensorRegistry.Keys.ToArray(); } }

    public TensorInfo GetInfo(string key)
    {
        if (!_tensorRegistry.ContainsKey(key))
        {
            throw new KeyNotFoundException($"Tensor with key '{key}' not found.");
        }
        return _tensorRegistry[key];
    }

    public byte[] GetTensorData(string key)
    {
        TensorInfo info = GetInfo(key);
        long dataLength = info.DataOffsets[1] - info.DataOffsets[0];
        byte[] tensorData = new byte[dataLength];
        using (FileStream fs = new(_currentFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            fs.Seek(info.DataOffsets[0], SeekOrigin.Begin);
            fs.ReadExactly(tensorData, 0, (int)dataLength);
        }
        return tensorData;
    }

    public torch.Tensor LoadTensor(string key)
    {
        TensorInfo info = GetInfo(key);
        SafetensorsDType dtype = SafetensorsDTypeExtensions.Parse(info.DType);
        byte[] data = GetTensorData(key);

        if (dtype is SafetensorsDType.FP8_E4M3 or SafetensorsDType.FP8_E5M2)
        {
            double[] values = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                values[i] = dtype == SafetensorsDType.FP8_E4M3
                    ? FP8Converters.E4M3ToDouble(data[i])
                    : FP8Converters.E5M2ToDouble(data[i]);
            }
            return torch.tensor(values, dtype: torch.ScalarType.Float64).reshape(info.Shape);
        }

        if (dtype == SafetensorsDType.I4)
        {
            long elementCount = info.Shape.Aggregate(1L, (a, b) => a * b);
            double[] values = new double[elementCount];
            for (int i = 0; i < elementCount; i++)
            {
                byte b = data[i / 2];
                int nibble = (i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                values[i] = INT4Converters.NibbleToDouble((byte)nibble);
            }
            return torch.tensor(values, dtype: torch.ScalarType.Float64).reshape(info.Shape);
        }

        torch.ScalarType scalarType = dtype switch
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
            _ => throw new NotSupportedException($"Data type '{info.DType}' is not supported.")
        };

        torch.Tensor tensor = dtype switch
        {
            SafetensorsDType.F64 => torch.tensor(MemoryMarshal.Cast<byte, double>(data).ToArray(), dtype: scalarType),
            SafetensorsDType.F32 => torch.tensor(MemoryMarshal.Cast<byte, float>(data).ToArray(), dtype: scalarType),
            SafetensorsDType.F16 or SafetensorsDType.BF16 => torch.tensor(MemoryMarshal.Cast<byte, Half>(data).ToArray(), dtype: scalarType),
            SafetensorsDType.I64 => torch.tensor(MemoryMarshal.Cast<byte, long>(data).ToArray(), dtype: scalarType),
            SafetensorsDType.I32 => torch.tensor(MemoryMarshal.Cast<byte, int>(data).ToArray(), dtype: scalarType),
            SafetensorsDType.I16 => torch.tensor(MemoryMarshal.Cast<byte, short>(data).ToArray(), dtype: scalarType),
            SafetensorsDType.I8 => torch.tensor(MemoryMarshal.Cast<byte, sbyte>(data).ToArray(), dtype: scalarType),
            SafetensorsDType.U8 => torch.tensor(data, dtype: scalarType),
            SafetensorsDType.BOOL => torch.tensor(data.Select(b => b != 0).ToArray(), dtype: scalarType),
            SafetensorsDType.Complex64 => throw new NotSupportedException("Complex64 tensors are not supported yet."),
            SafetensorsDType.Complex32 => throw new NotSupportedException("Complex32 tensors are not supported yet."),
            _ => throw new NotSupportedException($"Data type '{info.DType}' is not supported.")
        };

        return tensor.reshape(info.Shape).to_type(torch.ScalarType.Float64);
    }
}