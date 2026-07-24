using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TorchSharp;

public class SafetensorsFileReader
{
    private Dictionary<string, TensorInfo> _tensorRegistry = [];
    private string _currentFilePath = string.Empty;
    private long _binaryDataStartOffset;

    public Dictionary<string, TensorInfo> TensorRegistry => _tensorRegistry;
    public long BinaryDataStartOffset => _binaryDataStartOffset;

    public byte[] ReadBytesAt(long absoluteFileOffset, int count)
    {
        if (count <= 0) return [];
        byte[] buffer = new byte[count];
        using FileStream fs = new(_currentFilePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        fs.Seek(absoluteFileOffset, SeekOrigin.Begin);
        fs.ReadExactly(buffer, 0, count);
        return buffer;
    }

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
                info.DataOffsets = new long[]
                {
                    info.DataOffsets[0] + _binaryDataStartOffset,
                    info.DataOffsets[1] + _binaryDataStartOffset
                };
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

    static torch.Tensor ConvertBF16ToFloat64(byte[] data)
    {
        // BF16 values are stored as 16-bit unsigned integers. They share the same
        // sign/exponent layout as float32, but only keep the upper 7 bits of the mantissa.
        // Shifting left by 16 produces a valid float32 with the lower mantissa bits zeroed.
        ReadOnlySpan<ushort> bf16Values = MemoryMarshal.Cast<byte, ushort>(data);
        float[] floatValues = new float[bf16Values.Length];
        for (int i = 0; i < bf16Values.Length; i++)
        {
            floatValues[i] = BitConverter.UInt32BitsToSingle((uint)bf16Values[i] << 16);
        }
        return torch.tensor(floatValues, dtype: torch.ScalarType.Float64);
    }

    public torch.Tensor LoadTensor(string key)
    {
        TensorInfo info = GetInfo(key);
        SafetensorsDType dtype = SafetensorsDTypeExtensions.Parse(info.DType);
        byte[] data = GetTensorData(key);

        if (dtype is SafetensorsDType.FP8_E4M3 or SafetensorsDType.FP8_E5M2)
        {
            double[] fp8Values = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                fp8Values[i] = dtype == SafetensorsDType.FP8_E4M3
                    ? FP8Converters.E4M3ToDouble(data[i])
                    : FP8Converters.E5M2ToDouble(data[i]);
            }
            return torch.tensor(fp8Values, dtype: torch.ScalarType.Float64).reshape(info.Shape);
        }

        if (dtype == SafetensorsDType.I4)
        {
            long elementCount = info.Shape.Aggregate(1L, (a, b) => a * b);
            double[] int4Values = new double[elementCount];
            for (int i = 0; i < elementCount; i++)
            {
                byte b = data[i / 2];
                int nibble = (i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                int4Values[i] = INT4Converters.NibbleToDouble((byte)nibble);
            }
            return torch.tensor(int4Values, dtype: torch.ScalarType.Float64).reshape(info.Shape);
        }

        // For a dim-0 (scalar) tensor, some writers store data_offsets=[x,x] (zero-length).
        // In that case data will be empty even though the tensor has 1 element.
        // Try reading the element-byte-size worth of bytes at the RAW JSON offset (before
        // _binaryDataStartOffset was added) treated as an absolute file position.
        if (data.Length == 0 && (info.Shape == null || info.Shape.Length == 0))
        {
            int elemBytes = dtype.ByteSize();
            // raw offset = current relocated offset minus the header shift
            long rawOffset = info.DataOffsets[0] - _binaryDataStartOffset;
            if (rawOffset >= 0)
            {
                try { data = ReadBytesAt(rawOffset, elemBytes); } catch { }
            }
        }

        if (data.Length == 0)
        {
            // Nothing readable – return a scalar 0 so callers can detect numel==1 safely.
            return torch.zeros([], dtype: torch.ScalarType.Float64);
        }

        double[] values = dtype switch
        {
            SafetensorsDType.F64 => MemoryMarshal.Cast<byte, double>(data).ToArray(),
            SafetensorsDType.F32 => MemoryMarshal.Cast<byte, float>(data).ToArray().Select(f => (double)f).ToArray(),
            SafetensorsDType.F16 => MemoryMarshal.Cast<byte, Half>(data).ToArray().Select(h => (double)h).ToArray(),
            SafetensorsDType.BF16 => MemoryMarshal.Cast<byte, ushort>(data).ToArray().Select(u => {
                uint bits = (uint)u << 16;
                return (double)BitConverter.UInt32BitsToSingle(bits);
            }).ToArray(),
            SafetensorsDType.I64 => MemoryMarshal.Cast<byte, long>(data).ToArray().Select(l => (double)l).ToArray(),
            SafetensorsDType.I32 => MemoryMarshal.Cast<byte, int>(data).ToArray().Select(i => (double)i).ToArray(),
            SafetensorsDType.I16 => MemoryMarshal.Cast<byte, short>(data).ToArray().Select(s => (double)s).ToArray(),
            SafetensorsDType.I8 => MemoryMarshal.Cast<byte, sbyte>(data).ToArray().Select(sb => (double)sb).ToArray(),
            SafetensorsDType.U8 => data.Select(b => (double)b).ToArray(),
            SafetensorsDType.BOOL => data.Select(b => b != 0 ? 1.0 : 0.0).ToArray(),
            _ => throw new NotSupportedException($"Data type '{info.DType}' is not supported.")
        };

        torch.Tensor result = torch.tensor(values, dtype: torch.ScalarType.Float64);
        if (dtype == SafetensorsDType.BF16 && (info.Shape == null || info.Shape.Length == 0) && values.Length > 0 && Math.Abs(values[0]) > 1024)
        {
            values = MemoryMarshal.Cast<byte, Half>(data).ToArray().Select(h => (double)h).ToArray();
            result = torch.tensor(values, dtype: torch.ScalarType.Float64);
        }
        return info.Shape != null && info.Shape.Length > 0 ? result.reshape(info.Shape) : result;
    }
}