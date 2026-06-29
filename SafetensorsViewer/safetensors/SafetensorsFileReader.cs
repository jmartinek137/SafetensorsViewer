using System.IO;
using System.Text;
using System.Text.Json;

public class SafetensorsFileReader
{
    private Dictionary<string, TensorInfo> _tensorRegistry = new Dictionary<string, TensorInfo>();
    private string _currentFilePath = string.Empty;

    public Dictionary<string, TensorInfo> TensorRegistry => _tensorRegistry;

    public SafetensorsFileReader(string filePath)
    {
        _currentFilePath = filePath;
        _tensorRegistry.Clear();

        byte[] headerLengthBytes = new byte[8];
        ulong headerLength;
        byte[] headerBytes;
        using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            fs.ReadExactly(headerLengthBytes, 0, 8);
            headerLength = BitConverter.ToUInt64(headerLengthBytes, 0);

            headerBytes = new byte[headerLength];
            fs.ReadExactly(headerBytes, 0, (int)headerLength);
        }
        string jsonText = Encoding.UTF8.GetString(headerBytes);

        long binaryDataStartOffset = 8 + (long)headerLength;

        using JsonDocument doc = JsonDocument.Parse(jsonText);
        foreach (JsonProperty property in doc.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__") continue;

            TensorInfo info = JsonSerializer.Deserialize<TensorInfo>(property.Value.GetRawText())!;

            if (info.DataOffsets != null && info.DataOffsets.Length >= 2)
            {
                info.DataOffsets[0] += binaryDataStartOffset;
                info.DataOffsets[1] += binaryDataStartOffset;
            }

            _tensorRegistry.Add(property.Name, info);
        }
    }

    public string[] Keys { get { return _tensorRegistry.Keys.ToArray(); } }

    public byte[] GetTensor(string key)
    {
        if (!_tensorRegistry.ContainsKey(key))
        {
            throw new KeyNotFoundException($"Tensor with key '{key}' not found.");
        }
        TensorInfo info = _tensorRegistry[key];
        long dataLength = info.DataOffsets[1] - info.DataOffsets[0];
        byte[] tensorData = new byte[dataLength];
        using (FileStream fs = new FileStream(_currentFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            fs.Seek(info.DataOffsets[0], SeekOrigin.Begin);
            fs.ReadExactly(tensorData, 0, (int)dataLength);
        }
        return tensorData;
    }
}