using System.IO;
using System.Text;
using System.Text.Json;

public class SafetensorsLoader
{
    private Dictionary<string, TensorInfo> _tensorRegistry = new Dictionary<string, TensorInfo>();
    private string _currentFilePath = string.Empty;

    public Dictionary<string, TensorInfo> TensorRegistry => _tensorRegistry;

    public async Task<List<string>> LoadSafetensorsHeaderAsync(string filePath)
    {
        _currentFilePath = filePath;
        _tensorRegistry.Clear();

        await Task.Run(() =>
        {
            using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);

            byte[] headerLengthBytes = new byte[8];
            fs.ReadExactly(headerLengthBytes, 0, 8);
            ulong headerLength = BitConverter.ToUInt64(headerLengthBytes, 0);

            byte[] headerBytes = new byte[headerLength];
            fs.ReadExactly(headerBytes, 0, (int)headerLength);
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
        });

        return _tensorRegistry.Keys.OrderBy(k => k).ToList();
    }
}