using System.Collections.Generic;
using System.Text.Json.Serialization;

public class SafetensorHeader
{
    [JsonExtensionData]
    public Dictionary<string, System.Text.Json.JsonElement>? Tensors { get; set; }
}

public class TensorInfo
{
    [JsonPropertyName("dtype")]
    public string DType { get; set; } = string.Empty;

    [JsonPropertyName("shape")]
    public long[] Shape { get; set; } = Array.Empty<long>();

    [JsonPropertyName("data_offsets")]
    public long[] DataOffsets { get; set; } = Array.Empty<long>();
}