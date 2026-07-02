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