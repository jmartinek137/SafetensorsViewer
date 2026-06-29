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
    [JsonConverter(typeof(DTypeConverter))]
    public torch.ScalarType DType { get; set; } = 0;

    [JsonPropertyName("shape")]
    public long[] Shape { get; set; } = Array.Empty<long>();

    [JsonPropertyName("data_offsets")]
    public long[] DataOffsets { get; set; } = Array.Empty<long>();
}