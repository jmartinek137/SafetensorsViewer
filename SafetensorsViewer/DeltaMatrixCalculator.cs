using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;

namespace SafetensorsViewer
{
    public static class DeltaMatrixCalculator
    {
        public enum AdapterType
        {
            None,
            LoRA,
            LoHa,
            LoKrStandard,
            LoKrFullRank
        }

        public record AdapterInfo(AdapterType Type, string BasePrefix, Dictionary<string, string> SubkeyMap, double Alpha, int Rank);

        /// <summary>
        /// Attempts to detect an adapter (LoRA, LoHa, LoKr, or Full-Rank LoKr) starting from full key or node path prefix.
        /// </summary>
        public static AdapterInfo DetectAdapter(string pathPrefix, IEnumerable<string> allKeys, SafetensorsFileReader sfr)
        {
            // If the user clicked directly on an actual tensor key in the file (like ...alpha or ...lokr_w1),
            // load that specific tensor directly without computing a combined delta matrix.
            if (allKeys.Contains(pathPrefix, StringComparer.OrdinalIgnoreCase))
            {
                return new AdapterInfo(AdapterType.None, pathPrefix, new(), 1.0, 1);
            }

            var keys = allKeys.Where(k => k.StartsWith(pathPrefix, StringComparison.OrdinalIgnoreCase)).ToList();
            if (keys.Count == 0 && !allKeys.Any(k => k.StartsWith(pathPrefix + ".", StringComparison.OrdinalIgnoreCase))) 
                return new AdapterInfo(AdapterType.None, pathPrefix, new(), 1.0, 1);

            string basePrefix = pathPrefix.TrimEnd('.');
            string[] prefixCandidates = new[] { basePrefix };

            foreach (var candidate in prefixCandidates)
            {
                var keysUnderCandidate = allKeys.Where(k => k.StartsWith(candidate + ".", StringComparison.OrdinalIgnoreCase) || k.Equals(candidate, StringComparison.OrdinalIgnoreCase)).ToList();
                if (keysUnderCandidate.Count == 0) continue;

                var subkeyMap = keysUnderCandidate.ToDictionary(
                    k => k.Substring(candidate.Length).TrimStart('.'),
                    k => k,
                    StringComparer.OrdinalIgnoreCase
                );

                double alpha = ExtractAlpha(candidate, subkeyMap, sfr);

                // 1. Check Full Rank LoKr (e.g. lokr_w1 / lokr_w1.weight, lokr_w2 / lokr_w2.weight)
                if (IsMatch(subkeyMap, out var mapFullLokr, "lokr_w1", "lokr_w2"))
                {
                    if (!HasAny(subkeyMap, "lokr_w2_a", "lokr_w2.a", "lokr_w2_a.weight", "lokr_w2.a.weight"))
                    {
                        return new AdapterInfo(AdapterType.LoKrFullRank, candidate, mapFullLokr, alpha, 1);
                    }
                }

                // 2. Check Standard LoKr (lokr_w1 + lokr_w2_a + lokr_w2_b)
                if (IsMatch(subkeyMap, out var mapLokr, "lokr_w1", "lokr_w2_a", "lokr_w2_b"))
                {
                    return new AdapterInfo(AdapterType.LoKrStandard, candidate, mapLokr, alpha, 1);
                }

                // 3. Check LoHa (hada_w1_a + hada_w1_b + hada_w2_a + hada_w2_b)
                if (IsMatch(subkeyMap, out var mapLoha, "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"))
                {
                    return new AdapterInfo(AdapterType.LoHa, candidate, mapLoha, alpha, 1);
                }

                // 4. Check LoRA (lora_down / lora_a + lora_up / lora_b)
                if (IsMatchLora(subkeyMap, out var mapLora))
                {
                    return new AdapterInfo(AdapterType.LoRA, candidate, mapLora, alpha, 1);
                }
            }

            return new AdapterInfo(AdapterType.None, pathPrefix, new(), 1.0, 1);
        }

        private static bool HasAny(Dictionary<string, string> subkeyMap, params string[] keys)
        {
            return keys.Any(k => subkeyMap.ContainsKey(k));
        }

        private static bool IsMatch(Dictionary<string, string> subkeyMap, out Dictionary<string, string> matched, params string[] requiredRoles)
        {
            matched = new Dictionary<string, string>();
            foreach (var role in requiredRoles)
            {
                string? foundKey = subkeyMap.Keys.FirstOrDefault(k =>
                    k.Equals(role, StringComparison.OrdinalIgnoreCase) ||
                    k.Equals(role + ".weight", StringComparison.OrdinalIgnoreCase) ||
                    k.Equals(role + "_weight", StringComparison.OrdinalIgnoreCase)
                );

                if (foundKey == null) return false;
                matched[role] = subkeyMap[foundKey];
            }
            return true;
        }

        private static bool IsMatchLora(Dictionary<string, string> subkeyMap, out Dictionary<string, string> matched)
        {
            matched = new Dictionary<string, string>();

            string? downKey = subkeyMap.Keys.FirstOrDefault(k =>
                k.Equals("lora_down.weight", StringComparison.OrdinalIgnoreCase) ||
                k.Equals("lora_down", StringComparison.OrdinalIgnoreCase) ||
                k.Equals("lora_a.weight", StringComparison.OrdinalIgnoreCase) ||
                k.Equals("lora_a", StringComparison.OrdinalIgnoreCase)
            );

            string? upKey = subkeyMap.Keys.FirstOrDefault(k =>
                k.Equals("lora_up.weight", StringComparison.OrdinalIgnoreCase) ||
                k.Equals("lora_up", StringComparison.OrdinalIgnoreCase) ||
                k.Equals("lora_b.weight", StringComparison.OrdinalIgnoreCase) ||
                k.Equals("lora_b", StringComparison.OrdinalIgnoreCase)
            );

            if (downKey != null && upKey != null)
            {
                matched["down"] = subkeyMap[downKey];
                matched["up"] = subkeyMap[upKey];
                return true;
            }

            return false;
        }

        private static double ExtractAlpha(string basePrefix, Dictionary<string, string> subkeyMap, SafetensorsFileReader sfr)
        {
            string? alphaKey = subkeyMap.Keys.FirstOrDefault(k =>
                k.Equals("alpha", StringComparison.OrdinalIgnoreCase) ||
                k.Equals("alpha.weight", StringComparison.OrdinalIgnoreCase)
            );

            if (alphaKey != null && subkeyMap.TryGetValue(alphaKey, out string? fullAlphaKey))
            {
                try
                {
                    using torch.Tensor alphaTensor = sfr.LoadTensor(fullAlphaKey);
                    if (alphaTensor.numel() == 1)
                    {
                        return alphaTensor.item<double>();
                    }
                }
                catch { }
            }

            return 1.0;
        }

        /// <summary>
        /// Computes the delta matrix (2D float64 tensor) for the given adapter info.
        /// </summary>
        public static torch.Tensor ComputeDelta(AdapterInfo adapter, SafetensorsFileReader sfr)
        {
            switch (adapter.Type)
            {
                case AdapterType.LoRA:
                    {
                        using var down = sfr.LoadTensor(adapter.SubkeyMap["down"]);
                        using var up = sfr.LoadTensor(adapter.SubkeyMap["up"]);

                        long rank = down.shape[0];
                        double scale = rank > 0 ? (adapter.Alpha / rank) : 1.0;

                        torch.Tensor up2D = up.ndim > 2 ? up.flatten(1) : up;
                        torch.Tensor down2D = down.ndim > 2 ? down.flatten(1) : down;

                        torch.Tensor delta = torch.matmul(up2D, down2D);
                        if (Math.Abs(scale - 1.0) > 1e-9)
                        {
                            delta = delta * scale;
                        }
                        return delta;
                    }

                case AdapterType.LoHa:
                    {
                        using var w1a = sfr.LoadTensor(adapter.SubkeyMap["hada_w1_a"]);
                        using var w1b = sfr.LoadTensor(adapter.SubkeyMap["hada_w1_b"]);
                        using var w2a = sfr.LoadTensor(adapter.SubkeyMap["hada_w2_a"]);
                        using var w2b = sfr.LoadTensor(adapter.SubkeyMap["hada_w2_b"]);

                        long rank = w1a.shape[0];
                        double scale = rank > 0 ? (adapter.Alpha / rank) : 1.0;

                        using var w1 = torch.matmul(w1b.ndim > 2 ? w1b.flatten(1) : w1b, w1a.ndim > 2 ? w1a.flatten(1) : w1a);
                        using var w2 = torch.matmul(w2b.ndim > 2 ? w2b.flatten(1) : w2b, w2a.ndim > 2 ? w2a.flatten(1) : w2a);

                        torch.Tensor delta = w1 * w2;
                        if (Math.Abs(scale - 1.0) > 1e-9)
                        {
                            delta = delta * scale;
                        }
                        return delta;
                    }

                case AdapterType.LoKrStandard:
                    {
                        using var w1 = sfr.LoadTensor(adapter.SubkeyMap["lokr_w1"]);
                        using var w2a = sfr.LoadTensor(adapter.SubkeyMap["lokr_w2_a"]);
                        using var w2b = sfr.LoadTensor(adapter.SubkeyMap["lokr_w2_b"]);

                        long rank = w2a.shape[0];
                        double scale = rank > 0 ? (adapter.Alpha / rank) : 1.0;

                        using var w2 = torch.matmul(w2b.ndim > 2 ? w2b.flatten(1) : w2b, w2a.ndim > 2 ? w2a.flatten(1) : w2a);
                        torch.Tensor w1_2D = w1.ndim > 2 ? w1.flatten(1) : w1;

                        // Kronecker product order in LyCORIS / sd-scripts: kron(w2, w1)
                        torch.Tensor delta = torch.kron(w2, w1_2D);
                        if (Math.Abs(scale - 1.0) > 1e-9)
                        {
                            delta = delta * scale;
                        }
                        return delta;
                    }

                case AdapterType.LoKrFullRank:
                    {
                        using var w1 = sfr.LoadTensor(adapter.SubkeyMap["lokr_w1"]);
                        using var w2 = sfr.LoadTensor(adapter.SubkeyMap["lokr_w2"]);

                        torch.Tensor w1_2D = w1.ndim > 2 ? w1.flatten(1) : w1;
                        torch.Tensor w2_2D = w2.ndim > 2 ? w2.flatten(1) : w2;

                        double scale = adapter.Alpha;

                        // Kronecker product order in LyCORIS / sd-scripts: kron(w2, w1)
                        torch.Tensor delta = torch.kron(w2_2D, w1_2D);
                        if (Math.Abs(scale - 1.0) > 1e-9)
                        {
                            delta = delta * scale;
                        }
                        return delta;
                    }

                default:
                    throw new InvalidOperationException("Not a valid adapter type.");
            }
        }
    }
}
