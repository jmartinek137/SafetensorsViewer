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
                return new AdapterInfo(AdapterType.None, pathPrefix, [], 1.0, 1);
            }

            var keys = allKeys.Where(k => k.StartsWith(pathPrefix, StringComparison.OrdinalIgnoreCase)).ToList();
            if (keys.Count == 0 && !allKeys.Any(k => k.StartsWith(pathPrefix + ".", StringComparison.OrdinalIgnoreCase))) 
                return new AdapterInfo(AdapterType.None, pathPrefix, [], 1.0, 1);

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

            return new AdapterInfo(AdapterType.None, pathPrefix, [], 1.0, 1);
        }

        private static bool HasAny(Dictionary<string, string> subkeyMap, params string[] keys)
        {
            return keys.Any(k => subkeyMap.ContainsKey(k));
        }

        private static bool IsMatch(Dictionary<string, string> subkeyMap, out Dictionary<string, string> matched, params string[] requiredRoles)
        {
            matched = [];
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
            matched = [];

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

        public static torch.Tensor LoadTensorWithEdits(string key, SafetensorsFileReader sfr, Dictionary<string, List<TensorEdit>>? pendingEdits)
        {
            torch.Tensor t = sfr.LoadTensor(key);
            if (pendingEdits != null && pendingEdits.TryGetValue(key, out var edits) && edits.Count > 0)
            {
                int rows = t.ndim > 0 ? (int)t.shape[0] : 1;
                int cols = t.ndim > 1 ? (int)t.shape[1] : 1;
                torch.Tensor editable = t.reshape(rows, cols);
                foreach (var edit in edits)
                {
                    editable[edit.Y, edit.X] = torch.tensor(edit.NewValue);
                }
                return t.reshape(t.shape);
            }
            return t;
        }

        /// <summary>
        /// Computes the delta matrix (2D float64 tensor) for the given adapter info.
        /// </summary>
        public static torch.Tensor ComputeDelta(AdapterInfo adapter, SafetensorsFileReader sfr, Dictionary<string, List<TensorEdit>>? pendingEdits = null)
        {
            switch (adapter.Type)
            {
                case AdapterType.LoRA:
                    {
                        using var down = LoadTensorWithEdits(adapter.SubkeyMap["down"], sfr, pendingEdits);
                        using var up = LoadTensorWithEdits(adapter.SubkeyMap["up"], sfr, pendingEdits);

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
                        using var w1a = LoadTensorWithEdits(adapter.SubkeyMap["hada_w1_a"], sfr, pendingEdits);
                        using var w1b = LoadTensorWithEdits(adapter.SubkeyMap["hada_w1_b"], sfr, pendingEdits);
                        using var w2a = LoadTensorWithEdits(adapter.SubkeyMap["hada_w2_a"], sfr, pendingEdits);
                        using var w2b = LoadTensorWithEdits(adapter.SubkeyMap["hada_w2_b"], sfr, pendingEdits);

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
                        using var w1 = LoadTensorWithEdits(adapter.SubkeyMap["lokr_w1"], sfr, pendingEdits);
                        using var w2a = LoadTensorWithEdits(adapter.SubkeyMap["lokr_w2_a"], sfr, pendingEdits);
                        using var w2b = LoadTensorWithEdits(adapter.SubkeyMap["lokr_w2_b"], sfr, pendingEdits);

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
                        using var w1 = LoadTensorWithEdits(adapter.SubkeyMap["lokr_w1"], sfr, pendingEdits);
                        using var w2 = LoadTensorWithEdits(adapter.SubkeyMap["lokr_w2"], sfr, pendingEdits);

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

        /// <summary>
        /// Decomposes an edit at cell (dataY, x) on a computed delta matrix into minimal-norm updates
        /// on the underlying sub-tensors, recording them in pendingEdits.
        /// </summary>
        public static void DecomposeDeltaEdit(
            AdapterInfo adapter,
            SafetensorsFileReader sfr,
            int dataY,
            int x,
            double deltaChange,
            Dictionary<string, List<TensorEdit>> pendingEdits)
        {
            switch (adapter.Type)
            {
                case AdapterType.LoRA:
                    {
                        string upKey = adapter.SubkeyMap["up"];
                        string downKey = adapter.SubkeyMap["down"];
                        using var down = LoadTensorWithEdits(downKey, sfr, pendingEdits);
                        using var up = LoadTensorWithEdits(upKey, sfr, pendingEdits);

                        long rank = down.shape[0];
                        double scale = rank > 0 ? (adapter.Alpha / rank) : 1.0;
                        double d = deltaChange / scale;

                        torch.Tensor up2D = up.ndim > 2 ? up.flatten(1) : up;
                        torch.Tensor down2D = down.ndim > 2 ? down.flatten(1) : down;

                        int r = (int)rank;
                        double[] u = new double[r];
                        double[] v = new double[r];
                        double normU2 = 0, normV2 = 0;

                        using (var upAcc = up2D.data<double>())
                        using (var downAcc = down2D.data<double>())
                        {
                            int upCols = (int)up2D.shape[1];
                            int downCols = (int)down2D.shape[1];

                            for (int k = 0; k < r; k++)
                            {
                                u[k] = upAcc[dataY * upCols + k];
                                v[k] = downAcc[k * downCols + x];
                                normU2 += u[k] * u[k];
                                normV2 += v[k] * v[k];
                            }

                            double denom = normU2 + normV2;
                            for (int k = 0; k < r; k++)
                            {
                                double du = denom > 1e-12 ? (d * v[k] / denom) : (d / (2 * r));
                                double dv = denom > 1e-12 ? (d * u[k] / denom) : (d / (2 * r));

                                RecordEdit(pendingEdits, upKey, k, dataY, u[k] + du);
                                RecordEdit(pendingEdits, downKey, x, k, v[k] + dv);
                            }
                        }
                    }
                    break;

                case AdapterType.LoKrFullRank:
                    {
                        string w1Key = adapter.SubkeyMap["lokr_w1"];
                        string w2Key = adapter.SubkeyMap["lokr_w2"];
                        using var w1 = LoadTensorWithEdits(w1Key, sfr, pendingEdits);
                        using var w2 = LoadTensorWithEdits(w2Key, sfr, pendingEdits);

                        torch.Tensor w1_2D = w1.ndim > 2 ? w1.flatten(1) : w1;
                        torch.Tensor w2_2D = w2.ndim > 2 ? w2.flatten(1) : w2;

                        double scale = adapter.Alpha;
                        double d = deltaChange / scale;

                        int M1 = (int)w1_2D.shape[0];
                        int N1 = (int)w1_2D.shape[1];

                        int r2 = dataY / M1;
                        int r1 = dataY % M1;
                        int c2 = x / N1;
                        int c1 = x % N1;

                        double val1, val2;
                        using (var acc1 = w1_2D.data<double>())
                        using (var acc2 = w2_2D.data<double>())
                        {
                            val1 = acc1[r1 * N1 + c1];
                            val2 = acc2[r2 * (int)w2_2D.shape[1] + c2];
                        }

                        double denom = val1 * val1 + val2 * val2;
                        double dw1 = denom > 1e-12 ? (d * val2 / denom) : (d >= 0 ? Math.Sqrt(d) : -Math.Sqrt(-d));
                        double dw2 = denom > 1e-12 ? (d * val1 / denom) : (d >= 0 ? Math.Sqrt(d) : -Math.Sqrt(-d));

                        RecordEdit(pendingEdits, w1Key, c1, r1, val1 + dw1);
                        RecordEdit(pendingEdits, w2Key, c2, r2, val2 + dw2);
                    }
                    break;

                case AdapterType.LoKrStandard:
                    {
                        string w1Key = adapter.SubkeyMap["lokr_w1"];
                        string w2aKey = adapter.SubkeyMap["lokr_w2_a"];
                        string w2bKey = adapter.SubkeyMap["lokr_w2_b"];
                        using var w1 = LoadTensorWithEdits(w1Key, sfr, pendingEdits);
                        using var w2a = LoadTensorWithEdits(w2aKey, sfr, pendingEdits);
                        using var w2b = LoadTensorWithEdits(w2bKey, sfr, pendingEdits);

                        long rank = w2a.shape[0];
                        double scale = rank > 0 ? (adapter.Alpha / rank) : 1.0;
                        double d = deltaChange / scale;

                        torch.Tensor w1_2D = w1.ndim > 2 ? w1.flatten(1) : w1;
                        torch.Tensor w2a2D = w2a.ndim > 2 ? w2a.flatten(1) : w2a;
                        torch.Tensor w2b2D = w2b.ndim > 2 ? w2b.flatten(1) : w2b;
                        using var w2 = torch.matmul(w2b2D, w2a2D);

                        int M1 = (int)w1_2D.shape[0];
                        int N1 = (int)w1_2D.shape[1];

                        int r2 = dataY / M1;
                        int r1 = dataY % M1;
                        int c2 = x / N1;
                        int c1 = x % N1;

                        double val1, val2;
                        using (var acc1 = w1_2D.data<double>())
                        using (var acc2 = w2.data<double>())
                        {
                            val1 = acc1[r1 * N1 + c1];
                            val2 = acc2[r2 * (int)w2.shape[1] + c2];
                        }

                        double denom = val1 * val1 + val2 * val2;
                        double dw1 = denom > 1e-12 ? (d * val2 / denom) : (d >= 0 ? Math.Sqrt(d) : -Math.Sqrt(-d));
                        double dw2 = denom > 1e-12 ? (d * val1 / denom) : (d >= 0 ? Math.Sqrt(d) : -Math.Sqrt(-d));

                        RecordEdit(pendingEdits, w1Key, c1, r1, val1 + dw1);

                        // Decompose dw2 onto w2b and w2a
                        int r = (int)rank;
                        double[] u = new double[r];
                        double[] v = new double[r];
                        double normU2 = 0, normV2 = 0;
                        using (var bAcc = w2b2D.data<double>())
                        using (var aAcc = w2a2D.data<double>())
                        {
                            int bCols = (int)w2b2D.shape[1];
                            int aCols = (int)w2a2D.shape[1];
                            for (int k = 0; k < r; k++)
                            {
                                u[k] = bAcc[r2 * bCols + k];
                                v[k] = aAcc[k * aCols + c2];
                                normU2 += u[k] * u[k];
                                normV2 += v[k] * v[k];
                            }

                            double denom2 = normU2 + normV2;
                            for (int k = 0; k < r; k++)
                            {
                                double du = denom2 > 1e-12 ? (dw2 * v[k] / denom2) : (dw2 / (2 * r));
                                double dv = denom2 > 1e-12 ? (dw2 * u[k] / denom2) : (dw2 / (2 * r));

                                RecordEdit(pendingEdits, w2bKey, k, r2, u[k] + du);
                                RecordEdit(pendingEdits, w2aKey, c2, k, v[k] + dv);
                            }
                        }
                    }
                    break;

                case AdapterType.LoHa:
                    {
                        string w1aKey = adapter.SubkeyMap["hada_w1_a"];
                        string w1bKey = adapter.SubkeyMap["hada_w1_b"];
                        string w2aKey = adapter.SubkeyMap["hada_w2_a"];
                        string w2bKey = adapter.SubkeyMap["hada_w2_b"];

                        using var w1a = LoadTensorWithEdits(w1aKey, sfr, pendingEdits);
                        using var w1b = LoadTensorWithEdits(w1bKey, sfr, pendingEdits);
                        using var w2a = LoadTensorWithEdits(w2aKey, sfr, pendingEdits);
                        using var w2b = LoadTensorWithEdits(w2bKey, sfr, pendingEdits);

                        long rank = w1a.shape[0];
                        double scale = rank > 0 ? (adapter.Alpha / rank) : 1.0;
                        double d = deltaChange / scale;

                        torch.Tensor w1a2D = w1a.ndim > 2 ? w1a.flatten(1) : w1a;
                        torch.Tensor w1b2D = w1b.ndim > 2 ? w1b.flatten(1) : w1b;
                        torch.Tensor w2a2D = w2a.ndim > 2 ? w2a.flatten(1) : w2a;
                        torch.Tensor w2b2D = w2b.ndim > 2 ? w2b.flatten(1) : w2b;

                        using var w1 = torch.matmul(w1b2D, w1a2D);
                        using var w2 = torch.matmul(w2b2D, w2a2D);

                        double val1, val2;
                        using (var acc1 = w1.data<double>())
                        using (var acc2 = w2.data<double>())
                        {
                            int cols = (int)w1.shape[1];
                            val1 = acc1[dataY * cols + x];
                            val2 = acc2[dataY * cols + x];
                        }

                        double denom = val1 * val1 + val2 * val2;
                        double dw1 = denom > 1e-12 ? (d * val2 / denom) : (d >= 0 ? Math.Sqrt(d) : -Math.Sqrt(-d));
                        double dw2 = denom > 1e-12 ? (d * val1 / denom) : (d >= 0 ? Math.Sqrt(d) : -Math.Sqrt(-d));

                        // Decompose dw1 onto w1b and w1a
                        int r = (int)rank;
                        double[] u1 = new double[r];
                        double[] v1 = new double[r];
                        double normU1_2 = 0, normV1_2 = 0;
                        using (var bAcc = w1b2D.data<double>())
                        using (var aAcc = w1a2D.data<double>())
                        {
                            int bCols = (int)w1b2D.shape[1];
                            int aCols = (int)w1a2D.shape[1];
                            for (int k = 0; k < r; k++)
                            {
                                u1[k] = bAcc[dataY * bCols + k];
                                v1[k] = aAcc[k * aCols + x];
                                normU1_2 += u1[k] * u1[k];
                                normV1_2 += v1[k] * v1[k];
                            }
                            double denom1 = normU1_2 + normV1_2;
                            for (int k = 0; k < r; k++)
                            {
                                double du = denom1 > 1e-12 ? (dw1 * v1[k] / denom1) : (dw1 / (2 * r));
                                double dv = denom1 > 1e-12 ? (dw1 * u1[k] / denom1) : (dw1 / (2 * r));

                                RecordEdit(pendingEdits, w1bKey, k, dataY, u1[k] + du);
                                RecordEdit(pendingEdits, w1aKey, x, k, v1[k] + dv);
                            }
                        }

                        // Decompose dw2 onto w2b and w2a
                        double[] u2 = new double[r];
                        double[] v2 = new double[r];
                        double normU2_2 = 0, normV2_2 = 0;
                        using (var bAcc = w2b2D.data<double>())
                        using (var aAcc = w2a2D.data<double>())
                        {
                            int bCols = (int)w2b2D.shape[1];
                            int aCols = (int)w2a2D.shape[1];
                            for (int k = 0; k < r; k++)
                            {
                                u2[k] = bAcc[dataY * bCols + k];
                                v2[k] = aAcc[k * aCols + x];
                                normU2_2 += u2[k] * u2[k];
                                normV2_2 += v2[k] * v2[k];
                            }
                            double denom2 = normU2_2 + normV2_2;
                            for (int k = 0; k < r; k++)
                            {
                                double du = denom2 > 1e-12 ? (dw2 * v2[k] / denom2) : (dw2 / (2 * r));
                                double dv = denom2 > 1e-12 ? (dw2 * u2[k] / denom2) : (dw2 / (2 * r));

                                RecordEdit(pendingEdits, w2bKey, k, dataY, u2[k] + du);
                                RecordEdit(pendingEdits, w2aKey, x, k, v2[k] + dv);
                            }
                        }
                    }
                    break;
            }
        }

        private static void RecordEdit(Dictionary<string, List<TensorEdit>> pendingEdits, string key, int x, int y, double newValue)
        {
            if (!pendingEdits.TryGetValue(key, out var edits))
            {
                edits = [];
                pendingEdits[key] = edits;
            }
            int existingIdx = edits.FindIndex(e => e.X == x && e.Y == y);
            if (existingIdx >= 0)
            {
                edits[existingIdx] = new TensorEdit(x, y, newValue);
            }
            else
            {
                edits.Add(new TensorEdit(x, y, newValue));
            }
        }
    }
}
