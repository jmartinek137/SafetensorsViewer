using System;

public static class FP8Converters
{
    // FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
    public static double E4M3ToDouble(byte value)
    {
        if (value == 0) return 0.0;

        bool sign = (value & 0x80) != 0;
        int exponent = (value >> 3) & 0x0F;
        int mantissa = value & 0x07;

        if (exponent == 0x0F)
        {
            if (mantissa == 0x07)
                return sign ? double.NaN : double.NaN;
            // NaN pattern or reserved
            return double.NaN;
        }

        // E4M3 bias is 7
        int exp = exponent - 7;
        double significand = 1.0 + mantissa / 8.0;

        double result = significand * Math.Pow(2, exp);
        return sign ? -result : result;
    }

    public static byte DoubleToE4M3(double value)
    {
        if (value == 0.0 || double.IsNaN(value)) return 0;

        bool sign = value < 0;
        value = Math.Abs(value);

        // Clamp to E4M3 max range ~448
        double maxE4M3 = 448.0;
        if (value > maxE4M3) value = maxE4M3;

        int exp = (int)Math.Floor(Math.Log(value, 2));
        int biasedExp = exp + 7;

        if (biasedExp < 0) return 0;
        if (biasedExp > 0x0E) biasedExp = 0x0E;

        double significand = value / Math.Pow(2, exp);
        int mantissa = (int)Math.Round((significand - 1.0) * 8.0);
        if (mantissa > 7)
        {
            mantissa = 0;
            biasedExp++;
        }
        if (biasedExp > 0x0E) biasedExp = 0x0E;

        byte result = (byte)(((sign ? 1 : 0) << 7) | (biasedExp << 3) | mantissa);
        return result;
    }

    // FP8 E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits
    public static double E5M2ToDouble(byte value)
    {
        if (value == 0) return 0.0;

        bool sign = (value & 0x80) != 0;
        int exponent = (value >> 2) & 0x1F;
        int mantissa = value & 0x03;

        if (exponent == 0x1F)
        {
            return double.NaN;
        }

        // E5M2 bias is 15
        int exp = exponent - 15;
        double significand = 1.0 + mantissa / 4.0;

        double result = significand * Math.Pow(2, exp);
        return sign ? -result : result;
    }

    public static byte DoubleToE5M2(double value)
    {
        if (value == 0.0 || double.IsNaN(value)) return 0;

        bool sign = value < 0;
        value = Math.Abs(value);

        // Clamp to E5M2 max range ~57344
        double maxE5M2 = 57344.0;
        if (value > maxE5M2) value = maxE5M2;

        int exp = (int)Math.Floor(Math.Log(value, 2));
        int biasedExp = exp + 15;

        if (biasedExp < 0) return 0;
        if (biasedExp > 0x1D) biasedExp = 0x1D;

        double significand = value / Math.Pow(2, exp);
        int mantissa = (int)Math.Round((significand - 1.0) * 4.0);
        if (mantissa > 3)
        {
            mantissa = 0;
            biasedExp++;
        }
        if (biasedExp > 0x1D) biasedExp = 0x1D;

        byte result = (byte)(((sign ? 1 : 0) << 7) | (biasedExp << 2) | mantissa);
        return result;
    }
}
