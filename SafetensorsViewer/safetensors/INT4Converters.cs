using System;

public static class INT4Converters
{
    // Interpret INT4 as symmetric signed range -1..1.
    // Nibble value 0 -> -1.0
    // Nibble value 7 -> 0.0
    // Nibble value 14 -> +1.0
    // Value 15 is treated as NaN/invalid (we map it to 0 for safety).
    public static double NibbleToDouble(byte nibble)
    {
        if (nibble > 15)
            throw new ArgumentOutOfRangeException(nameof(nibble), "Nibble must be 0..15");

        if (nibble == 15)
            return 0.0;

        // Signed range -7..7 mapped to nibble 0..14, 15 reserved.
        int signed = (int)nibble - 7;
        return signed / 7.0;
    }

    public static byte DoubleToNibble(double value)
    {
        // Clamp to -1..1
        if (value < -1.0) value = -1.0;
        if (value > 1.0) value = 1.0;

        int signed = (int)Math.Round(value * 7.0);
        // signed is now -7..7
        int nibble = signed + 7;
        if (nibble < 0) nibble = 0;
        if (nibble > 14) nibble = 14;
        return (byte)nibble;
    }
}
