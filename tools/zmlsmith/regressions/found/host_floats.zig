const std = @import("std");
const zml = @import("zml");
const floats = zml.floats;

// Reference values from the OCP 8-bit float spec and ml_dtypes.

test "floats: FNUZ and B11FNUZ exponent bias" {
    // Smallest normal: exponent field 1, mantissa 0, is 2^(1 - bias).
    try std.testing.expectEqual(std.math.pow(f32, 2, -7), floats.Float8E4M3FNUZ.toF32(@bitCast(@as(u8, 0x08))));
    try std.testing.expectEqual(std.math.pow(f32, 2, -15), floats.Float8E5M2FNUZ.toF32(@bitCast(@as(u8, 0x04))));
    try std.testing.expectEqual(std.math.pow(f32, 2, -10), floats.Float8E4M3B11FNUZ.toF32(@bitCast(@as(u8, 0x08))));
}

test "floats: f8 subnormals" {
    // E4M3FN code 0x01 is the smallest subnormal, 2^-9.
    try std.testing.expectEqual(std.math.pow(f32, 2, -9), floats.Float8E4M3FN.toF32(@bitCast(@as(u8, 0x01))));
    // E5M2 code 0x01 is 2^-16.
    try std.testing.expectEqual(std.math.pow(f32, 2, -16), floats.Float8E5M2.toF32(@bitCast(@as(u8, 0x01))));
}

test "floats: e8m0 0xff is NaN" {
    try std.testing.expect(std.math.isNan(floats.Float8E8M0.toF32(@bitCast(@as(u8, 0xff)))));
}

test "floats: -0.0 doesn't encode to the FNUZ NaN" {
    // FNUZ types have no negative zero: 0x80 is their only NaN.
    try std.testing.expect(@as(u8, @bitCast(floats.Float8E4M3FNUZ.fromF32(-0.0))) != 0x80);
    try std.testing.expect(@as(u8, @bitCast(floats.Float8E5M2FNUZ.fromF32(-0.0))) != 0x80);
}
