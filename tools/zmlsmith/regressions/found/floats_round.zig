const std = @import("std");
const zml = @import("zml");
const floats = zml.floats;

// zml.floats fromF32 doesn't round to nearest even. BFloat16.fromF32 adds 0x8000 (half
// up); the f8/f4 truncMantissa adds half a unit and truncates the mantissa. These feed
// DataType.constant, so Tensor.scalar(x, .bf16) and f8 constants are affected.
// Found by zmlsmith host checks.
test "floats_round_bf16_tie" {
    // 1 + 2^-8 is halfway between bf16 1.0 and 1.0078125: the even one is 1.0.
    try std.testing.expectEqual(@as(f32, 1.0), floats.BFloat16.fromF32(1.0 + 0x1p-8).toF32());
}

test "floats_round_f4_tie" {
    // 0.25 is halfway between f4e2m1 0 and 0.5: the even one is 0.
    try std.testing.expectEqual(@as(f32, 0.0), floats.Float4E2M1.fromF32(0.25).toF32());
}

test "floats_round_bf16_nan" {
    // A NaN with a small payload rounds into the exponent: 0x7F800001 becomes +inf.
    const nan: f32 = @bitCast(@as(u32, 0x7F800001));
    try std.testing.expect(std.math.isNan(floats.BFloat16.fromF32(nan).toF32()));
}

test "floats_round_bf16_nan_wraps" {
    // 0x7FFF8000 + 0x8000 carries out of the exponent into the sign: -0.
    const nan: f32 = @bitCast(@as(u32, 0x7FFF8000));
    try std.testing.expect(std.math.isNan(floats.BFloat16.fromF32(nan).toF32()));
}

test "floats_round_f8_nan" {
    // The generic fromF32 takes the overflow branch for NaN: e5m2 returns inf.
    try std.testing.expect(floats.Float8E5M2.fromF32(std.math.nan(f32)).isNan());
}

test "floats_round_f8_nan_to_f32" {
    // The generic toF32 only special cases zero and inf: e5m2 NaN decodes as a finite value.
    try std.testing.expect(std.math.isNan(floats.Float8E5M2.nan.toF32()));
}

test "floats_round_f8_carry" {
    // 1.97 rounds to 2.0 in f8e4m3fn: the rounding carries into the exponent.
    try std.testing.expectEqual(@as(f32, 2.0), floats.Float8E4M3FN.fromF32(1.97).toF32());
}
