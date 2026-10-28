const std = @import("std");
const floats = @import("zml").floats;

// BFloat16.fromF32 rounds with `int += 0x8000` on the f32 bits without checking for NaN:
// negative NaNs with a high payload overflow u32, a panic in safe builds
// (zml/floats.zig:354 at 74c590b0). Found by reading the code while auditing zmlsmith.
test "bf16_nan_ovf" {
    const nan: f32 = @bitCast(@as(u32, 0xFFFF8000));
    try std.testing.expect(std.math.isNan(floats.BFloat16.fromF32(nan).toF32()));
}
