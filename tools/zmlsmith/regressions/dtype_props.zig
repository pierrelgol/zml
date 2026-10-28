//! DataType metadata against the Zig types it maps to, for every dtype.

const std = @import("std");
const zml = @import("zml");

test "dtype: zig type round trip, size, alignment and class" {
    inline for (comptime std.enums.values(zml.DataType)) |dt| {
        const T = dt.toZigType();
        try std.testing.expectEqual(dt, zml.DataType.fromZigType(T));
        try std.testing.expectEqual(@as(u16, @sizeOf(T)), dt.sizeOf());
        try std.testing.expectEqual(@as(u29, @alignOf(T)), dt.alignOf());
        const n = @intFromBool(dt.isInteger()) + @intFromBool(dt.isFloat()) + @intFromBool(dt.isComplex()) + @intFromBool(dt == .bool);
        try std.testing.expectEqual(1, n);
        if (dt.isSignedInt()) try std.testing.expect(dt.isInteger());
        try std.testing.expect(dt.bitSizeOf() <= 8 * dt.sizeOf());
    }
}
