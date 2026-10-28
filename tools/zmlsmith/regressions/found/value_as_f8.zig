const std = @import("std");
const zml = @import("zml");

// DataType.Value.as converts a value to a host type. Its switch (zml/dtype.zig:355-370) has
// no f8e4m3 arm, so as(f32) on an f8e4m3 value panics with "Unsupported conversion" even
// though the value is representable in f32. Expected: 1.0 round-trips.
test "value_as_f8" {
    const v = zml.DataType.Value.init(.f8e4m3, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), v.as(f32));
}
