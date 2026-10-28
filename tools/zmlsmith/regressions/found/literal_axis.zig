const std = @import("std");
const zml = @import("zml");

// Shape.axisFromTagMaybe resolves the `._N` tag syntax to axis N without checking it
// against the rank: hasTag(._5) on a rank 2 shape returns 5 instead of null, and APIs that
// take the tag as an axis (sum, slice, gather, ...) index past the dims instead of
// reporting a missing axis. Found by zmlsmith host checks.
test "literal_axis" {
    const sh: zml.Shape = .init(.{ .a = 2, .b = 3 }, .f32);
    try std.testing.expectEqual(@as(?u3, 1), sh.hasTag(._1));
    try std.testing.expectEqual(@as(?u3, null), sh.hasTag(._5));
}
