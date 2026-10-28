const std = @import("std");
const zml = @import("zml");

// zml.testing.expectClose on identical f8e4m3 slices hits `unreachable` (testing.zig:113).
// The left dtype switch accepts .f8e4m3 (testing.zig:85) but the right dtype switch
// (testing.zig:95-104) has no .f8e4m3 arm, so a tensor compared with itself panics instead
// of passing. Basis: a slice compared with itself is close under any tolerance.
test "close_f8" {
    const sh: zml.Shape = .init(.{2}, .f8e4m3);
    var data = [_]u8{ 0x38, 0x40 };
    const s: zml.Slice = .init(sh, std.mem.sliceAsBytes(&data));
    try zml.testing.expectClose(std.testing.io, s, s, .{});
}
