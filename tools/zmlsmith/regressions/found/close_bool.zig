const std = @import("std");
const zml = @import("zml");

test "zml.testing.expectClose on bool slices" {
    const sh: zml.Shape = .init(.{2}, .bool);
    var data = [_]bool{ true, false };
    const s: zml.Slice = .init(sh, std.mem.sliceAsBytes(&data));
    try zml.testing.expectClose(std.testing.io, s, s, .{});
}
