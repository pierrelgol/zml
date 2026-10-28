const std = @import("std");
const zml = @import("zml");

// Slice.prettyPrint of a rank-0 view reads `bytes.ptr` and ignores `offset_bytes`, so a
// scalar taken from a larger slice with subSlice + squeeze prints the first element of
// the buffer (zml/slice.zig:259 at 74c590b0). Found by auditing zmlsmith coverage of
// zml.Slice.
test "slice_print" {
    var data = [_]f32{ 10, 20, 30 };
    const s: zml.Slice = .init(zml.Shape.init(.{3}, .f32), std.mem.sliceAsBytes(&data));
    const last = s.subSlice(0, 2, 1).squeeze(0);
    const txt = try std.fmt.allocPrint(std.testing.allocator, "{d}", .{last});
    defer std.testing.allocator.free(txt);
    try std.testing.expectEqualStrings("30", txt);
}
