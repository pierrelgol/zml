const std = @import("std");
const zml = @import("zml");

// Upstream XLA: converting f4e2m1 to an integer type saturates +-6.0, the largest code, to
// the integer type's max/min. Every other code converts correctly, and so do conversions
// to floats. Found by zmlsmith from seed 900061.
test "xla_f4_to_int" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x.convert(.i32);
        }
    };

    // Codes 0x7 (6.0), 0x6 (4.0), 0xf (-6.0), 0xe (-4.0), low nibble first.
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .k = 4 }, .f4e2m1));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();

    var b: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, &[_]u8{ 0x67, 0xef });
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{b});
    defer res.deinit();

    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    try std.testing.expectEqualSlices(i32, &.{ 6, 4, -6, -4 }, got.items(i32));
}
