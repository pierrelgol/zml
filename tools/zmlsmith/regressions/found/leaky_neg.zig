const std = @import("std");
const zml = @import("zml");

// Tensor.leakyReLU computes relu(x) + min(slope * x, 0). Its doc formula is
// max(0, x) + slope * min(0, x): the two differ for a negative slope.
test "leaky_neg" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x.leakyReLU(-0.5);
        }
    };
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .a = 2 }, .f32));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();

    const in = [_]f32{ -2, 2 };
    var b: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(&in));
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{b});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    // max(0, -2) - 0.5 * min(0, -2) = 1, max(0, 2) - 0.5 * min(0, 2) = 2.
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, got.items(f32));
}
