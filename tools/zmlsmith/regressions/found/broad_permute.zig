const std = @import("std");
const zml = @import("zml");

// Tensor.broadcast returns self when the dims already match, ignoring `axes_`: mapping
// axis 0 to 1 and 1 to 0 on a square tensor should transpose it.
test "broad_permute" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x.broadcast(x.shape(), &.{ 1, 0 });
        }
    };
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ 2, 2 }, .f32));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();

    const in = [_]f32{ 1, 2, 3, 4 };
    var b: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(&in));
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{b});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 2, 4 }, got.items(f32));
}
