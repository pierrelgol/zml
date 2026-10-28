const std = @import("std");
const zml = @import("zml");

// Compiles under a short file name (cartesian_scalar.zig), not under this one: pjrt.zig
// Struct/structSize exceed the default eval branch quota when the root module name is long.
test "pjrt branch quota with a long root module name" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x0: zml.Tensor, x1: zml.Tensor) [1]zml.Tensor {
            const x2 = zml.Tensor.cartesianProduct(3, .{ x0, x1, x0 })[2];
            _ = &x2;
            return .{x2};
        }
    };

    const x0: zml.Tensor = .fromShape(zml.Shape.init(.{}, .f64));
    const x1: zml.Tensor = .fromShape(zml.Shape.init(.{ .f = 3 }, .f64));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{ x0, x1 }, platform, .{});
    defer exe.deinit();

    try zml.testing.expectEqualShapes(zml.Shape.init(.{ 1, 3, 1 }, .f64).withTags(.{ ._, .f, ._ }), exe.output_shapes[0]);

    var b0: zml.Buffer = try .fromBytes(io, platform, x0.shape(), .replicated, &[_]u8{
        0, 0, 0, 0, 0, 0, 0, 0,
    });
    defer b0.deinit();
    var b1: zml.Buffer = try .fromBytes(io, platform, x1.shape(), .replicated, &[_]u8{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    });
    defer b1.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{ b0, b1 });
    defer for (&res) |*r| r.deinit();

    {
        const expected_data align(16) = [_]u8{
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        };
        const expected: zml.Slice = .init(zml.Shape.init(.{ 1, 3, 1 }, .f64).withTags(.{ ._, .f, ._ }), @as([]const u8, &expected_data));
        try zml.testing.expectClose(io, expected, res[0], .exact_match);
    }
}
