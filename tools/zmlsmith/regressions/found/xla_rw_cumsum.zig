const std = @import("std");
const zml = @import("zml");

// Upstream XLA CPU (probably): a padded reduce_window (here an identity window with
// padding 2 on each side, so [0, 0, 1, 0, 0]) followed by Tensor.cumulativeSum, itself a
// padded reduce_window, gives a wrong prefix sum: element 0 is 1 instead of 0 and 2 of 5
// elements differ. Each op alone is correct. In the original seed the wrong elements
// were garbage (3834619964). Found by zmlsmith (seed 110613), reduced from 43 nodes.
test "xla_rw_cumsum" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x0: zml.Tensor) [1]zml.Tensor {
            const x1 = zml.ops.reduceWindow(1, .{x0}, .{zml.Tensor.constant(zml.DataType.zero(.u32))}, .{ .window_dimensions = &[_]i64{1}, .window_strides = &[_]i64{1}, .base_dilations = &[_]i64{1}, .window_dilations = &[_]i64{1}, .padding = &[_][2]i64{.{ 2, 2 }} }, &struct {
                fn f(a: zml.ops.ReduceArgs) [1]zml.Tensor {
                    return .{a.left.add(a.right)};
                }
            }.f)[0];
            _ = &x1;
            const x2 = x1.cumulativeSum(-1);
            _ = &x2;
            return .{x2};
        }
    };

    const x0: zml.Tensor = .fromShape(zml.Shape.init(.{ .h = 1 }, .u32));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x0}, platform, .{});
    defer exe.deinit();

    try zml.testing.expectEqualShapes(zml.Shape.init(.{ .h = 5 }, .u32), exe.output_shapes[0]);

    var b0: zml.Buffer = try .fromBytes(io, platform, x0.shape(), .replicated, &[_]u8{
        1, 0, 0, 0,
    });
    defer b0.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{b0});
    defer for (&res) |*r| r.deinit();

    {
        const expected_data align(16) = [_]u8{
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        };
        const expected: zml.Slice = .init(zml.Shape.init(.{ .h = 5 }, .u32), @as([]const u8, &expected_data));
        const got = try res[0].toSliceAlloc(allocator, io);
        defer got.free(allocator);
        std.debug.print("got {any}\n", .{got.constItems(u32)});
        try std.testing.expectEqualSlices(u8, expected.constData(), got.constData());
    }
}
