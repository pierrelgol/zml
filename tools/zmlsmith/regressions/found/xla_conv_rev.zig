const std = @import("std");
const zml = @import("zml");

// Upstream XLA CPU: a convolution with window reversal, a stride > 1 and padding returns a
// buffer of the wrong size: ZML and the StableHLO type say {1, 2, 2, 2} (8 elements), the
// executable writes 16, and reading it back fails ("`dst_size` must be >= 64, got 32").
// Any two of the three options work; every dtype is affected. Found by zmlsmith from
// seeds 1000183 and 1101637.
test "xla_conv_rev" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
            return x.conv2d(w, .{ .window_strides = &.{ 2, 2 }, .padding = &.{ 0, 1, 1, 1 }, .window_reversal = &.{ false, true } });
        }
    };
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ 1, 1, 5, 2 }, .f32));
    const w: zml.Tensor = .fromShape(zml.Shape.init(.{ 2, 1, 3, 2 }, .f32));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{ x, w }, platform, .{});
    defer exe.deinit();
    try zml.testing.expectEqualShapes(zml.Shape.init(.{ 1, 2, 2, 2 }, .f32), exe.output_shapes[0]);

    var xb: zml.Buffer = try .uninitialized(io, platform, x.shape(), .replicated, .{});
    defer xb.deinit();
    var wb: zml.Buffer = try .uninitialized(io, platform, w.shape(), .replicated, .{});
    defer wb.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{ xb, wb });
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
}
