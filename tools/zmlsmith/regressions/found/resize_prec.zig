const std = @import("std");
const zml = @import("zml");
// resizeBilinear with .precision = .f32 on a bf16 image must match resizeLinear1d run with the
// same options. The per-axis call in nn.resizeBilinear (zml/nn.zig:1418-1420 at 74c590b0)
// keeps only original_len, so the bilinear result runs in bf16 and differs from the f32 one.
test "resize_prec" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Local = struct {
        fn forward(x: zml.Tensor) [2]zml.Tensor {
            const opts: zml.nn.ResizeOpts = .{ .precision = .f32 };
            return .{ zml.nn.resizeBilinear(x, .{ .a = 7 }, opts), zml.nn.resizeLinear1d(x, 0, 7, opts) };
        }
    };
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .a = 3 }, .bf16));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();
    const in = [_]zml.floats.BFloat16{ .fromF32(1.0), .fromF32(1.0078125), .fromF32(250.0) };
    var b: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(&in));
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{b});
    defer for (&res) |*r| r.deinit();
    const bilinear = try res[0].toSliceAlloc(allocator, io);
    defer bilinear.free(allocator);
    const linear = try res[1].toSliceAlloc(allocator, io);
    defer linear.free(allocator);
    try std.testing.expectEqualSlices(u8, linear.constData(), bilinear.constData());
}
