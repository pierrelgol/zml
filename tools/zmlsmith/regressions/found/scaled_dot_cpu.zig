const std = @import("std");
const zml = @import("zml");

// nn.scaledDot is a `xla.scaled_dot` composite whose decomposition is the custom call
// `zml$scaled_dot_unmatched`. Nothing registers it on CPU, so every scaledDot, and every
// quantized nn.Linear.forward, fails to compile there: "No FFI handler registered for
// zml$scaled_dot_unmatched on a platform Host". The doc comment says "CPU has no
// specialized path", which reads as if the dequantize + dot fallback applied.
test "scaled_dot_cpu" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(lhs: zml.Tensor, rhs: zml.Tensor, lhs_scale: zml.Tensor, rhs_scale: zml.Tensor) zml.Tensor {
            return zml.nn.scaledDot(lhs, rhs, lhs_scale, rhs_scale, .f32, .k, .{});
        }
    };

    // MXFP8: f8e4m3fn values, one e8m0 scale per 32 contracted values.
    const lhs: zml.Tensor = .fromShape(zml.Shape.init(.{ .m = 2, .k = 64 }, .f8e4m3fn));
    const rhs: zml.Tensor = .fromShape(zml.Shape.init(.{ .n = 3, .k = 64 }, .f8e4m3fn));
    const lhs_scale: zml.Tensor = .fromShape(zml.Shape.init(.{ .m = 2, .k = 2 }, .f8e8m0));
    const rhs_scale: zml.Tensor = .fromShape(zml.Shape.init(.{ .n = 3, .k = 2 }, .f8e8m0));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{ lhs, rhs, lhs_scale, rhs_scale }, platform, .{});
    defer exe.deinit();
    try zml.testing.expectEqualShapes(zml.Shape.init(.{ .m = 2, .n = 3 }, .f32), exe.output_shapes[0]);
}
