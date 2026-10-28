const std = @import("std");
const zml = @import("zml");

// Tensor.slices with one dynamic slice takes the dynamic_slice path for every axis and sets
// each new dim to `s.len`, which is `to_the_end` for a static `.{ .start, .end }` slice.
// The doc comment's first example mixes the two this way.
test "slices_mixed" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor, off: zml.Tensor) zml.Tensor {
            return x.slices(.{ .a, .b }, &.{ .{ .start = 1, .end = 3 }, .{ .dyn_start = off, .len = 2 } });
        }
    };
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .a = 4, .b = 5 }, .f32));
    const off: zml.Tensor = .fromShape(zml.Shape.init(.{}, .i32));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{ x, off }, platform, .{});
    defer exe.deinit();
    try zml.testing.expectEqualShapes(zml.Shape.init(.{ .a = 2, .b = 2 }, .f32), exe.output_shapes[0]);
}
