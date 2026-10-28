const std = @import("std");
const zml = @import("zml");

// Compiler.repack splits the last dim of sub-byte arguments and results into
// `.{ -1, items_per_byte }`, which can't be inferred when the tensor has no elements:
// "Can't split { 2, 0 } along axis 1 into { 2, -1, 4 }: TooBig". Found by zmlsmith from
// seed 900133.
test "subbyte_empty" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x;
        }
    };

    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .a = 2, .b = 0 }, .i2));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();
    try zml.testing.expectEqualShapes(x.shape(), exe.output_shapes[0]);
}
