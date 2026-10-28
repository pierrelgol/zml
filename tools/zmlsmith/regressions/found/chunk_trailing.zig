const std = @import("std");
const zml = @import("zml");

// chunkAllowTrailing(axis, n) on a dim of 5 with n = 3 must return 3 chunks of 1 item
// plus one trailing chunk of the remaining 2 items (see the doc comment above
// Tensor.chunkAllowTrailing and its test). The tail is computed as @rem(d, chunk_size)
// instead of d - n * chunk_size, so no trailing chunk is produced.
test "chunk_trailing" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x0: zml.Tensor) [1]zml.Tensor {
            const chunks = x0.chunkAllowTrailing(1, 3);
            return .{chunks[chunks.len - 1]};
        }
    };

    const x0: zml.Tensor = .fromShape(zml.Shape.init(.{ 2, 5 }, .f32));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x0}, platform, .{});
    defer exe.deinit();

    try zml.testing.expectEqualShapes(zml.Shape.init(.{ 2, 2 }, .f32), exe.output_shapes[0]);
}
