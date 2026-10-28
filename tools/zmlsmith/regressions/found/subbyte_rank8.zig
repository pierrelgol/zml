const std = @import("std");
const zml = @import("zml");

// Compiler.repack packs a sub-byte result by reshaping it with its last axis split in
// two (zml/Compiler.zig:746 at 74c590b0): for a rank 8 result, Shape.MAX_RANK, the split
// shape has 9 axes and Shape.splitAxis panics with "attempt to unwrap error: Overflow"
// (zml/shape.zig:1330). Found by zmlsmith (seed 91633).
test "subbyte_rank8" {
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x;
        }
    };
    const x: zml.Tensor = .init(.{ 1, 1, 1, 1, 1, 1, 1, 2 }, .f4e2m1);
    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local.forward, .{x}, platform, .{});
    exe.deinit();
}
