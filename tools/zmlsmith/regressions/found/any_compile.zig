const std = @import("std");
const zml = @import("zml");

// Doesn't compile: see the zmlsmith report.
test "any_compile" {
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x.any(.b);
        }
    };
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .a = 2, .b = 3 }, .i32));
    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();
}
