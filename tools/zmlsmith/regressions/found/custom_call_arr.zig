const std = @import("std");
const zml = @import("zml");

// ops.customCall with a fixed-size array of output shapes ([2]zml.Shape) does not compile.
// The result mapping in zml/ops.zig:2507 reads `array_info.fields.len`, but
// std.builtin.Type.Array has `len` and no `fields`, so the array branch is a Zig compile error.
test "custom_call_arr" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        const out_shapes = [2]zml.Shape{ zml.Shape.init(.{4}, .f32), zml.Shape.init(.{4}, .f32) };
        fn forward(x: zml.Tensor) [2]zml.Tensor {
            return zml.ops.customCall("zml$custom_call_arr_probe", .{x}, out_shapes, .{}, .{ .has_side_effect = false });
        }
    };
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{4}, .f32));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();
    try std.testing.expectEqual(@as(usize, 2), exe.output_shapes.len);
}
