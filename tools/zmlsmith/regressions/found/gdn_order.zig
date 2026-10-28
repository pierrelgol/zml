const std = @import("std");
const zml = @import("zml");

// nn.GatedDeltaNet.forward validates its inputs by tags only (hasTags), but the loop body
// combines them with shape-matching mul/sub and dynamicUpdateSlice, so any axis order
// other than the documented .{ .s, .h, .k } / .{ .h, .v, .k } panics, e.g. a state given as .{ .v, .h, .k }
// (zml/nn.zig:1708-1717, :1734 at 74c590b0). Found by zmlsmith (gdn family).
test "gdn_order" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, a: zml.Tensor, b: zml.Tensor, s: zml.Tensor) zml.Tensor {
            return zml.nn.GatedDeltaNet.forward(q, k, v, a, b, .{ .s = s }, .scalar(2, .i32)).outputs;
        }
    };
    const q: zml.Tensor = .init(.{ .s = 2, .h = 1, .k = 2 }, .f32);
    const k: zml.Tensor = .init(.{ .s = 2, .h = 1, .k = 2 }, .f32);
    const v: zml.Tensor = .init(.{ .s = 2, .h = 1, .v = 2 }, .f32);
    const a: zml.Tensor = .init(.{ .s = 2, .h = 1 }, .f32);
    const b: zml.Tensor = .init(.{ .s = 2, .h = 1 }, .f32);
    const s: zml.Tensor = .init(.{ .v = 2, .h = 1, .k = 2 }, .f32);
    var exe = try zml.module.compile(allocator, io, Local.forward, .{ q, k, v, a, b, s }, platform, .{});
    exe.deinit();
}
