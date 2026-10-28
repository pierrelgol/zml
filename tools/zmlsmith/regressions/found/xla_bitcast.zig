const std = @import("std");
const zml = @import("zml");

// Upstream XLA: two chained narrowing bitcasts fail the HLO verifier after simplification.
test "xla_bitcast_chain" {
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x.bitCast(.u16).rename(.{ .bitcast = .b }).bitCast(.u8);
        }
    };
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .a = 2 }, .u32));
    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();
}
