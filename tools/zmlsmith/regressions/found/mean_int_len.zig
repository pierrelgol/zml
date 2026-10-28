const std = @import("std");
const zml = @import("zml");

// Tensor.mean divides the sum by the axis length with divByConst, which builds the length
// as a scalar of the tensor's dtype (zml/tensor.zig:1783 and :1227 at 74c590b0). For an
// integer dtype whose max is below the length, e.g. a u8 axis of 300, tracing panics with
// "integer does not fit in destination type" in DataType.constant. Found by zmlsmith
// (reduce family, seed 20967).
test "mean_int_len" {
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x.mean(.a);
        }
    };
    const x: zml.Tensor = .init(.{ .a = 300 }, .u8);
    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, Local.forward, .{x}, platform, .{});
    exe.deinit();
}
