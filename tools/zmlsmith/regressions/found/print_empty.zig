const std = @import("std");
const zml = @import("zml");

// Tensor.print of an empty tensor doesn't compile: XLA folds the empty operand into a
// constant and the SPMD partitioner rejects the side effecting zml$print custom call,
// "RET_CHECK failure (spmd_partitioner.cc:7508) hlo->has_sharding() Side-effect HLO must
// have sharding". Found by zmlsmith runtime coverage of Tensor.print.
fn run(comptime dt: zml.DataType, comptime dims: anytype) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Local = struct {
        fn f(x: zml.Tensor) zml.Tensor {
            x.print("x");
            return x;
        }
    };
    const x: zml.Tensor = .init(dims, dt);
    var exe = try platform.compileFn(allocator, io, Local.f, .{x}, .{});
    defer exe.deinit();
    const bytes = try allocator.alloc(u8, x.shape().byteSize());
    defer allocator.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 37 % 3);
    var b = try zml.Buffer.fromBytes(io, platform, x.shape(), .replicated, bytes);
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.f, .{b});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    try std.testing.expectEqualSlices(u8, bytes, got.constData());
}

test "print_empty" {
    try run(.f32, .{0});
}
