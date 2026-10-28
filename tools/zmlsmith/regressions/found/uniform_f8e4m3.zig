const std = @import("std");
const zml = @import("zml");

// Rng.uniform on an f8e4m3 (non-FN) tensor panics: mantissaSize (zml/dtype.zig:375) has no
// case for .f8e4m3, so the bit-masking step cannot pick a mantissa width. Every float dtype
// is a valid uniform() target, and the samples must lie in [min, max).
test "uniform_f8e4m3" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x0: zml.Tensor) [1]zml.Tensor {
            const r: zml.Tensor.Rng = .{ ._state = x0 };
            _, const x = r.uniform(zml.Shape.init(.{ .h = 4 }, .f8e4m3), .{ .min = 0, .max = 1 });
            return .{x};
        }
    };

    const x0: zml.Tensor = .fromShape(zml.Shape.init(.{2}, .u64));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x0}, platform, .{});
    defer exe.deinit();

    var b0: zml.Buffer = try .fromBytes(io, platform, x0.shape(), .replicated, &[_]u8{0} ** 16);
    defer b0.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{b0});
    defer for (&res) |*r| r.deinit();

    const got = try res[0].toSliceAlloc(allocator, io);
    defer got.free(allocator);
    for (got.constItems(u8)) |bits| {
        const v = zml.floats.Float8E4M3.toF32(@bitCast(bits));
        try std.testing.expect(v >= 0 and v <= 1);
    }
}
