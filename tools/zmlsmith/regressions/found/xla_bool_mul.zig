const std = @import("std");
const zml = @import("zml");

// Upstream XLA CPU: convert(bool) times a negative constant gives +0 for false, where
// IEEE multiplication gives 0 * -3 = -0; XLA probably rewrites the product of a converted
// predicate into a select with a literal 0. Found by zmlsmith (seed 310246).
test "xla_bool_mul" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Local = struct {
        fn f(x: zml.Tensor) zml.Tensor {
            return x.convert(.f64).scale(-3);
        }
    };
    const x: zml.Tensor = .init(.{2}, .bool);
    var exe = try platform.compileFn(allocator, io, Local.f, .{x}, .{});
    defer exe.deinit();
    var b = try zml.Buffer.fromBytes(io, platform, x.shape(), .replicated, &.{ 0, 1 });
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.f, .{b});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    const v = got.constItems(f64);
    std.debug.print("got {any}\n", .{v});
    try std.testing.expect(v[0] == 0 and std.math.signbit(v[0]));
    try std.testing.expectEqual(@as(f64, -3), v[1]);
}
