const std = @import("std");
const zml = @import("zml");

// IEEE-754 multiplication gives 0 * inf = NaN (StableHLO multiply is IEEE multiplication,
// docs/spec.md "### multiply"). XLA's algebraic simplifier rewrites Mul(Convert(pred), x)
// into select(pred, x, 0) unconditionally (algebraic_simplifier.cc HandleMultiply, the
// "Mul(Convert(Pred), operand)" rule), so false * inf is 0 and false * NaN is 0.
test "bool_mul_inf" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn f(x: zml.Tensor) zml.Tensor {
            return x.convert(.f32).scale(std.math.inf(f32));
        }
    };
    const x: zml.Tensor = .init(.{2}, .bool);
    var exe = try platform.compileFn(allocator, io, Local.f, .{x}, .{});
    defer exe.deinit();
    var b: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, &.{ 0, 1 });
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.f, .{b});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    const v = got.constItems(f32);
    try std.testing.expect(std.math.isNan(v[0]));
    try std.testing.expect(std.math.isInf(v[1]));
}
