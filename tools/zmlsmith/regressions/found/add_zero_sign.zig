const std = @import("std");
const zml = @import("zml");

// IEEE-754 addition gives -0 + +0 = +0 (StableHLO add is IEEE addition, docs/spec.md "### add").
// XLA's algebraic simplifier rewrites A + 0 => A unconditionally (algebraic_simplifier.cc
// HandleAdd, not gated by fast math), so the sign of a negative zero input is kept.
test "add_zero_sign" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x.addConstant(0);
        }
    };
    const x: zml.Tensor = .init(.{2}, .f32);
    var exe = try platform.compileFn(allocator, io, Local.forward, .{x}, .{});
    defer exe.deinit();

    var b: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{ -0.0, 1.0 }));
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{b});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    const v = got.constItems(f32);
    try std.testing.expect(!std.math.signbit(v[0]));
    try std.testing.expectEqual(@as(f32, 1.0), v[1]);
}
