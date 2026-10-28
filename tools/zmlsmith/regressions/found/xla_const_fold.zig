const std = @import("std");
const zml = @import("zml");

// Upstream XLA CPU (probably the algebraic simplifier): chained float additions of
// constants are folded, `x.subConstant(c1).addConstant(c2)` is computed as
// `x + (c2 - c1)` with one rounding instead of two, so `(x - 0.5) + 0.5` returns x
// bit for bit. StableHLO rounds after each op. Found by zmlsmith (seed 120975).
fn check(c1: f32, c2: f32) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Local = struct {
        fn forward(x: zml.Tensor, a: f32, b: f32) zml.Tensor {
            return x.subConstant(a).addConstant(b);
        }
    };
    const x: zml.Tensor = .init(.{4}, .f32);
    var exe = try zml.module.compile(allocator, io, Local.forward, .{ x, c1, c2 }, platform, .{});
    defer exe.deinit();
    const xs = [_]f32{ -1.5498009, 0.1, 1.0e7 + 1, 3.3333333 };
    var b: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(&xs));
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{b});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    for (xs, got.constItems(f32)) |v, g| {
        const want = (v - c1) + c2;
        if (want != g) std.debug.print("c1={d} c2={d} x={d}: want {d}, got {d}\n", .{ c1, c2, v, want, g });
        try std.testing.expectEqual(want, g);
    }
}

test "xla_const_fold_cancel" {
    try check(0.5, 0.5);
}

test "xla_const_fold_other" {
    try check(0.5, 0.25);
}
