const std = @import("std");
const zml = @import("zml");

// On CPU, maximum and the max reduction don't propagate NaN, as StableHLO specifies:
// maximum(1, NaN) is 1 and max([NaN, 1, NaN, -2]) is 1. XLA CPU enables fast min/max by
// default (xla_cpu_enable_fast_min_max) and ZML doesn't override it; with
// XLA_FLAGS=--xla_cpu_enable_fast_min_max=false both are NaN. Found by zmlsmith (ieee).
test "nan_minmax" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor, y: zml.Tensor) [2]zml.Tensor {
            return .{ x.maximum(y), x.max(0) };
        }
    };
    const s: zml.Shape = .init(.{4}, .f32);
    var exe = try zml.module.compile(allocator, io, Local.forward, .{ zml.Tensor.fromShape(s), zml.Tensor.fromShape(s) }, platform, .{});
    defer exe.deinit();

    const nan = std.math.nan(f32);
    var xb: zml.Buffer = try .fromBytes(io, platform, s, .replicated, std.mem.sliceAsBytes(&[_]f32{ nan, 1, nan, -2 }));
    defer xb.deinit();
    var yb: zml.Buffer = try .fromBytes(io, platform, s, .replicated, std.mem.sliceAsBytes(&[_]f32{ 0, nan, nan, 3 }));
    defer yb.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{ xb, yb });
    defer for (&res) |*r| r.deinit();

    const m = try res[0].toSliceAlloc(allocator, io);
    defer m.free(allocator);
    try std.testing.expect(std.math.isNan(m.items(f32)[1]));
    const r = try res[1].toSliceAlloc(allocator, io);
    defer r.free(allocator);
    try std.testing.expect(std.math.isNan(r.items(f32)[0]));
}
