const std = @import("std");
const zml = @import("zml");

// Upstream XLA: maximum(clamp(lo, x, hi), lo) is simplified to the clamp, which assumes
// clamp(...) >= lo. StableHLO defines clamp as min(max(x, lo), hi) without requiring
// lo <= hi: with x = hi = -100 and lo = -4 the clamp is -100 and the maximum must be -4.
// Found by zmlsmith from seed 1600886.
test "xla_clamp_max" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Local = struct {
        fn forward(x: zml.Tensor, lo: zml.Tensor) [2]zml.Tensor {
            const c = x.clamp(lo, x);
            return .{ c, c.maximum(lo) };
        }
    };
    const s: zml.Shape = .init(.{2}, .i64);
    var exe = try zml.module.compile(allocator, io, Local.forward, .{ zml.Tensor.fromShape(s), zml.Tensor.fromShape(s) }, platform, .{});
    defer exe.deinit();

    var xb: zml.Buffer = try .fromBytes(io, platform, s, .replicated, std.mem.sliceAsBytes(&[_]i64{ -100, 5 }));
    defer xb.deinit();
    var lb: zml.Buffer = try .fromBytes(io, platform, s, .replicated, std.mem.sliceAsBytes(&[_]i64{ -4, -4 }));
    defer lb.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{ xb, lb });
    defer for (&res) |*r| r.deinit();

    const c = try res[0].toSliceAlloc(allocator, io);
    defer c.free(allocator);
    try std.testing.expectEqualSlices(i64, &.{ -100, 5 }, c.items(i64));
    const m = try res[1].toSliceAlloc(allocator, io);
    defer m.free(allocator);
    try std.testing.expectEqualSlices(i64, &.{ -4, 5 }, m.items(i64));
}
