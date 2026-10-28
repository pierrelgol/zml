const std = @import("std");
const zml = @import("zml");

// Rng.uniform on an integer shape samples floats in [min, max) and converts them, which
// truncates toward zero: with min = -4, -4 is never produced and 0 gets [-1, 1), twice
// its share. Found by zmlsmith (rng uniform_stats).
test "uniform_int" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const n = 4096;
    const Local = struct {
        fn forward(r: zml.Tensor.Rng) zml.Tensor {
            _, const x = r.uniform(zml.Shape.init(.{n}, .i32), .{ .min = -4, .max = 4 });
            return x;
        }
    };
    var exe = try zml.module.compile(allocator, io, Local.forward, .{zml.Tensor.Rng.init()}, platform, .{});
    defer exe.deinit();
    var state = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, 1234);
    defer state._state.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{state});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);

    var counts: [8]usize = @splat(0);
    for (got.items(i32)) |v| counts[@intCast(v + 4)] += 1;
    // Each of -4..3 should come out about n / 8 = 512 times.
    for (counts) |c| try std.testing.expect(c > 256 and c < 1024);
}
