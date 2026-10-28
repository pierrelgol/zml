const std = @import("std");
const zml = @import("zml");

// nn.resizeLinear1d computes the sample coordinates (arange * ratio, floor) in the image
// dtype unless ResizeOpts.precision is given (zml/nn.zig:1456-1460 at 74c590b0): for a
// bf16 image, coordinates above 256 are rounded to multiples of 2 or 4, so the
// interpolation reads the wrong pixels with weights outside [0, 1]. Resizing a bf16
// pattern of 1000 values in 0..6 to 333 gives an error of 9.06 at output 170. f16 images
// are affected too, from a few hundred pixels. Found by zmlsmith (resize family, an f16
// axis of 125 resized to 76: 8 of 76 values off, by up to 0.75).
test "resize_coords" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const n = 1000;
    const m = 333;
    const Local = struct {
        fn f(x: zml.Tensor) zml.Tensor {
            return zml.nn.resizeLinear1d(x, 0, m, .{});
        }
    };
    const x: zml.Tensor = .init(.{n}, .bf16);
    var exe = try platform.compileFn(allocator, io, Local.f, .{x}, .{});
    defer exe.deinit();
    // Values i % 7: exact in bf16, and neighbors differ.
    var data: [n]zml.floats.BFloat16 = undefined;
    for (&data, 0..) |*d, i| d.* = .fromF32(@floatFromInt(i % 7));
    var b = try zml.Buffer.fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(&data));
    defer b.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.f, .{b});
    defer res.deinit();
    const got = try res.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    var worst: f64 = 0;
    var at: usize = 0;
    for (got.constItems(zml.floats.BFloat16), 0..) |g, i| {
        const pos = @as(f64, @floatFromInt(i)) * n / m;
        const l: usize = @intFromFloat(@floor(pos));
        const w = pos - @floor(pos);
        const r = @min(l + 1, n - 1);
        const want = @as(f64, @floatFromInt(l % 7)) * (1 - w) + @as(f64, @floatFromInt(r % 7)) * w;
        const d = @abs(@as(f64, g.toF32()) - want);
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    std.debug.print("worst error {d} at output {d}\n", .{ worst, at });
    // bf16 rounding of the result alone is below 0.05 for values below 7.
    try std.testing.expect(worst < 0.1);
}
