const std = @import("std");
const zml = @import("zml");

// Tensor.print lowers to the zml$print FFI callback, which maps the buffer dtype with an
// exhaustive enum and a switch (pjrt/ffi.zig:230-256, zml/platform.zig:927-950 at
// 74c590b0). The enum has no S4/U4/S2/U2/F8E8M0FNU entries ("switch on corrupt value"
// in the XLA runtime thread) and the switch sends f8e3m4, f8e4m3 and f4e2m1 to
// unreachable: printing a tensor of any of these dtypes aborts the process. Found by
// zmlsmith runtime coverage of Tensor.print.

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

test "print_dtypes" {
    try run(.i4, .{4});
}
