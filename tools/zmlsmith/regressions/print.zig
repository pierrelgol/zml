//! Tensor.print in a program: it doesn't change the result, for the dtypes its FFI
//! callback handles (the others abort, found/print_dtypes.zig).

const std = @import("std");
const zml = @import("zml");

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

test "print: f32" {
    try run(.f32, .{ 2, 3 });
}

test "print: bf16" {
    try run(.bf16, .{4});
}

test "print: bool" {
    try run(.bool, .{3});
}

test "print: f8e4m3fn" {
    try run(.f8e4m3fn, .{4});
}

test "print: scalar" {
    try run(.i32, .{});
}

test "print: u16" {
    try run(.u16, .{5});
}
