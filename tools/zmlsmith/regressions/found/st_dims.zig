const std = @import("std");
const zml = @import("zml");

// parseSafetensors copies the "shape" integers into the tensor Shape unchecked
// (zml/safetensors.zig:449-451 at 74c590b0): a negative dim, or dims whose product
// overflows i64, register fine, then TensorRegistry.totalBytes / Tensor.byteSize panic in
// Shape.count ("Can't count elements in shape with negative dimension", or "integer
// overflow" at zml/shape.zig:496). Found by zmlsmith parse-fuzz.
fn load(io: std.Io, dir: std.Io.Dir, header: []const u8) !void {
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(std.testing.allocator);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, header.len, .little);
    try bytes.appendSlice(std.testing.allocator, &len);
    try bytes.appendSlice(std.testing.allocator, header);
    try bytes.appendNTimes(std.testing.allocator, 0, 16);
    try dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = bytes.items });

    const file = try dir.openFile(io, "m.safetensors", .{});
    defer file.close(io);
    var registry: zml.safetensors.TensorRegistry = .init(std.testing.allocator);
    defer registry.deinit();
    zml.safetensors.parseSafetensors(std.testing.allocator, io, &registry, file) catch return;
    _ = registry.totalBytes();
}

test "st_dims" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try load(io, tmp.dir, "{\"w\":{\"dtype\":\"F32\",\"shape\":[9223372036854775807,2],\"data_offsets\":[0,16]}}");
    try load(io, tmp.dir, "{\"w\":{\"dtype\":\"BF16\",\"shape\":[-3],\"data_offsets\":[0,6]}}");
}
