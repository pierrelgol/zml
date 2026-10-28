const std = @import("std");
const zml = @import("zml");

// io.TensorStore.View.hasKey matches keys by prefix (`std.mem.startsWith(k, key)`,
// zml/io.zig:157-163 at 74c590b0): with only "lm_head.weight_scale_inv" in the checkpoint,
// hasKey("lm_head.weight") is true, and so is hasKey("model.layers.1") with only
// "model.layers.10.*". Callers use it to test for a tensor before creating it
// (examples/llm/models/qwen3_5/model.zig:158). Found while adding TensorStore coverage
// to zmlsmith.
const keys = [_][]const u8{ "model.layers.10.w", "model.layers.2.w", "model.norm", "lm_head.weight_scale_inv" };

/// Writes `keys` as f32[2] tensors holding 4 * index + {0, 1} and returns the registry.
fn checkpoint(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.safetensors.TensorRegistry {
    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    try header.writer.writeAll("{");
    for (keys, 0..) |k, i| try header.writer.print("{s}\"{s}\":{{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[{d},{d}]}}", .{ if (i > 0) "," else "", k, 8 * i, 8 * i + 8 });
    try header.writer.writeAll("}");
    var file: std.Io.Writer.Allocating = .init(allocator);
    defer file.deinit();
    try file.writer.writeInt(u64, header.written().len, .little);
    try file.writer.writeAll(header.written());
    for (0..keys.len) |i| for (0..2) |j| try file.writer.writeAll(std.mem.asBytes(&@as(f32, @floatFromInt(4 * i + j))));
    try dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = file.written() });
    var path_buf: [1024]u8 = undefined;
    const n = try dir.realPath(io, &path_buf);
    const path = try std.fmt.allocPrint(allocator, "{s}/m.safetensors", .{path_buf[0..n]});
    defer allocator.free(path);
    return .fromPath(allocator, io, path);
}

test "store_has_key" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry = try checkpoint(allocator, io, tmp.dir);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    try std.testing.expect(!store.view().hasKey("lm_head.weight"));
    try std.testing.expect(!store.view().hasKey("model.layers.1"));
}
