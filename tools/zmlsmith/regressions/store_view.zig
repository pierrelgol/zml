//! io.TensorStore.View over a checkpoint written here: prefixes, layers, parent, shapes,
//! counts and readers.

const std = @import("std");
const zml = @import("zml");

pub const keys = [_][]const u8{ "model.layers.10.w", "model.layers.2.w", "model.norm", "lm_head.weight_scale_inv" };

/// Writes `keys` as f32[2] tensors holding 4 * index + {0, 1} and returns the registry.
pub fn checkpoint(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.safetensors.TensorRegistry {
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

test "store view: prefixes, layers, shapes, counts and readers" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry = try checkpoint(allocator, io, tmp.dir);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const root = store.view();
    const layers = root.withPrefix("model").withPrefix("layers");
    try std.testing.expectEqualStrings("model.layers.", layers.prefix().?);
    try std.testing.expectEqual(@as(usize, 2), layers.count());
    try std.testing.expectEqual(@as(usize, 4), root.count());
    const l10 = layers.withLayer(10);
    try std.testing.expect(l10.getShape("w").?.eql(zml.Shape.init(.{2}, .f32)));
    try std.testing.expect(layers.withLayer(3).getShape("w") == null);
    try std.testing.expectEqualStrings("model.layers.", l10.parent().prefix().?);
    try std.testing.expect(root.withPrefix("model").parent().prefix() == null);
    try std.testing.expect(root.hasKey("model.norm"));
    try std.testing.expect(!root.hasKey("missing"));

    var buf: [64]u8 = undefined;
    var r = try layers.withLayer(2).getReader("w", io, &buf);
    var out: [8]u8 = undefined;
    try r.interface.readSliceAll(&out);
    const vals: [2]f32 = @bitCast(out);
    try std.testing.expectEqual([2]f32{ 4, 5 }, vals);
}
