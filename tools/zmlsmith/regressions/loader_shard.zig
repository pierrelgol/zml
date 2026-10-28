const std = @import("std");
const zml = @import("zml");

// io.Loader.load of sharded tensors: a checkpoint written here, tensors sharded on their
// first axis, on their second axis (non contiguous shards in the file) and replicated,
// loaded on the `data` mesh and read back; the bytes must be the file's.
const Model = struct { w0: zml.Tensor, w1: zml.Tensor, r: zml.Tensor };
const dims = [_][2]i64{ .{ 8, 6 }, .{ 6, 8 }, .{ 3, 5 } };
const names = [_][]const u8{ "w0", "w1", "r" };

fn writeCheckpoint(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, data: []u8) !usize {
    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    try header.writer.writeAll("{");
    var off: usize = 0;
    for (names, dims, 0..) |n, d, i| {
        const bytes: usize = @intCast(d[0] * d[1] * 4);
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ if (i > 0) "," else "", n, d[0], d[1], off, off + bytes });
        off += bytes;
    }
    try header.writer.writeAll("}");
    for (data[0..off], 0..) |*x, i| x.* = @truncate(i *% 131 +% 7);
    var file: std.Io.Writer.Allocating = .init(allocator);
    defer file.deinit();
    try file.writer.writeInt(u64, header.written().len, .little);
    try file.writer.writeAll(header.written());
    try file.writer.writeAll(data[0..off]);
    try dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = file.written() });
    return off;
}

fn roundTrip(mesh: zml.Sharding, chunk: usize) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var data: [4096]u8 = undefined;
    _ = try writeCheckpoint(allocator, io, tmp.dir, &data);
    var path_buf: [1024]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const path = try std.fmt.allocPrint(allocator, "{s}/m.safetensors", .{path_buf[0..n]});
    defer allocator.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const v = store.view();
    const model: Model = .{
        .w0 = v.createTensor("w0", .{ .a, .b }, .{ .a = .data }),
        .w1 = v.createTensor("w1", .{ .a, .b }, .{ .b = .data }),
        .r = v.createTensor("r", .{ .a, .b }, .replicated),
    };
    var buffers = try zml.mem.bufferize(allocator, Model, &model);
    var loader: zml.io.Loader = try .init(allocator, platform, .{ .parallelism = 4, .dma_chunks = 2, .dma_chunk_size = chunk });
    defer loader.deinit();
    try loader.load(io, Model, &model, &buffers, &store, &.{mesh}, .{});
    try loader.await(io);
    defer zml.Buffer.deinitAll(Model, &buffers);
    var off: usize = 0;
    inline for (@typeInfo(Model).@"struct".fields, dims) |f, d| {
        const bytes: usize = @intCast(d[0] * d[1] * 4);
        const got = try @field(buffers, f.name).toSliceAlloc(allocator, io);
        defer got.free(allocator);
        std.testing.expectEqualSlices(u8, data[off..][0..bytes], got.constData()) catch |err| {
            std.debug.print("tensor {s} (chunk {d}) differs\n", .{ f.name, chunk });
            return err;
        };
        off += bytes;
    }
}

test "loader_shard" {
    const mesh = try @constCast(zml.testing.env()).registerSharding("loader_shard", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }));
    try roundTrip(mesh, 64);
    try roundTrip(mesh, 4096);
}
