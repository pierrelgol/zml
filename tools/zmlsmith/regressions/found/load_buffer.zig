const std = @import("std");
const zml = @import("zml");

// io.Loader.loadBuffer passes `&tensor`, the address of its by-value parameter, to the
// async load task (zml/io.zig:428 at 74c590b0). The task runs after loadBuffer returned
// and reads a dead stack slot: the tensor shape is garbage ("Tags, dims and partitioning
// have diverged! ... partitioning=140728831883192") and it panics. The same round trip
// through Loader.load passes (regressions/loader.zig). Found while auditing zmlsmith
// coverage of zml/io.zig.

const Entry = struct { name: []const u8, dtype: []const u8, dims: []const i64, bytes: usize };
const entries = [_]Entry{
    .{ .name = "w", .dtype = "F32", .dims = &.{ 3, 5 }, .bytes = 60 },
    .{ .name = "b", .dtype = "BF16", .dims = &.{7}, .bytes = 14 },
    .{ .name = "s", .dtype = "I8", .dims = &.{}, .bytes = 1 },
    .{ .name = "big", .dtype = "U8", .dims = &.{1000}, .bytes = 1000 },
};

fn writeCheckpoint(io: std.Io, dir: std.Io.Dir, data: []u8) !void {
    var header: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer header.deinit();
    try header.writer.writeAll("{");
    var off: usize = 0;
    for (entries, 0..) |e, i| {
        if (i > 0) try header.writer.writeAll(",");
        try header.writer.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ e.name, e.dtype });
        for (e.dims, 0..) |d, j| try header.writer.print("{s}{d}", .{ if (j > 0) "," else "", d });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ off, off + e.bytes });
        off += e.bytes;
    }
    try header.writer.writeAll("}");
    for (data, 0..) |*x, i| x.* = @truncate(i *% 131 +% 7);
    var file: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer file.deinit();
    try file.writer.writeInt(u64, header.written().len, .little);
    try file.writer.writeAll(header.written());
    try file.writer.writeAll(data);
    try dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = file.written() });
}

const Model = struct { w: zml.Tensor, b: zml.Tensor, s: zml.Tensor, big: zml.Tensor };

fn roundTrip(parallelism: usize, chunk: usize, use_load_buffer: bool) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var data: [1075]u8 = undefined;
    try writeCheckpoint(io, tmp.dir, &data);

    var path_buf: [1024]u8 = undefined;
    const dir_len = try tmp.dir.realPath(io, &path_buf);
    const path = try std.fmt.allocPrint(allocator, "{s}/m.safetensors", .{path_buf[0..dir_len]});
    defer allocator.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const v = store.view();
    const model: Model = .{
        .w = v.createTensor("w", null, .replicated),
        .b = v.createTensor("b", null, .replicated),
        .s = v.createTensor("s", null, .replicated),
        .big = v.createTensor("big", null, .replicated),
    };
    var buffers = try zml.mem.bufferize(allocator, Model, &model);
    var loader: zml.io.Loader = try .init(allocator, platform, .{ .parallelism = parallelism, .dma_chunks = 2, .dma_chunk_size = chunk });
    defer loader.deinit();
    const shardings: []const zml.Sharding = &.{platform.replicated_sharding};
    if (use_load_buffer) {
        inline for (@typeInfo(Model).@"struct".fields) |f| {
            loader.loadBuffer(io, &@field(buffers, f.name), @field(model, f.name), &store, shardings, .{});
        }
    } else {
        try loader.load(io, Model, &model, &buffers, &store, shardings, .{});
    }
    try loader.await(io);
    defer zml.Buffer.deinitAll(Model, &buffers);

    var off: usize = 0;
    inline for (@typeInfo(Model).@"struct".fields, entries) |f, e| {
        const got = try @field(buffers, f.name).toSliceAlloc(allocator, io);
        defer got.free(allocator);
        try std.testing.expectEqualSlices(u8, data[off..][0..e.bytes], got.constData());
        off += e.bytes;
    }
}

test "loader_rt_load_buffer" {
    try roundTrip(1, 64, true);
    try roundTrip(4, 64, true);
}
