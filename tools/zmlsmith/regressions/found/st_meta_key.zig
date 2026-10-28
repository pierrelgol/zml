const std = @import("std");
const zml = @import("zml");

// parseSafetensors flattens __metadata__ keys into a 1024 byte stack buffer with
// appendSliceAssumeCapacity (zml/safetensors.zig:783-847 at 74c590b0): a metadata key of
// 1100 characters, or deep enough nesting, writes past the buffer (a panic in Debug,
// memory corruption in release) instead of failing the load. Found while auditing
// zmlsmith coverage of zml/safetensors.zig.
test "st_meta_key" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    try header.writer.writeAll("{\"__metadata__\":{\"");
    try header.writer.splatByteAll('k', 1100);
    try header.writer.writeAll("\":\"v\"},\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}");
    var file: std.Io.Writer.Allocating = .init(allocator);
    defer file.deinit();
    try file.writer.writeInt(u64, header.written().len, .little);
    try file.writer.writeAll(header.written());
    try file.writer.writeAll(&.{ 0, 0, 0, 0 });
    try tmp.dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = file.written() });
    const f = try tmp.dir.openFile(io, "m.safetensors", .{});
    defer f.close(io);
    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    zml.safetensors.parseSafetensors(allocator, io, &registry, f) catch {};
}
