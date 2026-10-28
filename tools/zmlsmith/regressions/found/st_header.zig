const std = @import("std");
const zml = @import("zml");

// parseSafetensors accesses the JSON header without checking its schema
// (`headers.object`, `.get("shape").?`, `.integer`, `@intCast`), so a malformed header
// panics instead of returning an error (zml/safetensors.zig:421-451 at 74c590b0). Here
// the tensor entry has no "shape". Found by reading the code while auditing zmlsmith.
test "st_header" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header = "{\"w\":{\"dtype\":\"F32\",\"data_offsets\":[0,4]}}";
    var bytes: [8 + header.len + 4]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = &bytes });

    const file = try tmp.dir.openFile(io, "m.safetensors", .{});
    defer file.close(io);
    var registry: zml.safetensors.TensorRegistry = .init(std.testing.allocator);
    defer registry.deinit();
    if (zml.safetensors.parseSafetensors(std.testing.allocator, io, &registry, file)) |_| {
        return error.TestUnexpectedResult;
    } else |_| {}
}
