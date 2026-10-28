const std = @import("std");
const zml = @import("zml");

// parseSafetensors never checks data_offsets: the end offset is ignored, and neither
// offset is compared with the file size or with the dtype and shape. A truncated or
// corrupted checkpoint registers fine and the loader later reads garbage or past the end
// (zml/safetensors.zig:445-458 at 74c590b0). Found by reading the code while auditing
// zmlsmith.
test "st_offsets" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // f32[4] is 16 bytes, but the offsets claim 0..1000 and the file holds 4 data bytes.
    const header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,1000]}}";
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
