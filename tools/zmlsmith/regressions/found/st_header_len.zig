const std = @import("std");
const zml = @import("zml");

// parseSafetensors adds the untrusted u64 header length to BYTES_HEADER without an
// overflow check and allocates that many bytes, so an 8 byte file of 0xff panics with
// "integer overflow" (and a large but smaller length tries a huge allocation) instead of
// returning an error (zml/safetensors.zig:408-410 at 74c590b0). Found by zmlsmith
// parse-fuzz, reduced from 247 to 8 bytes.
test "st_header_len" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = &([_]u8{0xff} ** 8) });

    const file = try tmp.dir.openFile(io, "m.safetensors", .{});
    defer file.close(io);
    var registry: zml.safetensors.TensorRegistry = .init(std.testing.allocator);
    defer registry.deinit();
    if (zml.safetensors.parseSafetensors(std.testing.allocator, io, &registry, file)) |_| {
        return error.TestUnexpectedResult;
    } else |_| {}
}
