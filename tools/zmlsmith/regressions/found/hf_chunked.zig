const std = @import("std");
const fx = @import("hf_fixture.zig");

// vfs HF fetchTreeFromAPI reads the tree listing with `res.head.content_length.?`
// (vfs/hf.zig:491 at 74c590b0): an API answer without Content-Length (chunked, as a
// proxy or a gzip-less streaming answer sends it) panics with "attempt to use null value"
// on the first open instead of failing it. Found by zmlsmith's HF stand-in.
test "hf_chunked" {
    const routes =
        \\[["/api/models/", "raw", "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\n[]\r\n0\r\n\r\n"]]
    ;
    var f: fx.Fixture = undefined;
    try f.init(routes);
    defer f.deinit();
    const vio = f.vfs.io();
    var file = std.Io.Dir.openFile(.cwd(), vio, "hf://org/model/config.json", .{ .mode = .read_only }) catch return;
    file.close(vio);
}
