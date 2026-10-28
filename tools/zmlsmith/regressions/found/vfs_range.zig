const std = @import("std");
const zml = @import("zml");

// vfs HTTP (and parallel_read, used by HF, S3 and GCS) parseContentRange slices the
// header between the first ' ', '-' and '/' without checking their order: a malformed
// "Content-Range: 0-15 bytes/16" from a server panics with "start index is larger than
// end index" instead of returning null (vfs/http.zig:416-425, vfs/parallel_read.zig:161
// at 74c590b0). Found while adding vfs coverage to zmlsmith.

/// Serves `head` to HEAD requests and `get` to the others, `n` connections.
const Server = struct {
    server: std.Io.net.Server,
    head: []const u8,
    get: []const u8,
    n: usize,

    fn run(s: *Server) void {
        const io = std.testing.io;
        for (0..s.n) |_| {
            const conn = s.server.accept(io) catch return;
            defer conn.close(io);
            var rbuf: [4096]u8 = undefined;
            var r = conn.reader(io, &rbuf);
            const first = r.interface.takeDelimiterExclusive('\n') catch return;
            const is_head = std.mem.startsWith(u8, first, "HEAD");
            r.interface.toss(1);
            while (true) {
                const line = r.interface.takeDelimiterExclusive('\n') catch return;
                r.interface.toss(1);
                if (line.len <= 1) break;
            }
            var wbuf: [4096]u8 = undefined;
            var w = conn.writer(io, &wbuf);
            w.interface.writeAll(if (is_head) s.head else s.get) catch return;
            w.interface.flush() catch return;
        }
    }
};

fn readThroughVfs(head: []const u8, get: []const u8, n: usize) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var s: Server = .{ .server = try addr.listen(io, .{ .reuse_address = true }), .head = head, .get = get, .n = n };
    defer s.server.deinit(io);
    const t = try std.Thread.spawn(.{}, Server.run, .{&s});
    defer t.join();

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http: zml.io.VFS.HTTP = try .init(allocator, io, &client, .http);
    defer http.deinit();
    var vfs: zml.io.VFS = try .init(allocator, io);
    defer vfs.deinit();
    try vfs.register("http", http.io());
    const vio = vfs.io();
    var url_buf: [64]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/f.bin", .{s.server.socket.address.getPort()});
    var file = std.Io.Dir.openFile(.cwd(), vio, url, .{ .mode = .read_only }) catch return;
    defer file.close(vio);
    var buf: [16]u8 = undefined;
    _ = file.readPositional(vio, &.{&buf}, 0) catch return;
}

test "vfs_range" {
    try readThroughVfs("HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 16\r\n\r\n", "HTTP/1.1 206 Partial Content\r\nConnection: close\r\nContent-Range: 0-15 bytes/16\r\nContent-Length: 16\r\n\r\n0123456789abcdef", 2);
}
