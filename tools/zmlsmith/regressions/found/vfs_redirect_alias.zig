const std = @import("std");
const zml = @import("zml");

// vfs HTTP fetchSize writes the request URL into aux_buffer (== &redirect_buffer) and parses
// it there, so the base URI points into that buffer (vfs/http.zig:317, :321-323 at
// 74c590b0). On a 3xx a relative Location is copied to the start of the same buffer (:345)
// before Uri.resolveInPlace reads the base, which overwrites it. With `Location: g.bin` the
// base host is then garbage and the `catch unreachable` at :346 panics (InvalidHostName);
// other lengths trip "@memcpy arguments alias" inside std.Uri. Found by zmlsmith parse-fuzz
// (vfs target). Once fixed, the redirect to g.bin is followed and the 4 bytes are read.

/// Answers connection i with `answers[i]`, recording each request line.
const Server = struct {
    server: std.Io.net.Server,
    answers: []const []const u8,
    lines: [4][64]u8 = undefined,
    lens: [4]usize = .{ 0, 0, 0, 0 },

    fn run(s: *Server) void {
        const io = std.testing.io;
        for (s.answers, 0..) |answer, i| {
            const conn = s.server.accept(io) catch return;
            defer conn.close(io);
            var rbuf: [4096]u8 = undefined;
            var r = conn.reader(io, &rbuf);
            const first = r.interface.takeDelimiterExclusive('\n') catch return;
            r.interface.toss(1);
            s.lens[i] = @min(first.len, 64);
            @memcpy(s.lines[i][0..s.lens[i]], first[0..s.lens[i]]);
            while (true) {
                const line = r.interface.takeDelimiterExclusive('\n') catch return;
                r.interface.toss(1);
                if (line.len <= 1) break;
            }
            var wbuf: [4096]u8 = undefined;
            var w = conn.writer(io, &wbuf);
            w.interface.writeAll(answer) catch return;
            w.interface.flush() catch return;
        }
    }
};

test "vfs_redirect_alias" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var s: Server = .{ .server = try addr.listen(io, .{ .reuse_address = true }), .answers = &.{
        "HTTP/1.1 302 Found\r\nConnection: close\r\nLocation: g.bin\r\nContent-Length: 0\r\n\r\n",
        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 4\r\n\r\n",
        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 4\r\n\r\nabcd",
    } };
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
    const url = try std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/d/f.bin", .{s.server.socket.address.getPort()});
    var file = try std.Io.Dir.openFile(.cwd(), vio, url, .{ .mode = .read_only });
    defer file.close(vio);
    try std.testing.expectStringStartsWith(s.lines[1][0..s.lens[1]], "HEAD /d/g.bin ");
    var buf: [4]u8 = undefined;
    const n = try file.readPositional(vio, &.{&buf}, 0);
    try std.testing.expectEqualStrings("abcd", buf[0..n]);
}
