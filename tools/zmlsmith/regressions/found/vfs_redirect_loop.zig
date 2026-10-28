const std = @import("std");
const zml = @import("zml");

// vfs HTTP fetchSize follows redirects in its own loop with no limit (`redirect_behavior
// = .not_allowed`, then `continue`, vfs/http.zig:324-348 at 74c590b0): a server that
// redirects a file to itself, here with an empty Location, makes opening it hang
// forever. The server below redirects 200 times, then fails: the open follows every
// redirect. Found by zmlsmith parse-fuzz (vfs target).
var heads: std.atomic.Value(u32) = .init(0);
var done: std.atomic.Value(bool) = .init(false);

fn serve(server: *std.Io.net.Server) void {
    const io = std.testing.io;
    while (true) {
        const conn = server.accept(io) catch return;
        defer conn.close(io);
        if (done.load(.acquire)) return;
        var rbuf: [4096]u8 = undefined;
        var r = conn.reader(io, &rbuf);
        while (true) {
            const line = r.interface.takeDelimiterExclusive('\n') catch break;
            r.interface.toss(1);
            if (line.len <= 1) break;
        }
        const n = heads.fetchAdd(1, .monotonic);
        var wbuf: [256]u8 = undefined;
        var w = conn.writer(io, &wbuf);
        w.interface.writeAll(if (n < 200) "HTTP/1.1 302 Found\r\nConnection: close\r\nLocation: \r\nContent-Length: 0\r\n\r\n" else "HTTP/1.1 500 Stop\r\nConnection: close\r\nContent-Length: 0\r\n\r\n") catch {};
        w.interface.flush() catch {};
        if (n >= 200) return;
    }
}

test "vfs_redirect_loop" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var server = try addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    const t = try std.Thread.spawn(.{}, serve, .{&server});
    defer {
        done.store(true, .release);
        if (server.socket.address.connect(io, .{ .mode = .stream })) |c| c.close(io) else |_| {}
        t.join();
    }

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http: zml.io.VFS.HTTP = try .init(allocator, io, &client, .http);
    defer http.deinit();
    var vfs: zml.io.VFS = try .init(allocator, io);
    defer vfs.deinit();
    try vfs.register("http", http.io());
    var url_buf: [64]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/f.bin", .{server.socket.address.getPort()});
    if (std.Io.Dir.openFile(.cwd(), vfs.io(), url, .{ .mode = .read_only })) |file| file.close(vfs.io()) else |_| {}
    std.debug.print("redirects followed: {d}\n", .{heads.load(.monotonic) - 1});
    try std.testing.expect(heads.load(.monotonic) < 20);
}
