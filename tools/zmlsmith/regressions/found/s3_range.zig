const std = @import("std");
const zml = @import("zml");

// vfs parallel_read.parseContentRange, used by the S3, GCS and HF backends, has the same
// unchecked slicing as the HTTP one (vfs/parallel_read.zig:161-171 at 74c590b0): an S3
// endpoint answering a ranged GET with "Content-Range: 0-15 bytes/16" panics with "start
// index is larger than end index" in a read pool worker. Found while adding vfs coverage
// to zmlsmith, against a local endpoint.

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

fn readThroughS3(head: []const u8, get: []const u8, n: usize) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var s: Server = .{ .server = try addr.listen(io, .{ .reuse_address = true }), .head = head, .get = get, .n = n };
    defer s.server.deinit(io);
    const t = try std.Thread.spawn(.{}, Server.run, .{&s});
    defer t.join();

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var endpoint_buf: [64]u8 = undefined;
    const endpoint = try std.fmt.bufPrint(&endpoint_buf, "http://127.0.0.1:{d}", .{s.server.socket.address.getPort()});
    var s3: zml.io.VFS.S3 = try .init(allocator, io, &client, .{ .access_key = "a", .secret_key = "b", .endpoint_url = endpoint, .region = "us-east-1" }, .{ .read_pool = .{ .chunk_size = 16, .num_workers = 1, .queue_capacity = 4, .max_retries = 0, .retry_initial_delay = .fromMilliseconds(1), .retry_max_delay = .fromMilliseconds(1) } });
    defer s3.deinit();
    var vfs: zml.io.VFS = try .init(allocator, io);
    defer vfs.deinit();
    try vfs.register("s3", s3.io());
    const vio = vfs.io();
    var file = std.Io.Dir.openFile(.cwd(), vio, "s3://bucket/f.bin", .{ .mode = .read_only }) catch return;
    defer file.close(vio);
    var buf: [16]u8 = undefined;
    _ = file.readPositional(vio, &.{&buf}, 0) catch return;
}

test "s3_range" {
    try readThroughS3("HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 16\r\n\r\n", "HTTP/1.1 206 Partial Content\r\nConnection: close\r\nContent-Range: 0-15 bytes/16\r\nContent-Length: 16\r\n\r\n0123456789abcdef", 2);
}
