//! Shared by hf_vfs.zig and the found/hf_* reproducers (not a test itself): vfs HF
//! against a local stand-in for huggingface.co. The HF backend hardcodes
//! https://huggingface.co, so the client is pointed at a local https proxy (python3) that
//! answers as huggingface.co with a self-signed test certificate, trusted only by this
//! client.

const std = @import("std");
const zml = @import("zml");

/// Test certificate for huggingface.co (P-256, self-signed, CA, valid until 2126).
pub const cert_pem =
    \\-----BEGIN CERTIFICATE-----
    \\MIIBkzCCATmgAwIBAgIUfBF+wcUB4HREkXWKxdTRnzk3p+AwCgYIKoZIzj0EAwIw
    \\GTEXMBUGA1UEAwwOaHVnZ2luZ2ZhY2UuY28wIBcNMjYxMDI1MjEyNDQ1WhgPMjEy
    \\NjEwMDEyMTI0NDVaMBkxFzAVBgNVBAMMDmh1Z2dpbmdmYWNlLmNvMFkwEwYHKoZI
    \\zj0CAQYIKoZIzj0DAQcDQgAEhy77fejpiH+Ubo+BajYme8kf+R/yRe8qq+shQdfp
    \\Md9jBa7T6bTp2O3ZEYY6udhUIeE7+8UGuFEkf5zmbjzvNqNdMFswHQYDVR0OBBYE
    \\FCubPFTMZ9VqGN9bQF29hRmxvJh4MBkGA1UdEQQSMBCCDmh1Z2dpbmdmYWNlLmNv
    \\MA8GA1UdEwEB/wQFMAMBAf8wDgYDVR0PAQH/BAQDAgKEMAoGCCqGSM49BAMCA0gA
    \\MEUCIQDYoPc6o0i5QUacseq2OGWsmpP+d6apysEr5M18u1Z9JwIgaLCec1X54SUw
    \\uEH9qjVqla5ddOHUmdIRs5z+ew63esM=
    \\-----END CERTIFICATE-----
    \\
;
pub const key_pem =
    \\-----BEGIN PRIVATE KEY-----
    \\MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgraph8ZflqHA25mgL
    \\z0DRsDkq9+LPRvg3bTB/mtkvE/GhRANCAASHLvt96OmIf5Ruj4FqNiZ7yR/5H/JF
    \\7yqr6yFB1+kx32MFrtPptOnY7dkRhjq52FQh4Tv7xQa4USR/nOZuPO82
    \\-----END PRIVATE KEY-----
    \\
;

/// TLS proxy (std.http.Client does the TLS handshake with an https proxy itself, as the
/// proxied host, then sends CONNECT inside it). Argv: CERT KEY ROUTES; prints its port. ROUTES is a
/// JSON list of [path_substring, kind, payload]: "raw" is a whole HTTP response, "file"
/// is file content served with Range support.
pub const server_py =
    \\import json, socket, ssl, sys, threading
    \\cert, key, routes_path = sys.argv[1:4]
    \\routes = json.load(open(routes_path))
    \\ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    \\ctx.load_cert_chain(cert, key)
    \\def read_head(f):
    \\    lines = []
    \\    while True:
    \\        line = f.readline()
    \\        if not line or line in (b"\r\n", b"\n"):
    \\            return lines
    \\        lines.append(line.decode("latin-1").rstrip("\r\n"))
    \\def respond(path, headers):
    \\    for sub, kind, payload in routes:
    \\        if sub not in path:
    \\            continue
    \\        if kind == "raw":
    \\            return payload.encode("latin-1")
    \\        data = payload.encode("latin-1")
    \\        rng = headers.get("range")
    \\        if rng and rng.startswith("bytes="):
    \\            a, b = rng[6:].split("-")
    \\            a, b = int(a), min(int(b), len(data) - 1)
    \\            body = data[a : b + 1]
    \\            head = f"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {a}-{b}/{len(data)}\r\n"
    \\        else:
    \\            body, head = data, "HTTP/1.1 200 OK\r\n"
    \\        return (head + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    \\    return b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    \\def serve(conn):
    \\    try:
    \\        tls = ctx.wrap_socket(conn, server_side=True)
    \\        f = tls.makefile("rb")
    \\        head = read_head(f)
    \\        if not head or not head[0].startswith("CONNECT"):
    \\            return tls.close()
    \\        tls.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
    \\        head = read_head(f)
    \\        if head:
    \\            headers = {}
    \\            for h in head[1:]:
    \\                k, _, v = h.partition(":")
    \\                headers[k.strip().lower()] = v.strip()
    \\            tls.sendall(respond(head[0].split(" ")[1], headers))
    \\        tls.close()
    \\    except Exception:
    \\        conn.close()
    \\srv = socket.socket()
    \\srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    \\srv.bind(("127.0.0.1", 0))
    \\srv.listen(64)
    \\print(srv.getsockname()[1], flush=True)
    \\while True:
    \\    c, _ = srv.accept()
    \\    threading.Thread(target=serve, args=(c,), daemon=True).start()
    \\
;

/// A running stand-in and an HF VFS registered as "hf" whose client goes through it.
pub const Fixture = struct {
    tmp: std.testing.TmpDir,
    child: std.process.Child,
    proxy: std.http.Client.Proxy,
    client: std.http.Client,
    hf: zml.io.VFS.HF,
    vfs: zml.io.VFS,

    /// `routes` is the ROUTES JSON described above.
    pub fn init(f: *Fixture, routes: []const u8) !void {
        const allocator = std.testing.allocator;
        const io = std.testing.io;
        f.tmp = std.testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        inline for (.{ .{ "cert.pem", cert_pem }, .{ "key.pem", key_pem }, .{ "server.py", server_py } }) |e| {
            try f.tmp.dir.writeFile(io, .{ .sub_path = e[0], .data = e[1] });
        }
        try f.tmp.dir.writeFile(io, .{ .sub_path = "routes.json", .data = routes });
        var buf: [4][std.fs.max_path_bytes]u8 = undefined;
        var paths: [4][]const u8 = undefined;
        for ([_][]const u8{ "server.py", "cert.pem", "key.pem", "routes.json" }, 0..) |name, i| {
            const n = try f.tmp.dir.realPathFile(io, name, &buf[i]);
            paths[i] = buf[i][0..n];
        }
        f.child = try std.process.spawn(io, .{ .argv = &.{ "python3", paths[0], paths[1], paths[2], paths[3] }, .stdout = .pipe });
        errdefer f.child.kill(io);
        var rbuf: [64]u8 = undefined;
        var r = f.child.stdout.?.reader(io, &rbuf);
        const line = try r.interface.takeDelimiterExclusive('\n');
        const port = try std.fmt.parseInt(u16, std.mem.trim(u8, line, " \r"), 10);

        f.proxy = .{ .protocol = .tls, .host = try .init("127.0.0.1"), .authorization = null, .port = port, .supports_connect = true };
        f.client = .{ .allocator = allocator, .io = io };
        errdefer f.client.deinit();
        f.client.https_proxy = &f.proxy;
        const now = std.Io.Clock.real.now(io);
        try f.client.ca_bundle.addCertsFromFilePathAbsolute(allocator, io, now, paths[1]);
        f.client.now = now;

        f.hf = try .init(allocator, io, &f.client, null, .{ .read_pool = .{ .chunk_size = 1 << 20, .num_workers = 2, .queue_capacity = 8, .max_retries = 0, .retry_initial_delay = .fromMilliseconds(1), .retry_max_delay = .fromMilliseconds(1) } });
        errdefer f.hf.deinit();
        f.vfs = try .init(allocator, io);
        try f.vfs.register("hf", f.hf.io());
    }

    pub fn deinit(f: *Fixture) void {
        const io = std.testing.io;
        f.vfs.deinit();
        f.hf.deinit();
        f.client.deinit();
        f.child.kill(io);
        f.tmp.cleanup();
    }
};

/// A HF tree API answer listing `files` as {path, size} pairs.
pub fn treeJson(allocator: std.mem.Allocator, files: []const struct { []const u8, u64 }) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    try out.writer.writeByte('[');
    for (files, 0..) |e, i| {
        if (i > 0) try out.writer.writeByte(',');
        try out.writer.print("{{\"type\":\"file\",\"oid\":\"x\",\"path\":\"{s}\",\"size\":{d}}}", .{ e[0], e[1] });
    }
    try out.writer.writeByte(']');
    return out.toOwnedSlice();
}

/// ROUTES serving `tree` with a Content-Length and `content` for every file.
pub fn routesJson(allocator: std.mem.Allocator, tree: []const u8, content: []const u8) ![]u8 {
    const resp = try std.fmt.allocPrint(allocator, "HTTP/1.1 200 OK\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ tree.len, tree });
    defer allocator.free(resp);
    return std.json.Stringify.valueAlloc(allocator, .{ .{ "/api/models/", "raw", resp }, .{ "/resolve/", "file", content } }, .{});
}
