//! vfs HF through the stand-in of found/hf_fixture.zig.

const std = @import("std");
const fx = @import("found/hf_fixture.zig");

test "vfs: HF list, stat and read through the stand-in" {
    const allocator = std.testing.allocator;
    const content = "0123456789abcdef";
    const tree = try fx.treeJson(allocator, &.{ .{ "config.json", 16 }, .{ "sub/model.safetensors", 16 } });
    defer allocator.free(tree);
    const routes = try fx.routesJson(allocator, tree, content);
    defer allocator.free(routes);
    var f: fx.Fixture = undefined;
    try f.init(routes);
    defer f.deinit();
    const vio = f.vfs.io();

    var file = try std.Io.Dir.openFile(.cwd(), vio, "hf://org/model/sub/model.safetensors", .{ .mode = .read_only });
    defer file.close(vio);
    try std.testing.expectEqual(16, (try file.stat(vio)).size);
    var buf: [16]u8 = undefined;
    const n = try file.readPositional(vio, &.{&buf}, 0);
    try std.testing.expectEqualStrings(content, buf[0..n]);

    var dir = try std.Io.Dir.openDir(.cwd(), vio, "hf://org/model", .{ .iterate = true });
    defer dir.close(vio);
    var it = dir.iterate();
    var names: usize = 0;
    while (try it.next(vio)) |_| names += 1;
    try std.testing.expectEqual(2, names);
}

// Random repository layouts: every file opens with its size and every directory lists
// exactly its children.
test "vfs: HF random trees" {
    const allocator = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0x4f1);
    const rng = prng.random();
    const names = [_][]const u8{ "a", "b", "config.json", "model.safetensors", "sub", "tok" };
    for (0..12) |_| {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        // Paths of depth 1 to 3; a name is a file or a directory, never both.
        var files: std.ArrayList(struct { []const u8, u64 }) = .empty;
        var dirs: std.StringHashMapUnmanaged(void) = .empty;
        const n = 1 + rng.uintLessThan(usize, 10);
        gen: for (0..n) |_| {
            const depth = 1 + rng.uintLessThan(usize, 3);
            var parts: [3][]const u8 = undefined;
            for (parts[0..depth]) |*p| p.* = names[rng.uintLessThan(usize, names.len)];
            const path = try std.mem.join(a, "/", parts[0..depth]);
            if (dirs.contains(path)) continue;
            for (files.items) |e| {
                if (std.mem.eql(u8, e[0], path)) continue :gen;
                if (std.mem.startsWith(u8, path, e[0]) and path[e[0].len] == '/') continue :gen;
            }
            for (1..depth) |d| try dirs.put(a, try std.mem.join(a, "/", parts[0..d]), {});
            try files.append(a, .{ path, 1 + rng.uintLessThan(u64, 1 << 20) });
        }
        const tree = try fx.treeJson(a, files.items);
        const routes = try fx.routesJson(a, tree, "x");
        var f: fx.Fixture = undefined;
        try f.init(routes);
        defer f.deinit();
        const vio = f.vfs.io();

        for (files.items) |e| {
            const p = try std.fmt.allocPrint(a, "hf://org/model/{s}", .{e[0]});
            var file = try std.Io.Dir.openFile(.cwd(), vio, p, .{ .mode = .read_only });
            defer file.close(vio);
            try std.testing.expectEqual(e[1], (try file.stat(vio)).size);
        }
        // The root and every directory list their direct children once.
        var dir_paths: std.ArrayList([]const u8) = .empty;
        try dir_paths.append(a, "");
        var it = dirs.keyIterator();
        while (it.next()) |k| try dir_paths.append(a, k.*);
        for (dir_paths.items) |d| {
            var expected: std.StringHashMapUnmanaged(void) = .empty;
            for (files.items) |e| {
                const rest = if (d.len == 0) e[0] else if (std.mem.startsWith(u8, e[0], d) and e[0].len > d.len and e[0][d.len] == '/') e[0][d.len + 1 ..] else continue;
                try expected.put(a, rest[0 .. std.mem.indexOfScalar(u8, rest, '/') orelse rest.len], {});
            }
            const p = try std.fmt.allocPrint(a, "hf://org/model/{s}", .{d});
            var dir = try std.Io.Dir.openDir(.cwd(), vio, p, .{ .iterate = true });
            defer dir.close(vio);
            var dit = dir.iterate();
            var count: usize = 0;
            while (try dit.next(vio)) |entry| {
                try std.testing.expect(expected.contains(entry.name));
                count += 1;
            }
            try std.testing.expectEqual(expected.count(), count);
        }
    }
}
