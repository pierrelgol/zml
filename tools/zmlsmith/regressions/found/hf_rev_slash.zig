const std = @import("std");
const fx = @import("hf_fixture.zig");

// vfs HF Repo.parse splits the URI on '/' before looking for '@' (vfs/hf.zig:221-233 at
// 74c590b0): a revision containing a slash, such as the Hub's pull request refs
// "refs/pr/1" or "refs/convert/parquet", becomes revision "refs" and the rest is taken as
// the file path, and the revision is put in URLs without percent-encoding. Opening
// hf://org/model@refs/pr/1/config.json requests the tree of revision "refs" and fails.
// Found by zmlsmith's HF stand-in.
test "hf_rev_slash" {
    const allocator = std.testing.allocator;
    const tree = try fx.treeJson(allocator, &.{.{ "config.json", 4 }});
    defer allocator.free(tree);
    const resp = try std.fmt.allocPrint(allocator, "HTTP/1.1 200 OK\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ tree.len, tree });
    defer allocator.free(resp);
    const routes = try std.json.Stringify.valueAlloc(allocator, .{.{ "/tree/refs%2Fpr%2F1", "raw", resp }}, .{});
    defer allocator.free(routes);
    var f: fx.Fixture = undefined;
    try f.init(routes);
    defer f.deinit();
    const vio = f.vfs.io();
    var file = try std.Io.Dir.openFile(.cwd(), vio, "hf://org/model@refs/pr/1/config.json", .{ .mode = .read_only });
    file.close(vio);
}
