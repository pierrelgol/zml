const std = @import("std");
const fx = @import("hf_fixture.zig");

// vfs HF insertTreeNode adds every file size to its parent directories unchecked
// (`parent.size += item.size`, vfs/hf.zig:634 at 74c590b0), and getSizeFromTree sums the
// top level files the same way (:577): a tree listing whose sizes overflow u64 panics with
// "integer overflow" instead of failing the open. The listing comes from the server.
// Found by zmlsmith's HF stand-in.
test "hf_size_ovf" {
    const allocator = std.testing.allocator;
    const tree = try fx.treeJson(allocator, &.{ .{ "d/a", std.math.maxInt(u64) }, .{ "d/b", 2 } });
    defer allocator.free(tree);
    const routes = try fx.routesJson(allocator, tree, "abcd");
    defer allocator.free(routes);
    var f: fx.Fixture = undefined;
    try f.init(routes);
    defer f.deinit();
    const vio = f.vfs.io();
    var file = std.Io.Dir.openFile(.cwd(), vio, "hf://org/model/d/a", .{ .mode = .read_only }) catch return;
    file.close(vio);
}
