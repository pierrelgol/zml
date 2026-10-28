const std = @import("std");
const fx = @import("hf_fixture.zig");

// vfs HF lists a repository with one tree API request (`?recursive=true&limit=1000`,
// vfs/hf.zig:160, :461-508 at 74c590b0) and ignores the `Link: <...>; rel="next"` header
// with which the Hub paginates long listings (huggingface_hub follows it in
// list_repo_tree): files past the first page don't exist for ZML, so opening them fails
// with FileNotFound. Found by zmlsmith's HF stand-in.
test "hf_paginate" {
    const allocator = std.testing.allocator;
    const page2 = try fx.treeJson(allocator, &.{.{ "model-00002.safetensors", 4 }});
    defer allocator.free(page2);
    const page1 = try fx.treeJson(allocator, &.{.{ "model-00001.safetensors", 4 }});
    defer allocator.free(page1);
    const r2 = try std.fmt.allocPrint(allocator, "HTTP/1.1 200 OK\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ page2.len, page2 });
    defer allocator.free(r2);
    const r1 = try std.fmt.allocPrint(allocator, "HTTP/1.1 200 OK\r\nContent-Length: {d}\r\nLink: <https://huggingface.co/api/models/org/model/tree/main?recursive=true&cursor=p2>; rel=\"next\"\r\nConnection: close\r\n\r\n{s}", .{ page1.len, page1 });
    defer allocator.free(r1);
    const routes = try std.json.Stringify.valueAlloc(allocator, .{ .{ "cursor=p2", "raw", r2 }, .{ "/api/models/", "raw", r1 }, .{ "/resolve/", "file", "abcd" } }, .{});
    defer allocator.free(routes);
    var f: fx.Fixture = undefined;
    try f.init(routes);
    defer f.deinit();
    const vio = f.vfs.io();
    var file = try std.Io.Dir.openFile(.cwd(), vio, "hf://org/model/model-00002.safetensors", .{ .mode = .read_only });
    file.close(vio);
}
