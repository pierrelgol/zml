//! mem.FixedBufferPool under contention: 8 threads get blocks, stamp them and check the
//! stamp before putting them back; a block handed to two owners shows up as a stamp
//! overwritten by another thread. (The same test finds the DynamicBufferPool race,
//! found/pool_aba.zig.)

const std = @import("std");
const zml = @import("zml");

const threads = 8;
const blocks = 12;
const block = 64;

var storage: [blocks * block]u8 = undefined;
var pool: zml.mem.FixedBufferPool = undefined;
var double_owned: std.atomic.Value(u32) = .init(0);

fn worker(id: u8) void {
    const io = std.testing.io;
    for (0..20_000) |_| {
        const b = pool.get(io) catch unreachable;
        @memset(b, id);
        for (b) |x| if (x != id) {
            _ = double_owned.fetchAdd(1, .monotonic);
            break;
        };
        pool.put(io, b);
    }
}

test "fixed buffer pool: no block has two owners" {
    pool = try .init(std.testing.allocator, &storage, blocks);
    defer pool.deinit(std.testing.allocator);
    var ts: [threads]std.Thread = undefined;
    for (&ts, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, worker, .{@as(u8, @intCast(i + 1))});
    for (ts) |t| t.join();
    try std.testing.expectEqual(@as(u32, 0), double_owned.load(.monotonic));
}
