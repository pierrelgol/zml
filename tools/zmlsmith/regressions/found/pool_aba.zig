const std = @import("std");
const zml = @import("zml");

// mem.DynamicBufferPool keeps free blocks in a lock-free stack whose pop reads
// `head.next` before the compare-exchange, without an ABA guard (zml/mem.zig:303 at
// 74c590b0). When another thread pops head and its successor and pushes head back in
// between, the stale `next` is installed and a block still in use is handed out twice.
// Each thread here stamps the blocks it holds and checks the stamp before returning them.
// The race needs threads running at the same time: under load (a whole test suite in
// parallel) a round can pass, so rounds are repeated until the first collision.
// Found by reading the code while auditing zmlsmith coverage.

const threads = 8;
const iters = 100_000;

var pool: zml.mem.DynamicBufferPool = .init(threads * 2, 64);
var double_owned: std.atomic.Value(u32) = .init(0);

fn worker(id: u8) void {
    const io = std.testing.io;
    var held: [2][]u8 = undefined;
    for (0..iters) |_| {
        for (&held) |*b| {
            b.* = pool.get(std.heap.page_allocator, io) catch unreachable;
            @memset(b.*[8..], id);
        }
        for (held) |b| {
            for (b[8..]) |x| if (x != id) {
                _ = double_owned.fetchAdd(1, .monotonic);
                break;
            };
            pool.put(io, b);
        }
    }
}

test "pool_aba" {
    for (0..20) |_| {
        var ts: [threads]std.Thread = undefined;
        for (&ts, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, worker, .{@as(u8, @intCast(i + 1))});
        for (ts) |t| t.join();
        if (double_owned.load(.monotonic) > 0) break;
    }
    try std.testing.expectEqual(@as(u32, 0), double_owned.load(.monotonic));
}
