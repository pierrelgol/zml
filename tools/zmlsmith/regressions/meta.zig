//! zml.meta traversal on a struct mixing every container visit handles: arrays,
//! optionals, slices, unions, pointers, optional pointers, tuples and BoundedArray. Random
//! runtime shapes; leaves are numbered in declaration order and every traversal must see
//! them all, once, in that order.

const std = @import("std");
const zml = @import("zml");
const meta = zml.meta;

const Leaf = struct { id: u32 };

const Mixed = struct {
    a: Leaf,
    b: [3]Leaf,
    c: ?Leaf,
    d: []Leaf,
    e: union(enum) { x: Leaf, y: [2]Leaf, z: u8 },
    f: *Leaf,
    g: ?*Leaf,
    h: zml.stdx.BoundedArray(Leaf, 4),
    t: struct { Leaf, u8, Leaf },
    skip: u32 = 7,
};

const Built = struct {
    m: Mixed,
    expected: u32,
    // Storage for the slice and pointer fields.
    d_buf: [3]Leaf = undefined,
    f_val: Leaf = undefined,
    g_val: Leaf = undefined,
};

/// Fills `b` with a random runtime shape; ids follow declaration order.
fn build(b: *Built, rng: std.Random) void {
    var next: u32 = 0;
    const leaf = struct {
        fn f(n: *u32) Leaf {
            n.* += 1;
            return .{ .id = n.* - 1 };
        }
    }.f;
    b.m.a = leaf(&next);
    for (&b.m.b) |*l| l.* = leaf(&next);
    b.m.c = if (rng.boolean()) leaf(&next) else null;
    const dn = rng.uintLessThan(usize, 4);
    for (b.d_buf[0..dn]) |*l| l.* = leaf(&next);
    b.m.d = b.d_buf[0..dn];
    b.m.e = switch (rng.uintLessThan(u8, 3)) {
        0 => .{ .x = leaf(&next) },
        1 => .{ .y = .{ leaf(&next), leaf(&next) } },
        else => .{ .z = 3 },
    };
    b.f_val = leaf(&next);
    b.m.f = &b.f_val;
    if (rng.boolean()) {
        b.g_val = leaf(&next);
        b.m.g = &b.g_val;
    } else b.m.g = null;
    b.m.h = .empty;
    for (0..rng.uintLessThan(usize, 5)) |_| b.m.h.appendAssumeCapacity(leaf(&next));
    const t0 = leaf(&next);
    const t2 = leaf(&next);
    b.m.t = .{ t0, 9, t2 };
    b.m.skip = 7;
    b.expected = next;
}

fn idOf(l: Leaf) u32 {
    return l.id;
}

test "meta: count, collect and visit agree with the structure" {
    var prng: std.Random.DefaultPrng = .init(0x3e7a);
    const rng = prng.random();
    const allocator = std.testing.allocator;
    for (0..200) |_| {
        var b: Built = .{ .m = undefined, .expected = 0 };
        build(&b, rng);

        try std.testing.expectEqual(b.expected, meta.count(Leaf, &b.m));

        const ptrs = try meta.collectPtrs(Leaf, allocator, &b.m);
        defer allocator.free(ptrs);
        try std.testing.expectEqual(b.expected, ptrs.len);
        for (ptrs, 0..) |p, i| try std.testing.expectEqual(@as(u32, @intCast(i)), p.id);

        const ids = try meta.collectAlloc(idOf, {}, allocator, &b.m);
        defer allocator.free(ids);
        for (ids, 0..) |id, i| try std.testing.expectEqual(@as(u32, @intCast(i)), id);

        const buf = try allocator.alloc(u32, b.expected);
        defer allocator.free(buf);
        meta.collectBuf(idOf, {}, &b.m, buf);
        try std.testing.expectEqualSlices(u32, ids, buf);

        // A mutating visit reaches every leaf once.
        meta.visit(struct {
            fn cb(_: *const void, l: *Leaf) void {
                l.id += 1000;
            }
        }.cb, &{}, &b.m);
        const after = try meta.collectAlloc(idOf, {}, allocator, &b.m);
        defer allocator.free(after);
        for (after, 0..) |id, i| try std.testing.expectEqual(@as(u32, @intCast(i)) + 1000, id);
        try std.testing.expectEqual(@as(u32, 7), b.m.skip);
        try std.testing.expectEqual(@as(u8, 9), b.m.t[1]);

        // forEachVisit passes indices in order.
        var seen: u32 = 0;
        meta.forEachVisit(&b.m, *const Leaf, struct {
            fn f(i: usize, l: *const Leaf, s: *u32) void {
                if (l.id == i + 1000) s.* += 1;
            }
        }.f, .{&seen});
        try std.testing.expectEqual(b.expected, seen);
    }
}

test "meta: Contains" {
    try std.testing.expect(meta.Contains(Mixed, Leaf));
    try std.testing.expect(meta.Contains(?[]const Leaf, Leaf));
    try std.testing.expect(meta.Contains(?[]Leaf, Leaf));
    try std.testing.expect(meta.Contains(union(enum) { a: u8, b: [2]?*Leaf }, Leaf));
    try std.testing.expect(!meta.Contains(struct { a: u8, b: []const u16 }, Leaf));
    try std.testing.expect(!meta.Contains(*anyopaque, Leaf));
}
