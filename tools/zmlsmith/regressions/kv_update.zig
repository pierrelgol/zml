//! paged_attention.KvCache.update (stablehlo backend) against a host scatter: token t
//! with slot s writes its new k and v rows at page s / page_size, offset s % page_size.
//! Slots are unique; slot -1 marks a padding token, which must leave the cache untouched.

const std = @import("std");
const zml = @import("zml");
const paged = zml.attention.paged_attention;

const ps = 4;
const num_pages = 6;
const hkv = 2;
const hd = 3;
const tokens = 7;

const Fns = struct {
    fn update(k: zml.Tensor, v: zml.Tensor, nk: zml.Tensor, nv: zml.Tensor, slots: zml.Tensor) [2]zml.Tensor {
        const cache: paged.KvCache = .{ .split = .{ .k = k, .v = v } };
        const r = cache.update(nk, nv, slots, ps, .stablehlo);
        return .{ r.split.k, r.split.v };
    }
};

test "kv cache update: host scatter" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const k: zml.Tensor = .init(.{ .page = num_pages, .k_chunk = ps, .hkv = hkv, .hd = hd }, .f32);
    const v: zml.Tensor = .init(.{ .page = num_pages, .k_chunk = ps, .hkv = hkv, .hd = hd }, .f32);
    const nk: zml.Tensor = .init(.{ .b = tokens, .hkv = hkv, .hd = hd }, .f32);
    const nv: zml.Tensor = .init(.{ .b = tokens, .hkv = hkv, .hd = hd }, .f32);
    const slots: zml.Tensor = .init(.{ .b = tokens }, .i32);
    var exe = try platform.compileFn(allocator, io, Fns.update, .{ k, v, nk, nv, slots }, .{});
    defer exe.deinit();

    var prng: std.Random.DefaultPrng = .init(77);
    const rng = prng.random();
    const row = hkv * hd;
    for (0..20) |round| {
        var kh: [num_pages * ps * row]f32 = undefined;
        var vh: [num_pages * ps * row]f32 = undefined;
        var nkh: [tokens * row]f32 = undefined;
        var nvh: [tokens * row]f32 = undefined;
        for (&kh, &vh) |*a, *b| {
            a.* = rng.floatNorm(f32);
            b.* = rng.floatNorm(f32);
        }
        for (&nkh, &nvh) |*a, *b| {
            a.* = rng.floatNorm(f32);
            b.* = rng.floatNorm(f32);
        }
        var all: [num_pages * ps]i32 = undefined;
        for (&all, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, &all);
        var sl: [tokens]i32 = undefined;
        for (&sl, 0..) |*x, i| x.* = if (round % 2 == 1 and rng.uintLessThan(u8, 3) == 0) -1 else all[i];

        var want_k = kh;
        var want_v = vh;
        for (sl, 0..) |s, t| if (s >= 0) {
            const o: usize = @intCast(s);
            @memcpy(want_k[o * row ..][0..row], nkh[t * row ..][0..row]);
            @memcpy(want_v[o * row ..][0..row], nvh[t * row ..][0..row]);
        };

        var bk = try zml.Buffer.fromBytes(io, platform, k.shape(), .replicated, std.mem.sliceAsBytes(&kh));
        defer bk.deinit();
        var bv = try zml.Buffer.fromBytes(io, platform, v.shape(), .replicated, std.mem.sliceAsBytes(&vh));
        defer bv.deinit();
        var bnk = try zml.Buffer.fromBytes(io, platform, nk.shape(), .replicated, std.mem.sliceAsBytes(&nkh));
        defer bnk.deinit();
        var bnv = try zml.Buffer.fromBytes(io, platform, nv.shape(), .replicated, std.mem.sliceAsBytes(&nvh));
        defer bnv.deinit();
        var bs = try zml.Buffer.fromBytes(io, platform, slots.shape(), .replicated, std.mem.sliceAsBytes(&sl));
        defer bs.deinit();
        var res = try zml.testing.autoCall(allocator, io, &exe, Fns.update, .{ bk, bv, bnk, bnv, bs });
        defer for (&res) |*r| r.deinit();
        const gk = try res[0].toSliceAlloc(allocator, io);
        defer gk.free(allocator);
        const gv = try res[1].toSliceAlloc(allocator, io);
        defer gv.free(allocator);
        if (!std.mem.eql(f32, gk.constItems(f32), &want_k) or !std.mem.eql(f32, gv.constItems(f32), &want_v)) {
            std.debug.print("round {d}: slots {any}\n", .{ round, sl });
            for (gk.constItems(f32), want_k, 0..) |g, w, i| if (g != w) {
                std.debug.print("k slot {d} lane {d}: got {d}, want {d}\n", .{ i / row, i % row, g, w });
                break;
            };
            return error.TestUnexpectedResult;
        }
    }
}
