const std = @import("std");
const zml = @import("zml");
const paged = zml.attention.paged_attention;

// Random configurations of the stablehlo (CPU) paged attention against the host reference
// of found/paged_attn.zig, copied here: random context lengths, page tables and decode or page aligned
// prefill queries per sequence. Empty sequences, prefills that end in a partial page and
// fewer tokens than a page are known bugs (found/paged_attn.zig, found/paged_small.zig)
// and are not generated. Only the query rows in use are compared.

const ps = 4; // page_size
const num_pages = 8;
const max_pages = 4;
const hd = 4;

const Case = struct {
    q_start: []const i32, // query_start_len, batch + 1 entries
    seq_lens: []const i32,
    block_table: []const [max_pages]i32,
};

/// q and out are [token][hkv][hg][hd], the caches [page][slot][hkv][hd]: query head g of
/// kv head h attends with that kv head.
fn reference(c: Case, nkv: usize, ng: usize, q: []const f32, kc: []const f32, vc: []const f32, out: []f32) void {
    @memset(out, 0);
    const scale = 1 / @sqrt(@as(f32, hd));
    for (c.seq_lens, 0..) |len, s| {
        const qs: usize = @intCast(c.q_start[s]);
        const qe: usize = @intCast(c.q_start[s + 1]);
        const n = qe - qs;
        for (0..n) |j| for (0..nkv) |h| for (0..ng) |g| {
            const qo = (((qs + j) * nkv + h) * ng + g) * hd;
            const pos: usize = @as(usize, @intCast(len)) - n + j;
            var w: [max_pages * ps]f32 = undefined;
            var m: f32 = -std.math.inf(f32);
            for (0..pos + 1) |p| {
                const page: usize = @intCast(c.block_table[s][p / ps]);
                const ko = ((page * ps + p % ps) * nkv + h) * hd;
                var d: f32 = 0;
                for (0..hd) |x| d += q[qo + x] * kc[ko + x];
                w[p] = d * scale;
                m = @max(m, w[p]);
            }
            var sum: f32 = 0;
            for (0..pos + 1) |p| {
                w[p] = @exp(w[p] - m);
                sum += w[p];
            }
            for (0..hd) |x| {
                var acc: f32 = 0;
                for (0..pos + 1) |p| {
                    const page: usize = @intCast(c.block_table[s][p / ps]);
                    acc += w[p] / sum * vc[((page * ps + p % ps) * nkv + h) * hd + x];
                }
                out[qo + x] = acc;
            }
        };
    }
}

fn check(comptime batch: u32, comptime tokens: u32, comptime nkv: u32, comptime ng: u32, c: Case, seed: u64) !void {
    const platform = zml.testing.env();
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    const parameters: paged.Parameters = .init(.fromBackend(.{
        .backend = .stablehlo,
        .is_prefill = true,
        .batch_size = batch,
        .seq_len = max_pages * ps,
        .max_num_pages = max_pages,
        .max_token_count = tokens,
        .num_heads = nkv * ng,
        .num_kv_heads = nkv,
        .head_dim = hd,
        .max_seqlen_q = tokens,
    }));
    const q: zml.Tensor = .init(.{ .b = tokens, .hkv = nkv, .hg = ng, .hd = hd }, .f32);
    const nk: zml.Tensor = .init(.{ .b = tokens, .hkv = nkv, .hd = hd }, .f32);
    const nv: zml.Tensor = .init(.{ .b = tokens, .hkv = nkv, .hd = hd }, .f32);
    const cache: paged.KvCache = .{ .split = .{
        .k = .init(.{ .page = num_pages, .k_chunk = ps, .hkv = nkv, .hd = hd }, .f32),
        .v = .init(.{ .page = num_pages, .k_chunk = ps, .hkv = nkv, .hd = hd }, .f32),
    } };
    const exe = try platform.compileFn(allocator, io, paged.pagedAttention, .{ parameters, q, nk, nv, cache, .{} }, .{});
    defer exe.deinit();

    var q_h: [tokens * nkv * ng * hd]f32 = undefined;
    var k_h: [num_pages * ps * nkv * hd]f32 = undefined;
    var v_h: [num_pages * ps * nkv * hd]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(seed);
    const rng = prng.random();
    for (&q_h) |*x| x.* = rng.floatNorm(f32);
    for (&k_h) |*x| x.* = rng.floatNorm(f32);
    for (&v_h) |*x| x.* = rng.floatNorm(f32);
    var bt: [batch][max_pages]i32 = undefined;
    @memcpy(&bt, c.block_table);
    var sl: [batch]i32 = undefined;
    @memcpy(&sl, c.seq_lens);
    var qsl: [batch + 1]i32 = undefined;
    @memcpy(&qsl, c.q_start);

    var q_d: zml.Buffer = try .fromBytes(io, platform, q.shape(), .replicated, std.mem.sliceAsBytes(&q_h));
    defer q_d.deinit();
    var nk_d: zml.Buffer = try .fromBytes(io, platform, nk.shape(), .replicated, std.mem.sliceAsBytes(q_h[0 .. tokens * nkv * hd]));
    defer nk_d.deinit();
    var nv_d: zml.Buffer = try .fromBytes(io, platform, nv.shape(), .replicated, std.mem.sliceAsBytes(q_h[0 .. tokens * nkv * hd]));
    defer nv_d.deinit();
    var kv_d: zml.Bufferized(paged.KvCache) = .{ .split = .{
        .k = try .fromBytes(io, platform, cache.split.k.shape(), .replicated, std.mem.sliceAsBytes(&k_h)),
        .v = try .fromBytes(io, platform, cache.split.v.shape(), .replicated, std.mem.sliceAsBytes(&v_h)),
    } };
    defer zml.Buffer.deinitAll(paged.KvCache, &kv_d);
    var params_d: zml.Bufferized(paged.Parameters) = .{ .stablehlo = .{
        .block_table = try .fromBytes(io, platform, parameters.stablehlo.block_table.shape(), .replicated, std.mem.sliceAsBytes(&bt)),
        .seq_lens = try .fromBytes(io, platform, parameters.stablehlo.seq_lens.shape(), .replicated, std.mem.sliceAsBytes(&sl)),
        .query_start_len = try .fromBytes(io, platform, parameters.stablehlo.query_start_len.shape(), .replicated, std.mem.sliceAsBytes(&qsl)),
    } };
    defer zml.Buffer.deinitAll(paged.Parameters, &params_d);

    var out_d = try zml.testing.autoCall(allocator, io, &exe, paged.pagedAttention, .{ params_d, q_d, nk_d, nv_d, kv_d });
    defer out_d.deinit();
    const got = try out_d.toSliceAlloc(allocator, io);
    defer got.free(allocator);

    var want: [tokens * nkv * ng * hd]f32 = undefined;
    reference(c, nkv, ng, &q_h, &k_h, &v_h, &want);
    const used: usize = @as(usize, @intCast(c.q_start[batch])) * nkv * ng * hd;
    for (got.constItems(f32)[0..used], want[0..used], 0..) |g, w, i| {
        if (@abs(g - w) > 1e-3 * @max(1, @abs(w))) {
            std.debug.print("seed {d}: q_start {any} seq_lens {any} block_table {any}\nheads {d}x{d}, flat {d}: got {d}, want {d}\n", .{ seed, c.q_start, c.seq_lens, c.block_table, nkv, ng, i, g, w });
            return error.TestUnexpectedResult;
        }
    }
}

fn randomCase(comptime batch: u32, comptime tokens: u32, rng: std.Random, q_start: *[batch + 1]i32, seq_lens: *[batch]i32, bt: *[batch][max_pages]i32) bool {
    var pages: [num_pages]i32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    rng.shuffle(i32, &pages);
    var next_page: usize = 0;
    q_start[0] = 0;
    for (0..batch) |s| {
        const n: i32 = if (rng.boolean()) 1 else ps * rng.intRangeAtMost(i32, 1, 2);
        const len = n + rng.intRangeAtMost(i32, 0, max_pages * ps - n);
        const n_pages: usize = @intCast(@divFloor(len + ps - 1, ps));
        if (next_page + n_pages > num_pages) return false;
        bt[s] = @splat(-1);
        for (0..n_pages) |p| bt[s][p] = pages[next_page + p];
        next_page += n_pages;
        seq_lens[s] = len;
        q_start[s + 1] = q_start[s] + n;
    }
    return q_start[batch] <= tokens;
}

fn randomChecks(comptime batch: u32, comptime tokens: u32, comptime nkv: u32, comptime ng: u32, seed0: u64) !void {
    var prng: std.Random.DefaultPrng = .init(seed0);
    const rng = prng.random();
    var done: usize = 0;
    var tries: usize = 0;
    while (done < 12 and tries < 200) : (tries += 1) {
        var q_start: [batch + 1]i32 = undefined;
        var seq_lens: [batch]i32 = undefined;
        var bt: [batch][max_pages]i32 = undefined;
        if (!randomCase(batch, tokens, rng, &q_start, &seq_lens, &bt)) continue;
        try check(batch, tokens, nkv, ng, .{ .q_start = &q_start, .seq_lens = &seq_lens, .block_table = &bt }, seed0 + tries);
        done += 1;
    }
}

test "paged attention: random, one head" {
    try randomChecks(1, 12, 1, 1, 100);
    try randomChecks(2, 12, 1, 1, 200);
    try randomChecks(3, 12, 1, 1, 300);
}

test "paged attention: random, grouped query heads" {
    try randomChecks(2, 12, 1, 2, 400);
    try randomChecks(2, 12, 2, 1, 500);
    try randomChecks(3, 12, 2, 3, 600);
}
