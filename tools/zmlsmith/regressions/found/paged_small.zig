const std = @import("std");
const zml = @import("zml");
const paged = zml.attention.paged_attention;

// The stablehlo (CPU) backend of attention.paged_attention.pagedAttention, against a host
// reference of causal paged attention: query j of a sequence with n queries and seq_len L
// sits at position L - n + j and attends keys 0..L - n + j, read from the pages listed in
// block_table. Found by auditing zmlsmith coverage of zml/attention. Same harness as
// paged_attn.zig, split out because this case panics at compile time.

const ps = 4; // page_size
const num_pages = 8;
const max_pages = 4;
const hd = 4;

const Case = struct {
    q_start: []const i32, // query_start_len, batch + 1 entries
    seq_lens: []const i32,
    block_table: []const [max_pages]i32,
};

fn reference(c: Case, q: []const f32, kc: []const f32, vc: []const f32, out: []f32) void {
    @memset(out, 0);
    const scale = 1 / @sqrt(@as(f32, hd));
    for (c.seq_lens, 0..) |len, s| {
        const qs: usize = @intCast(c.q_start[s]);
        const qe: usize = @intCast(c.q_start[s + 1]);
        const n = qe - qs;
        for (0..n) |j| {
            const pos: usize = @as(usize, @intCast(len)) - n + j;
            var w: [max_pages * ps]f32 = undefined;
            var m: f32 = -std.math.inf(f32);
            for (0..pos + 1) |p| {
                const page: usize = @intCast(c.block_table[s][p / ps]);
                var d: f32 = 0;
                for (0..hd) |x| d += q[(qs + j) * hd + x] * kc[(page * ps + p % ps) * hd + x];
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
                    acc += w[p] / sum * vc[(page * ps + p % ps) * hd + x];
                }
                out[(qs + j) * hd + x] = acc;
            }
        }
    }
}

fn check(comptime batch: u32, comptime tokens: u32, c: Case) !void {
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
        .num_heads = 1,
        .num_kv_heads = 1,
        .head_dim = hd,
        .max_seqlen_q = tokens,
    }));
    const q: zml.Tensor = .init(.{ .b = tokens, .hkv = 1, .hg = 1, .hd = hd }, .f32);
    const nk: zml.Tensor = .init(.{ .b = tokens, .hkv = 1, .hd = hd }, .f32);
    const nv: zml.Tensor = .init(.{ .b = tokens, .hkv = 1, .hd = hd }, .f32);
    const cache: paged.KvCache = .{ .split = .{
        .k = .init(.{ .page = num_pages, .k_chunk = ps, .hkv = 1, .hd = hd }, .f32),
        .v = .init(.{ .page = num_pages, .k_chunk = ps, .hkv = 1, .hd = hd }, .f32),
    } };
    const exe = try platform.compileFn(allocator, io, paged.pagedAttention, .{ parameters, q, nk, nv, cache, .{} }, .{});
    defer exe.deinit();

    var q_h: [tokens * hd]f32 = undefined;
    var k_h: [num_pages * ps * hd]f32 = undefined;
    var v_h: [num_pages * ps * hd]f32 = undefined;
    for (&q_h, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 7 % 11)) - 5)) / 4;
    for (&k_h, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i * 5 % 13)) - 6)) / 4;
    for (&v_h, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 17));
    var bt: [batch][max_pages]i32 = undefined;
    @memcpy(&bt, c.block_table);
    var sl: [batch]i32 = undefined;
    @memcpy(&sl, c.seq_lens);
    var qsl: [batch + 1]i32 = undefined;
    @memcpy(&qsl, c.q_start);

    var q_d: zml.Buffer = try .fromBytes(io, platform, q.shape(), .replicated, std.mem.sliceAsBytes(&q_h));
    defer q_d.deinit();
    var nk_d: zml.Buffer = try .fromBytes(io, platform, nk.shape(), .replicated, std.mem.sliceAsBytes(q_h[0 .. tokens * hd]));
    defer nk_d.deinit();
    var nv_d: zml.Buffer = try .fromBytes(io, platform, nv.shape(), .replicated, std.mem.sliceAsBytes(q_h[0 .. tokens * hd]));
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

    var want: [tokens * hd]f32 = undefined;
    reference(c, &q_h, &k_h, &v_h, &want);
    for (got.constItems(f32), want, 0..) |g, w, i| {
        if (@abs(g - w) > 1e-3 * @max(1, @abs(w))) {
            std.debug.print("token {d} lane {d}: got {d}, want {d}\n", .{ i / hd, i % hd, g, w });
            return error.TestUnexpectedResult;
        }
    }
}

test "paged_small" {
    // Two decodes, fewer query tokens than page_size: the prefill branch of the loop body
    // slices a full page of queries from q, so the program doesn't compile ("slice size 4
    // greater than dimension size 2"), although no sequence is a prefill.
    try check(2, 2, .{
        .q_start = &.{ 0, 1, 2 },
        .seq_lens = &.{ 5, 7 },
        .block_table = &.{ .{ 0, 1, -1, -1 }, .{ 2, 3, -1, -1 } },
    });
}
