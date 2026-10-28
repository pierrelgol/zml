const std = @import("std");
const zml = @import("zml");

// Upstream XLA CPU: a convolution with negative padding and non default dimension numbers
// (any of the input, kernel or output layouts permuted) gives wrong values, against a
// naive grouped convolution with StableHLO semantics; 8 of 120 random cases differ.
// The same cases with padding >= 0 (conv_layouts.zig), or negative padding in the
// default layout, are right. ZML passes the padding through unchanged
// (zml/tensor.zig:1040-1065). Found by zmlsmith (conv family, seed 250785).

/// out[b][o][y][x] = sum over c in o's group, ky, kx of in[b][g*cg + c][y*s + ky - pl][x*s + kx - pl'].
fn naive(in: []const f32, k: []const f32, out: []f32, d: [4]usize, kd: [4]usize, od: [4]usize, s: [2]i64, pad: [4]i64, groups: usize) void {
    const cg = d[1] / groups;
    const og = kd[0] / groups;
    for (0..od[0]) |b| for (0..od[1]) |o| for (0..od[2]) |y| for (0..od[3]) |x| {
        var acc: f32 = 0;
        const g = o / og;
        for (0..cg) |c| for (0..kd[2]) |ky| for (0..kd[3]) |kx| {
            const iy = @as(i64, @intCast(y)) * s[0] + @as(i64, @intCast(ky)) - pad[0];
            const ix = @as(i64, @intCast(x)) * s[1] + @as(i64, @intCast(kx)) - pad[2];
            if (iy < 0 or ix < 0 or iy >= d[2] or ix >= d[3]) continue;
            const ii = ((b * d[1] + g * cg + c) * d[2] + @as(usize, @intCast(iy))) * d[3] + @as(usize, @intCast(ix));
            const ki = ((o * kd[1] + c) * kd[2] + ky) * kd[3] + kx;
            acc += in[ii] * k[ki];
        };
        out[((b * od[1] + o) * od[2] + y) * od[3] + x] = acc;
    };
}

/// Physical layouts: perm[r] is the physical axis of logical role r ([b|o, c|i, y, x]).
const S = struct {
    var st: [2]i64 = undefined;
    var pd: [4]i64 = undefined;
    var groups: i64 = 1;
    var il: [4]i64 = .{ 0, 1, 2, 3 };
    var kl: [4]i64 = .{ 0, 1, 2, 3 };
    var ol: [4]i64 = .{ 0, 1, 2, 3 };
    fn f(x: zml.Tensor, k: zml.Tensor) zml.Tensor {
        return x.conv2d(k, .{
            .window_strides = &st,
            .padding = &pd,
            .feature_group_count = groups,
            .input_batch_dimension = il[0],
            .input_feature_dimension = il[1],
            .input_spatial_dimensions = il[2..4],
            .kernel_output_feature_dimension = kl[0],
            .kernel_input_feature_dimension = kl[1],
            .kernel_spatial_dimensions = kl[2..4],
            .output_batch_dimension = ol[0],
            .output_feature_dimension = ol[1],
            .output_spatial_dimensions = ol[2..4],
        });
    }
};

/// Logical row-major [4] dims and data to the physical layout `perm`.
fn physical(perm: [4]i64, d: [4]usize, logical: []const f32, out: []f32) [4]usize {
    var pd: [4]usize = undefined;
    for (0..4) |r| pd[@intCast(perm[r])] = d[r];
    for (0..d[0]) |a| for (0..d[1]) |b| for (0..d[2]) |c| for (0..d[3]) |e| {
        const li = ((a * d[1] + b) * d[2] + c) * d[3] + e;
        var pi: [4]usize = undefined;
        const idx = [4]usize{ a, b, c, e };
        for (0..4) |r| pi[@intCast(perm[r])] = idx[r];
        out[((pi[0] * pd[1] + pi[1]) * pd[2] + pi[2]) * pd[3] + pi[3]] = logical[li];
    };
    return pd;
}

test "xla_conv_neg" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var prng: std.Random.DefaultPrng = .init(3);
    const rng = prng.random();
    var fails: usize = 0;
    for (0..120) |round| {
        const groups: usize = rng.intRangeAtMost(usize, 1, 2);
        const d: [4]usize = .{ 1, groups * rng.intRangeAtMost(usize, 1, 2), rng.intRangeAtMost(usize, 3, 6), rng.intRangeAtMost(usize, 3, 6) };
        const kd: [4]usize = .{ groups * rng.intRangeAtMost(usize, 1, 2), d[1] / groups, 2, 2 };
        const s: [2]i64 = .{ rng.intRangeAtMost(i64, 1, 2), rng.intRangeAtMost(i64, 1, 2) };
        const pad: [4]i64 = .{ rng.intRangeAtMost(i64, -1, 1), rng.intRangeAtMost(i64, -1, 1), rng.intRangeAtMost(i64, -1, 1), rng.intRangeAtMost(i64, -1, 1) };
        var od: [4]usize = .{ d[0], kd[0], 0, 0 };
        for (0..2) |a| {
            const padded = @as(i64, @intCast(d[2 + a])) + pad[2 * a] + pad[2 * a + 1];
            if (padded < 2) break;
            od[2 + a] = @intCast(@divFloor(padded - 2, s[a]) + 1);
        } else {
            S.st = s;
            S.pd = pad;
            S.groups = @intCast(groups);
            for ([_]*[4]i64{ &S.il, &S.kl, &S.ol }) |l| {
                l.* = .{ 0, 1, 2, 3 };
                if (rng.boolean()) rng.shuffle(i64, l);
            }
            var pxd: [4]usize = undefined;
            var pkd: [4]usize = undefined;
            for (0..4) |r| {
                pxd[@intCast(S.il[r])] = d[r];
                pkd[@intCast(S.kl[r])] = kd[r];
            }
            const xs: zml.Shape = .init(.{ pxd[0], pxd[1], pxd[2], pxd[3] }, .f32);
            const ks: zml.Shape = .init(.{ pkd[0], pkd[1], pkd[2], pkd[3] }, .f32);
            var exe = try platform.compileFn(allocator, io, S.f, .{ zml.Tensor.fromShape(xs), zml.Tensor.fromShape(ks) }, .{});
            defer exe.deinit();
            const xin = try allocator.alloc(f32, xs.count());
            defer allocator.free(xin);
            const kin = try allocator.alloc(f32, ks.count());
            defer allocator.free(kin);
            for (xin) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
            for (kin) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
            const xp = try allocator.alloc(f32, xs.count());
            defer allocator.free(xp);
            const kp = try allocator.alloc(f32, ks.count());
            defer allocator.free(kp);
            _ = physical(S.il, d, xin, xp);
            _ = physical(S.kl, kd, kin, kp);
            var bx = try zml.Buffer.fromBytes(io, platform, xs, .replicated, std.mem.sliceAsBytes(xp));
            defer bx.deinit();
            var bk = try zml.Buffer.fromBytes(io, platform, ks, .replicated, std.mem.sliceAsBytes(kp));
            defer bk.deinit();
            var res = try zml.testing.autoCall(allocator, io, &exe, S.f, .{ bx, bk });
            defer res.deinit();
            const got = try res.toSliceAlloc(allocator, io);
            defer got.free(allocator);
            const want = try allocator.alloc(f32, od[0] * od[1] * od[2] * od[3]);
            defer allocator.free(want);
            naive(xin, kin, want, d, kd, od, s, pad, groups);
            const wantp = try allocator.alloc(f32, want.len);
            defer allocator.free(wantp);
            _ = physical(S.ol, od, want, wantp);
            if (!std.mem.eql(f32, got.constItems(f32), wantp)) {
                fails += 1;
                if (fails <= 3) std.debug.print("round {d}: in {any} kernel {any} strides {any} pad {any} groups {d} layouts {any} {any} {any}\n  got  {any}\n  want {any}\n", .{ round, d, kd, s, pad, groups, S.il, S.kl, S.ol, got.constItems(f32), wantp });
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 0), fails);
}
