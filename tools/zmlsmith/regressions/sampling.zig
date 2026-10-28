//! nn.sampleTokensDynamic, statistically: 4096 rows of the same probabilities in one
//! call, each with its own gumbel noise. The top_p bugs are in found/sample_top_p*.zig.

const std = @import("std");
const zml = @import("zml");

const rows = 4096;

const Fns = struct {
    fn sample(logits: zml.Tensor, opts: zml.nn.DynamicSamplingStrategy, rng: zml.Tensor.Rng) zml.Tensor {
        const t, _ = zml.nn.sampleTokensDynamic(logits, opts, rng);
        return t;
    }
};

/// Samples `rows` times from the same probabilities and counts each token.
fn counts(comptime voc: usize, probs: [voc]f32, top_k: u32, top_p: f32, min_p: f32) ![voc]u32 {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const logits: zml.Tensor = .init(.{ .s = rows, .voc = voc }, .f32);
    const opts: zml.nn.DynamicSamplingStrategy = .init(.f32, 0);
    var exe = try platform.compileFn(allocator, io, Fns.sample, .{ logits, opts, zml.Tensor.Rng.init() }, .{});
    defer exe.deinit();
    var data: [rows * voc]f32 = undefined;
    for (0..rows) |r| for (probs, 0..) |p, v| {
        data[r * voc + v] = @log(p);
    };
    var bl = try zml.Buffer.fromBytes(io, platform, logits.shape(), .replicated, std.mem.sliceAsBytes(&data));
    defer bl.deinit();
    var bo = try zml.nn.DynamicSamplingStrategy.makeBuffers(io, platform, .f32, .{ .top_k = top_k, .top_p = top_p, .min_p = min_p });
    defer zml.nn.DynamicSamplingStrategy.deinitBuffers(&bo);
    var br = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, 1234);
    defer zml.Tensor.Rng.deinitBuffer(&br);
    var res = try zml.testing.autoCall(allocator, io, &exe, Fns.sample, .{ bl, bo, br });
    defer res.deinit();
    const s = try res.toSliceAlloc(allocator, io);
    defer s.free(allocator);
    var c: [voc]u32 = @splat(0);
    for (s.constItems(i32)) |t| c[@intCast(t)] += 1;
    std.debug.print("probs {any} top_k {d} top_p {d} min_p {d}: counts {any}\n", .{ probs, top_k, top_p, min_p, c });
    return c;
}

test "sampling: top_p = 1 keeps every token of a uniform distribution" {
    const c = try counts(4, .{ 0.25, 0.25, 0.25, 0.25 }, 4, 1, 0);
    for (c) |x| try std.testing.expect(x > 0);
}

test "sampling: min_p drops tokens below min_p * max" {
    const c = try counts(3, .{ 0.6, 0.3, 0.1 }, 3, 1, 0.25);
    try std.testing.expect(c[0] > 0 and c[1] > 0);
    try std.testing.expectEqual(@as(u32, 0), c[2]);
}
