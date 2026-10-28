const std = @import("std");
const zml = @import("zml");

// nn.sampleTokensDynamic keeps tokens whose inclusive cumulative probability is <= top_p
// (zml/nn.zig:2123 at 74c590b0), which drops the token that crosses the threshold.
// Nucleus sampling keeps the smallest set whose probabilities reach top_p: with
// probabilities {0.5, 0.45, 0.05} and top_p = 0.9 it samples tokens 0 and 1, ZML always
// samples token 0 (4096 of 4096 draws), so moderate top_p values collapse to greedy.
// Found by zmlsmith sampling property tests.

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

test "top_p keeps the token that reaches it" {
    // 0.5 < 0.9: nucleus sampling also keeps the second token (0.5 + 0.45 >= 0.9).
    const c = try counts(3, .{ 0.5, 0.45, 0.05 }, 3, 0.9, 0);
    try std.testing.expect(c[1] > 0);
    try std.testing.expectEqual(@as(u32, 0), c[2]);
}
