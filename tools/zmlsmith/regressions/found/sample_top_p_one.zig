const std = @import("std");
const zml = @import("zml");

// nn.sampleTokensDynamic with top_p = 1, the default of DynamicSamplingStrategy.Opts,
// still drops tokens: candidates are `cumsum(probs) <= top_p` (zml/nn.zig:2114-2123 at
// 74c590b0) and the f32 cumulative sum rounds above 1 before the last tokens. With
// probabilities {0.3, 0.3, 0.2, 0.1, 0.1} the last token is never sampled in 4096 draws.
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

test "top_p = 1 keeps every token, uneven" {
    const c = try counts(5, .{ 0.3, 0.3, 0.2, 0.1, 0.1 }, 5, 1, 0);
    for (c) |x| try std.testing.expect(x > 0);
}
