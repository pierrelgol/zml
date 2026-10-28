const std = @import("std");
const zml = @import("zml");
const homemade = zml.tokenizer.homemade;

// Streaming decode with the homemade backend (Decoder.feedOne then finalize) against a
// batch decode of the same ids with a fresh decoder, on random id sequences: like the
// sentencepiece one (spm_stream.zig) it decodes a window of recent ids and emits the
// difference, and repeated tokens get dropped: ids {5, 7, 4, 6, 3, 3, 3, 3, 3} stream
// "abéb aaaa" where the batch decode is "abéb aaaaa" (zml/tokenizer/homemade.zig:479 at
// 74c590b0). Found by zmlsmith tokenizer property tests.
test "tok_hm_stream" {
    const allocator = std.testing.allocator;
    var t = try homemade.Tokenizer.init(allocator, 9, 8, null, .{ .bos = 0, .eos = 1, .unk = 2 }, true);
    defer t.deinit();
    for ([_][]const u8{ "<s>", "</s>", "<unk>", "a", "b", "ab", " ", "\xc3\xa9", "aaa" }) |p| try t.addToken(0, p);
    var tok: zml.tokenizer.Tokenizer = .{ .homemade = &t };
    var prng: std.Random.DefaultPrng = .init(9);
    const rng = prng.random();
    for (0..300) |_| {
        var ids: [12]u32 = undefined;
        const n = rng.intRangeAtMost(usize, 1, ids.len);
        for (ids[0..n]) |*id| id.* = rng.intRangeAtMost(u32, 3, 8);
        var batch_dec = try tok.decoder();
        defer batch_dec.deinit();
        var want = try batch_dec.decodeAlloc(allocator, ids[0..n]);
        defer want.deinit(allocator);
        var dec = try tok.decoder();
        defer dec.deinit();
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(allocator);
        var buf: [256]u8 = undefined;
        for (ids[0..n]) |id| try got.appendSlice(allocator, try dec.feedOne(id, &buf));
        try got.appendSlice(allocator, try dec.finalize(&buf));
        if (!std.mem.eql(u8, got.items, want.items)) {
            std.debug.print("ids {any}: streamed \"{s}\", batch \"{s}\"\n", .{ ids[0..n], got.items, want.items });
            return error.TestUnexpectedResult;
        }
    }
}
