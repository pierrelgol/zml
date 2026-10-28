const std = @import("std");
const zml = @import("zml");

// Streaming decode with the sentencepiece backend (Decoder.feedOne then finalize) against
// a batch decode of the same ids with a fresh decoder, on random id sequences. next()
// decodes a sliding window of the last 4 ids and emits the part of the result that wasn't
// in the previous one, matched by suffix (zml/tokenizer/sentencepiece/sentencepiece.zig:
// 137-163 at 74c590b0): once the window slides over repeated tokens the new decode equals
// the old one and the next token's text is never emitted: ids {6, 3, 3, 3, 6, 4} stream
// "a   " where the batch decode is "a    aa". Found by zmlsmith tokenizer
// property tests.
/// A minimal sentencepiece ModelProto: <unk>, <s>, </s>, then ▁ a b ▁a ab.
const model = [_]u8{ 10, 14, 10, 5, 60, 117, 110, 107, 62, 21, 0, 0, 0, 0, 24, 2, 10, 12, 10, 3, 60, 115, 62, 21, 0, 0, 0, 0, 24, 3, 10, 13, 10, 4, 60, 47, 115, 62, 21, 0, 0, 0, 0, 24, 3, 10, 12, 10, 3, 226, 150, 129, 21, 0, 0, 128, 191, 24, 1, 10, 10, 10, 1, 97, 21, 0, 0, 0, 192, 24, 1, 10, 10, 10, 1, 98, 21, 0, 0, 0, 192, 24, 1, 10, 13, 10, 4, 226, 150, 129, 97, 21, 0, 0, 192, 191, 24, 1, 10, 11, 10, 2, 97, 98, 21, 0, 0, 128, 191, 24, 1 };

test "spm_stream" {
    const allocator = std.testing.allocator;
    var tok = try zml.tokenizer.Tokenizer.fromBytes(allocator, &model);
    defer tok.deinit();
    var prng: std.Random.DefaultPrng = .init(5);
    const rng = prng.random();
    for (0..300) |_| {
        var ids: [12]u32 = undefined;
        const n = rng.intRangeAtMost(usize, 1, ids.len);
        for (ids[0..n]) |*id| id.* = rng.intRangeAtMost(u32, 3, 7);
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
