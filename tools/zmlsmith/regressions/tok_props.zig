//! homemade.Tokenizer on random vocabularies: every printable ASCII byte plus random
//! multi byte tokens, no normalizer. decode(encode(text)) gives the text back, and the
//! pieces of the ids concatenate to it. Empty inputs and unknown UTF-8 are known bugs
//! (found/tok_empty.zig, found/tok_unk_utf8.zig) and stay out.

const std = @import("std");
const homemade = @import("zml").tokenizer.homemade;

test "homemade tokenizer: encode/decode round trip" {
    const allocator = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0x70c);
    const rng = prng.random();
    for (0..300) |_| {
        const n_extra = rng.uintLessThan(u32, 40);
        const vocab: u32 = 3 + 95 + n_extra;
        var t = try homemade.Tokenizer.init(allocator, vocab, 8, null, .{ .bos = 0, .eos = 1, .unk = 2 }, true);
        defer t.deinit();
        try t.addToken(0, "<s>");
        try t.addToken(0, "</s>");
        try t.addToken(0, "<unk>");
        for (32..127) |c| try t.addToken(0, &[_]u8{@intCast(c)});
        const alphabet = "ab c";
        var buf: [8]u8 = undefined;
        for (0..n_extra) |_| {
            const len = rng.intRangeAtMost(usize, 2, 6);
            for (buf[0..len]) |*b| b.* = alphabet[rng.uintLessThan(usize, alphabet.len)];
            try t.addToken(rng.float(f32) * 10, buf[0..len]);
        }
        var text_buf: [48]u8 = undefined;
        const tl = rng.intRangeAtMost(usize, 1, text_buf.len);
        for (text_buf[0..tl]) |*b| b.* = if (rng.boolean()) alphabet[rng.uintLessThan(usize, alphabet.len)] else @intCast(rng.intRangeAtMost(u8, 32, 126));
        const text = text_buf[0..tl];

        // encode's result can't be freed (found/tok_encode_free.zig): use an arena.
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const ids = try t.encode(arena.allocator(), text, .{ .add_bos = false });
        var w: std.Io.Writer.Allocating = .init(allocator);
        defer w.deinit();
        try t.decode(ids, &w.writer);
        if (!std.mem.eql(u8, w.written(), text)) {
            std.debug.print("text \"{s}\" -> {any} -> \"{s}\"\n", .{ text, ids, w.written() });
            return error.TestUnexpectedResult;
        }
        var cat: std.ArrayList(u8) = .empty;
        defer cat.deinit(allocator);
        for (ids) |id| try cat.appendSlice(allocator, t.lookupPiece(id));
        try std.testing.expectEqualStrings(text, cat.items);
    }
}
