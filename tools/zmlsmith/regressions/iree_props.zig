const std = @import("std");
const zml = @import("zml");

// The IREE tokenizer (zml.tokenizer.Tokenizer.fromBytes on a valid byte level BPE
// tokenizer.json): decode(encode(text)) is the text, and streaming decode (feedOne, then
// finalize) agrees with batch decode on random id sequences. Invalid files can't be
// tested: any error smashes the stack (found/tok_iree_error.zig).
const tok_bpe =
    \\{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":5,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
    \\"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
    \\"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
    \\"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,
    \\"vocab":{"a":0,"b":1,"ab":2,"Ġ":3,"Ġa":4,"<s>":5},"merges":["a b","Ġ a"]}}
;

test "iree tokenizer: round trip and streaming" {
    const allocator = std.testing.allocator;
    var tok = try zml.tokenizer.Tokenizer.fromBytes(allocator, tok_bpe);
    defer tok.deinit();
    var prng: std.Random.DefaultPrng = .init(11);
    const rng = prng.random();
    const alphabet = "ab ";
    for (0..200) |_| {
        var text_buf: [24]u8 = undefined;
        const n = rng.intRangeAtMost(usize, 1, text_buf.len);
        for (text_buf[0..n]) |*c| c.* = alphabet[rng.uintLessThan(usize, alphabet.len)];
        const text = text_buf[0..n];
        var enc = try tok.encoder();
        defer enc.deinit();
        const ids = try enc.encodeAlloc(allocator, text);
        defer allocator.free(ids);
        var dec = try tok.decoder();
        defer dec.deinit();
        var back = try dec.decodeAlloc(allocator, ids);
        defer back.deinit(allocator);
        if (!std.mem.eql(u8, back.items, text)) {
            std.debug.print("text \"{s}\" -> {any} -> \"{s}\"\n", .{ text, ids, back.items });
            return error.TestUnexpectedResult;
        }
        var sdec = try tok.decoder();
        defer sdec.deinit();
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(allocator);
        var buf: [256]u8 = undefined;
        for (ids) |id| try got.appendSlice(allocator, try sdec.feedOne(id, &buf));
        try got.appendSlice(allocator, try sdec.finalize(&buf));
        if (!std.mem.eql(u8, got.items, text)) {
            std.debug.print("ids {any}: streamed \"{s}\", batch \"{s}\"\n", .{ ids, got.items, text });
            return error.TestUnexpectedResult;
        }
    }
}
