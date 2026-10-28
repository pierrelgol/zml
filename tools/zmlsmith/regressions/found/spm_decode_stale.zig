const std = @import("std");
const zml = @import("zml");

// sentencepiece Decoder.decode pushes the ids into its std::vector without clearing it
// first (zml/tokenizer/sentencepiece/sentencepiece.zig:120-128 at 74c590b0): a second
// decode with the same decoder also decodes the ids of the first call. Found by reading
// the code while auditing zmlsmith, confirmed with a model built by the spm target.
/// A minimal sentencepiece ModelProto: <unk>, <s>, </s>, then ▁ a b ▁a ab.
const model = [_]u8{ 10, 14, 10, 5, 60, 117, 110, 107, 62, 21, 0, 0, 0, 0, 24, 2, 10, 12, 10, 3, 60, 115, 62, 21, 0, 0, 0, 0, 24, 3, 10, 13, 10, 4, 60, 47, 115, 62, 21, 0, 0, 0, 0, 24, 3, 10, 12, 10, 3, 226, 150, 129, 21, 0, 0, 128, 191, 24, 1, 10, 10, 10, 1, 97, 21, 0, 0, 0, 192, 24, 1, 10, 10, 10, 1, 98, 21, 0, 0, 0, 192, 24, 1, 10, 13, 10, 4, 226, 150, 129, 97, 21, 0, 0, 192, 191, 24, 1, 10, 11, 10, 2, 97, 98, 21, 0, 0, 128, 191, 24, 1 };

test "spm_decode_stale" {
    const allocator = std.testing.allocator;
    var tok = try zml.tokenizer.Tokenizer.fromBytes(allocator, &model);
    defer tok.deinit();
    var dec = try tok.decoder();
    defer dec.deinit();
    var first = try dec.decodeAlloc(allocator, &.{4}); // "a"
    defer first.deinit(allocator);
    var second = try dec.decodeAlloc(allocator, &.{5}); // "b"
    defer second.deinit(allocator);
    try std.testing.expectEqualStrings("b", second.items);
}
