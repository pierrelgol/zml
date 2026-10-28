const std = @import("std");
const homemade = @import("zml").tokenizer.homemade;

// homemade.Tokenizer.encode advances its input offset by 1 byte for an unknown token,
// but the char iterator consumed a whole UTF-8 codepoint for it (homemade.zig:198, :233
// vs :544 at 74c590b0). After an unknown multi-byte character the merge candidates are
// read at the wrong offset: in Debug builds the corruption check panics with "invalid
// tokenization", in release builds merges are looked up on the wrong bytes. Found by
// reading the code while auditing zmlsmith coverage.
test "tok_unk_utf8" {
    var t = try homemade.Tokenizer.init(std.testing.allocator, 6, 4, null, .{ .unk = 0, .bos = 1, .eos = 2 }, true);
    defer t.deinit();
    try t.addToken(0, "<unk>");
    try t.addToken(0, "<s>");
    try t.addToken(0, "</s>");
    try t.addToken(0, "a");
    try t.addToken(0, "b");
    try t.addToken(1, "ab");
    // "é" is two bytes and not in the vocabulary.
    const ids = try t.encode(std.testing.allocator, "\xc3\xa9ab", .{ .add_bos = false });
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 5 }, ids);
}
