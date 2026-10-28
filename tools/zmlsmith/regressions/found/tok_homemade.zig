const std = @import("std");
const homemade = @import("zml").tokenizer.homemade;

// homemade.Tokenizer.lookupPiece guards invalid ids with `id > tokens.len`, so
// `id == tokens.len` indexes one past the end instead of returning "<oob>"
// (zml/tokenizer/homemade.zig:298 at 74c590b0). Found by reading the code while auditing
// zmlsmith coverage.
test "tok_homemade" {
    var t = try homemade.Tokenizer.init(std.testing.allocator, 4, 4, null, .{ .bos = 0, .eos = 1, .unk = 2 }, true);
    defer t.deinit();
    try t.addToken(0, "<s>");
    try t.addToken(0, "</s>");
    try t.addToken(0, "a");
    try t.addToken(0, "b");
    try std.testing.expectEqualStrings("<oob>", t.lookupPiece(t.tokens.len));
}
