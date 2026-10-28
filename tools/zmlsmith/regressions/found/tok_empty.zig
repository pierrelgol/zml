const std = @import("std");
const homemade = @import("zml").tokenizer.homemade;

// homemade.Tokenizer.encode on an input that normalizes to nothing: zero char tokens, and
// `num_tokens - 1` underflows in the merge loop (zml/tokenizer/homemade.zig:196 at
// 74c590b0). Found by reading the code while auditing zmlsmith coverage.
test "tok_empty" {
    var t = try homemade.Tokenizer.init(std.testing.allocator, 4, 4, null, .{ .bos = 0, .eos = 1, .unk = 2 }, true);
    defer t.deinit();
    try t.addToken(0, "<s>");
    try t.addToken(0, "</s>");
    try t.addToken(0, "a");
    try t.addToken(0, "b");
    const ids = try t.encode(std.testing.allocator, "", .{ .add_bos = false });
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 0), ids.len);
}
