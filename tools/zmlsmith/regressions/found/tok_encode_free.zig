const std = @import("std");
const homemade = @import("zml").tokenizer.homemade;

// homemade.Tokenizer.encode shrinks its result with `_ = allocator.resize(buf, n)` and
// ignores the result, then returns buf[0..n] (zml/tokenizer/homemade.zig:288-289 at
// 74c590b0). When the allocator can't resize in place the caller holds a slice shorter
// than the allocation, and freeing it, as encode's callers do, is an invalid free
// (detected by the testing allocator; undefined behavior otherwise). Found by zmlsmith
// tokenizer property tests.
test "tok_encode_free" {
    const allocator = std.testing.allocator;
    var t = try homemade.Tokenizer.init(allocator, 5, 4, null, .{ .bos = 0, .eos = 1, .unk = 2 }, true);
    defer t.deinit();
    try t.addToken(0, "<s>");
    try t.addToken(0, "</s>");
    try t.addToken(0, "<unk>");
    try t.addToken(0, "a");
    try t.addToken(1, "aa");
    const ids = try t.encode(allocator, "aaaaaaaa", .{ .add_bos = false });
    allocator.free(ids);
}
