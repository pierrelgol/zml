const std = @import("std");
const zml = @import("zml");

// zml.tokenizer.Tokenizer.fromBytes picks the backend from `bytes[0]` without checking
// the length: an empty tokenizer file indexes out of bounds instead of returning an
// error (zml/tokenizer/tokenizer.zig:143 at 74c590b0). JSON with leading whitespace or a
// BOM is also routed to the sentencepiece protobuf parser. Found by reading the code while
// auditing zmlsmith.
test "tok_from_bytes" {
    if (zml.tokenizer.Tokenizer.fromBytes(std.testing.allocator, "")) |t| {
        var tok = t;
        tok.deinit();
        return error.TestUnexpectedResult;
    } else |_| {}
}
