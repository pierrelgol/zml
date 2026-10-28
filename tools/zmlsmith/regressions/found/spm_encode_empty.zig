const std = @import("std");
const zml = @import("zml");

// sentencepiece Encoder.ids makes a slice from std_vector_int_data, which is null for an
// empty vector (zml/tokenizer/sentencepiece/sentencepiece.zig:75-80 at 74c590b0): encoding
// a text that gives no tokens, such as "", panics with "attempt to use null value" in
// ffi.ZigSlice.to. Found by zmlsmith parse-fuzz (spm target).
/// A minimal sentencepiece ModelProto: <unk>, <s>, </s>, then ▁ a b ▁a ab.
const model = [_]u8{ 10, 14, 10, 5, 60, 117, 110, 107, 62, 21, 0, 0, 0, 0, 24, 2, 10, 12, 10, 3, 60, 115, 62, 21, 0, 0, 0, 0, 24, 3, 10, 13, 10, 4, 60, 47, 115, 62, 21, 0, 0, 0, 0, 24, 3, 10, 12, 10, 3, 226, 150, 129, 21, 0, 0, 128, 191, 24, 1, 10, 10, 10, 1, 97, 21, 0, 0, 0, 192, 24, 1, 10, 10, 10, 1, 98, 21, 0, 0, 0, 192, 24, 1, 10, 13, 10, 4, 226, 150, 129, 97, 21, 0, 0, 192, 191, 24, 1, 10, 11, 10, 2, 97, 98, 21, 0, 0, 128, 191, 24, 1 };

test "spm_encode_empty" {
    var tok = try zml.tokenizer.Tokenizer.fromBytes(std.testing.allocator, &model);
    defer tok.deinit();
    var enc = try tok.encoder();
    defer enc.deinit();
    const ids = try enc.encodeAlloc(std.testing.allocator, "");
    std.testing.allocator.free(ids);
}
