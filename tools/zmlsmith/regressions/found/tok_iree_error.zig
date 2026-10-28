const std = @import("std");
const zml = @import("zml");

// Any malformed tokenizer.json, even "{}", smashes the stack: iree.checkOk formats the
// error status into a 1024 byte stack buffer (zml/tokenizer/iree/tokenizer.zig:70-72 at
// 74c590b0), and the stack trace payload formatter of IREE keeps adding the untruncated
// length, so once the buffer is full `capacity - length` wraps and snprintf writes past it
// (iree/base/status_stack_trace.c:60-81). The process dies with SIGILL or a general
// protection fault on return from checkOk instead of fromBytes returning an error.
// Found by zmlsmith parse-fuzz.
test "tok_iree_error" {
    const inputs = [_][]const u8{
        "{\"\":\"\",\"\":{\"\":\"\",\"\":n",
        "{}",
    };
    for (inputs) |in| {
        if (zml.tokenizer.Tokenizer.fromBytes(std.testing.allocator, in)) |t| {
            var tok = t;
            tok.deinit();
        } else |_| {}
    }
}
