const std = @import("std");
const homemade = @import("zml").tokenizer.homemade;

// homemade.fromHfJson and homemade.fromTinyLlamaFile, the only loaders of the pure Zig
// tokenizer, call std.fs.cwd(), which Zig 0.16 removed: any use fails to compile
// (zml/tokenizer/homemade.zig:1057, :1207 at 74c590b0). They have no callers, so the build
// never analyzes them. Found by zmlsmith parse-fuzz.
test "tok_homemade_load" {
    if (homemade.fromHfJson(std.testing.allocator, std.testing.io, "/nonexistent/tokenizer.json")) |t| {
        t.deinit();
    } else |_| {}
}
