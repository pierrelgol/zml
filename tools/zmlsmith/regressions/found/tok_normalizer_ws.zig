const std = @import("std");
const homemade = @import("zml").tokenizer.homemade;

// homemade.Normalizer.fromHfJson appends the "content" of every Replace " " step to the
// fixed capacity _whitespace buffer with appendSliceAssumeCapacity, so a replacement
// longer than the buffer, or several Replace steps, overflow it instead of returning
// error.InvalidNormalizerJson (zml/tokenizer/homemade.zig:802-804 at 74c590b0). Found by
// reading the code while auditing zmlsmith, reached by zmlsmith parse-fuzz.
test "tok_normalizer_ws" {
    const json =
        \\{"type":"Replace","pattern":{"String":" "},"content":"0123456789abcdef"}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    _ = homemade.Normalizer.fromHfJson(parsed.value.object) catch {};
}
