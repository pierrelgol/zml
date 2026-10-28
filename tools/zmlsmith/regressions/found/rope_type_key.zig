const std = @import("std");
const zml = @import("zml");

// RopeOpts.Scaling.jsonParse only reads `rope_type`. HF configs written before
// transformers renamed the key use `type` (e.g. {"type": "linear", "factor": 2.0}), and
// transformers still accepts it; ZML fails with MissingField (zml/nn.zig:529 at 74c590b0).
// Found by reading the code while auditing zmlsmith.
test "rope_type_key" {
    const json =
        \\{"type": "linear", "factor": 2.0}
    ;
    const parsed = try std.json.parseFromSlice(zml.nn.RopeOpts.Scaling, std.testing.allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(f32, 2.0), parsed.value.linear.factor);
}
