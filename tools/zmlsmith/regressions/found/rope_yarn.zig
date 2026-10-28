const std = @import("std");
const zml = @import("zml");

// RopeOpts.Scaling.jsonParse accepts a yarn config without `factor` (an optional field),
// then attentionScaling and the yarn inverse frequencies unwrap `factor.?` and panic
// (zml/nn.zig:491,560,562 at 74c590b0). Found by reading the code while auditing zmlsmith.
test "rope_yarn" {
    const json =
        \\{"rope_type": "yarn", "original_max_position_embeddings": 4096}
    ;
    // Rejecting the config at parse time is a valid fix (HF reads `factor` unconditionally).
    const parsed = std.json.parseFromSlice(zml.nn.RopeOpts.Scaling, std.testing.allocator, json, .{}) catch return;
    defer parsed.deinit();
    try std.testing.expect(std.math.isFinite(parsed.value.attentionScaling()));
}
