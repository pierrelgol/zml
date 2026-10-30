const std = @import("std");
const mem = std.mem;

const Generation = @This();

eos_token_id: []const u32,
max_denoising_steps: u32,
t_min: f32,
t_max: f32,
sampler_config: struct {
    entropy_bound: f32,
},
confidence_threshold: f32,
stability_threshold: u32,

pub fn validate(self: Generation) !void {
    if (self.eos_token_id.len == 0 or self.max_denoising_steps == 0) {
        return error.InvalidGenerationConfig;
    }

    const valid_schedule = self.t_min >= 0 and self.t_max > self.t_min;

    if (!valid_schedule or self.sampler_config.entropy_bound <= 0 or self.confidence_threshold <= 0) {
        return error.InvalidGenerationConfig;
    }
}

pub fn isEndOfTurn(self: Generation, token: u32) bool {
    return mem.indexOfScalar(u32, self.eos_token_id, token) != null;
}
