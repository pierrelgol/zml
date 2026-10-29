const std = @import("std");
const mem = std.mem;

const Generation = @This();

eos_token_id: []const u32,

pub fn validate(self: Generation) !void {
    if (self.eos_token_id.len == 0) {
        return error.InvalidGenerationConfig;
    }
}

pub fn isEndOfTurn(self: Generation, token: u32) bool {
    return mem.indexOfScalar(u32, self.eos_token_id, token) != null;
}
