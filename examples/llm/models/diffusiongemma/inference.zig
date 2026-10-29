const LoadedModel = @import("LoadedModel.zig");

pub const CompiledModel = struct {
    loaded_model: *const LoadedModel,
    seqlen: u32,

    pub fn deinit(self: *CompiledModel) void {
        _ = self;
    }
};
