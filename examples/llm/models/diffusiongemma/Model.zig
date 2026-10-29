const std = @import("std");
const mem = std.mem;
const zml = @import("zml");
const Config = @import("Config.zig");
const Stack = @import("Stack.zig");
const TensorStore = zml.io.TensorStore;

pub const Model = @This();

stack: Stack,

pub fn init(allocator: mem.Allocator, store: TensorStore.View, config: Config) !Model {
    const decoder_store = store.withPrefix("model.decoder");
    const encoder_store = store.withPrefix("model.encoder.language_model");

    return .{
        .stack = try .init(allocator, decoder_store, encoder_store, config.text_config),
    };
}

pub fn deinit(self: Model, allocator: mem.Allocator) void {
    self.stack.deinit(allocator);
}
