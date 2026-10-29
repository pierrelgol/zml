const std = @import("std");
const mem = std.mem;
const zml = @import("zml");
const zml_io = zml.io;

const Config = @import("Config.zig");
const decoder = @import("decoder.zig");
const Stack = @import("Stack.zig");

const Model = @This();

stack: Stack,
self_conditioning: decoder.SelfConditioning,

pub fn init(allocator: mem.Allocator, store: zml_io.TensorStore.View, config: Config) !Model {
    const decoder_store = store.withPrefix("model.decoder");
    const encoder_store = store.withPrefix("model.encoder.language_model");
    const text = config.text_config;
    return .{
        .stack = try .init(allocator, decoder_store, encoder_store, text),
        .self_conditioning = .init(decoder_store.withPrefix("self_conditioning"), text.rms_norm_eps),
    };
}

pub fn deinit(self: Model, allocator: mem.Allocator) void {
    self.stack.deinit(allocator);
}
