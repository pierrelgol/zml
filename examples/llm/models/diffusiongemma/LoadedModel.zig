const std = @import("std");
const json = std.json;
const mem = std.mem;
const zml = @import("zml");
const attention = zml.attention;
const Io = std.Io;
const TensorStore = zml.io.TensorStore;

const common = @import("../common.zig");
const Buffers = @import("../diffusiongemma.zig").Buffers;
const Config = @import("Config.zig");
const Generation = @import("Generation.zig");
const inference = @import("inference.zig");
const log = std.log.scoped(.diffusiongemma_loaded_model);
pub const LoadedModel = @This();

parsed_config: json.Parsed(Config),
parsed_generation: json.Parsed(Generation),

pub fn init(allocator: mem.Allocator, io: std.Io, repo: Io.Dir, store: TensorStore.View, options: common.GenerationOptions) !LoadedModel {
    _ = store;
    _ = options;

    const parsed_config = try common.parseConfig(Config, allocator, io, repo);
    errdefer parsed_config.deinit();
    try parsed_config.value.validate();

    const parsed_generation = try common.parseConfig(Generation, allocator, io, repo);
    errdefer parsed_generation.deinit();
    try parsed_generation.value.validate();

    const loaded_model: LoadedModel = .{
        .parsed_config = parsed_config,
        .parsed_generation = parsed_generation,
    };

    loaded_model.logModelInfo();
    return loaded_model;
}

pub fn deinit(self: *LoadedModel) void {
    self.parsed_config.deinit();
    self.parsed_generation.deinit();
}

pub fn config(self: *const LoadedModel) Config {
    return self.parsed_config.value;
}

pub fn generation(self: *const LoadedModel) Generation {
    return self.parsed_generation.value;
}

pub fn logModelInfo(self: *const LoadedModel) void {
    const model_config = self.config();
    const model_text_config = self.config().text_config;
    const model_generation = self.generation();

    log.info("Text stack | Total Layers : {d} | Sliding Attention : {d} of size : {d}| Full Attention : {d} | Hidden size : {d}", .{
        model_text_config.num_hidden_layers,
        model_config.countLayers(.sliding_attention),
        model_text_config.sliding_window,
        model_config.countLayers(.full_attention),
        model_text_config.hidden_size,
    });

    log.info("Attention | Querry Heads : {d} | {d}/{d} key-value heads (sliding/full) | Head Dimension : {d}/{d}", .{
        model_text_config.num_attention_heads,
        model_text_config.num_key_value_heads,
        model_text_config.num_global_key_value_heads,
        model_text_config.head_dim,
        model_text_config.global_head_dim,
    });

    log.info("Experts | {d} Per layer | {d} Active per token | Vocabulary size : {d}", .{
        model_text_config.num_experts,
        model_text_config.top_k_experts,
        model_text_config.vocab_size,
    });

    log.info("Generation | Eos_token_id(s) : {any}", .{
        model_generation.eos_token_id,
    });
}
