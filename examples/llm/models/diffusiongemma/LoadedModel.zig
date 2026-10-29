const std = @import("std");
const json = std.json;
const mem = std.mem;
const time = std.time;
const zml = @import("zml");
const attention = zml.attention;
const Io = std.Io;
const TensorStore = zml.io.TensorStore;

const common = @import("../common.zig");
const Buffers = @import("../diffusiongemma.zig").Buffers;
const Config = @import("Config.zig");
const Generation = @import("Generation.zig");
const Model = @import("Model.zig");
const inference = @import("inference.zig");
const log = std.log.scoped(.diffusiongemma_loaded_model);
pub const LoadedModel = @This();

model: Model,
parsed_config: json.Parsed(Config),
parsed_generation: json.Parsed(Generation),

pub fn init(allocator: mem.Allocator, io: std.Io, repo: Io.Dir, store: TensorStore.View, options: common.GenerationOptions) !LoadedModel {
    _ = options;

    const parsed_config = try common.parseConfig(Config, allocator, io, repo);
    errdefer parsed_config.deinit();
    try parsed_config.value.validate();

    const parsed_generation = try common.parseConfig(Generation, allocator, io, repo);
    errdefer parsed_generation.deinit();
    try parsed_generation.value.validate();

    const loaded_model: LoadedModel = .{
        .model = try .init(allocator, store, parsed_config.value),
        .parsed_config = parsed_config,
        .parsed_generation = parsed_generation,
    };

    loaded_model.logModelInfo();
    return loaded_model;
}

pub fn deinit(self: *LoadedModel, allocator: mem.Allocator) void {
    self.model.deinit(allocator);
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

pub fn loadBuffers(
    self: *const LoadedModel,
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
) !Buffers {
    progress.increaseEstimatedTotalItems(store.view().count());
    const started: std.Io.Timestamp = .now(io, .awake);

    var buffers = try zml.mem.bufferize(allocator, Model, &self.model);
    errdefer self.unloadBuffers(&buffers, allocator);

    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .dma_chunks = 32,
        .dma_chunk_size = 256 * zml.MiB,
        .parallelism = 16,
    });
    defer loader.deinit();

    try loader.load(io, Model, &self.model, &buffers, store, .{ .progress = progress });
    try loader.await(io);

    const took = started.untilNow(io, .awake);
    const total_bytes: u64 = loader.bytes_loaded.raw;
    const seconds = @as(f64, @floatFromInt(took.nanoseconds)) / time.ns_per_s;
    const bytes_per_second: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / seconds);
    log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_second });

    return buffers;
}

pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: mem.Allocator) void {
    _ = self;
    zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Model, buffers);
    allocator.free(buffers.stack.layers);
}

pub fn compile(
    self: *const LoadedModel,
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    backend: attention.Backend,
    meshes: common.Meshes,
    seqlen: usize,
    progress: *std.Progress.Node,
) !inference.CompiledModel {
    _ = allocator;
    _ = io;
    _ = platform;
    _ = backend;
    _ = meshes;
    _ = progress;
    return .{
        .loaded_model = self,
        .seqlen = @intCast(seqlen),
    };
}
