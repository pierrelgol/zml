const std = @import("std");
const json = std.json;
const mem = std.mem;
const time = std.time;
const zml = @import("zml");
const attention = zml.attention;
const moe = zml.moe;

const zml_io = zml.io;
const zml_mem = zml.mem;

const common = @import("../common.zig");
const Buffers = @import("../diffusiongemma.zig").Buffers;
const Config = @import("Config.zig");
const Generation = @import("Generation.zig");
const inference = @import("inference.zig");
const KvCache = @import("KvCache.zig");
const Model = @import("Model.zig");
const sampler = @import("sampler.zig");

const log = std.log.scoped(.diffusiongemma);

const LoadedModel = @This();

model: Model,
parsed_config: json.Parsed(Config),
parsed_generation: json.Parsed(Generation),

pub fn init(
    allocator: mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
    store: zml_io.TensorStore.View,
    options: common.GenerationOptions,
) !LoadedModel {
    _ = options;

    const parsed_config = try common.parseConfig(Config, allocator, io, repo);
    errdefer parsed_config.deinit();
    try parsed_config.value.validate();

    const parsed_generation = try common.parseJsonFile(Generation, allocator, io, repo, "generation_config.json");
    errdefer parsed_generation.deinit();
    try parsed_generation.value.validate();

    const loaded: LoadedModel = .{
        .model = try .init(allocator, store, parsed_config.value),
        .parsed_config = parsed_config,
        .parsed_generation = parsed_generation,
    };
    loaded.logSummary();
    return loaded;
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

fn logSummary(self: LoadedModel) void {
    const model_config = self.config();
    const text = model_config.text_config;
    log.info("Text stack: {d} layers ({d} sliding, window {d}, and {d} full), hidden size {d}", .{
        text.num_hidden_layers,
        model_config.countLayers(.sliding_attention),
        text.sliding_window,
        model_config.countLayers(.full_attention),
        text.hidden_size,
    });
    log.info("Attention: {d} query heads, {d}/{d} key-value heads (sliding/full), head dim {d}/{d}", .{
        text.num_attention_heads,
        text.num_key_value_heads,
        text.num_global_key_value_heads,
        text.head_dim,
        text.global_head_dim,
    });
    log.info("Experts: {d} per layer, {d} active per token, vocabulary of {d} tokens", .{
        text.num_experts,
        text.top_k_experts,
        text.vocab_size,
    });
}

pub fn loadBuffers(
    self: *const LoadedModel,
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml_io.TensorStore,
    progress: *std.Progress.Node,
) !Buffers {
    progress.increaseEstimatedTotalItems(store.view().count());
    const started: std.Io.Timestamp = .now(io, .awake);

    var buffers = try zml_mem.bufferize(allocator, Model, &self.model);
    errdefer self.unloadBuffers(&buffers, allocator);

    var loader: zml_io.Loader = try .init(allocator, platform, .{
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
    _ = backend;
    const experts_dtype = self.model.stack.layers[0].experts.gate_up_proj.dtype();
    const moe_backend = moe.Backend.auto(platform, null, experts_dtype) catch .stablehlo;
    log.info("Selected MoE backend: {}", .{moe_backend});

    const dtype = self.model.stack.embed_tokens.weight.dtype();
    return inference.compile(allocator, io, platform, self, .{
        .seqlen = @intCast(seqlen),
        .canvas_length = self.config().canvas_length,
        .hidden_size = self.config().text_config.hidden_size,
        .kv_cache = .init(self.config(), @intCast(seqlen), dtype, meshes.model),
        .settings = .init(self.generation()),
        .moe_backend = moe_backend,
        .meshes = meshes,
    }, progress);
}
