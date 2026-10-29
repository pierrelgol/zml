const std = @import("std");
const ascii = std.ascii;
const math = std.math;
const mem = std.mem;
const zml = @import("zml");
const tokenizer = zml.tokenizer;

const Buffers = @import("../diffusiongemma.zig").Buffers;
const Config = @import("Config.zig");
const Generation = @import("Generation.zig");
const inference = @import("inference.zig");
const KvCache = @import("KvCache.zig");

const log = std.log.scoped(.diffusiongemma);

const EncodeRunner = inference.EncodeExe.Runner(.{.weights});
const DenoiseRunner = inference.DenoiseExe.Runner(.{.weights});

const Session = @This();

allocator: mem.Allocator,
io: std.Io,
platform: *const zml.Platform,
text_tokenizer: tokenizer.Tokenizer,
config: Config,
generation: Generation,
shapes: inference.Shapes,

encode_prompt: EncodeRunner,
encode_canvas: EncodeRunner,
denoise: DenoiseRunner,

kv_cache: zml.Bufferized(KvCache),

rng: zml.Bufferized(zml.Tensor.Rng),
random: std.Random.DefaultPrng,

pub fn init(
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    text_tokenizer: tokenizer.Tokenizer,
    compiled: *inference.CompiledModel,
    buffers: *Buffers,
) !Session {
    var kv_cache = try compiled.shapes.kv_cache.initBuffer(io, platform);
    errdefer KvCache.deinitBuffer(&kv_cache);

    var encode_prompt: EncodeRunner = try .init(&compiled.encode_prompt, allocator, .{ .weights = buffers.* });
    errdefer encode_prompt.deinit(allocator);
    var encode_canvas: EncodeRunner = try .init(&compiled.encode_canvas, allocator, .{ .weights = buffers.* });
    errdefer encode_canvas.deinit(allocator);

    const seed: u64 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
    var rng = try zml.Tensor.Rng.init().initBuffer(io, platform, seed);
    errdefer zml.Tensor.Rng.deinitBuffer(&rng);

    return .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .text_tokenizer = text_tokenizer,
        .config = compiled.loaded_model.config(),
        .generation = compiled.loaded_model.generation(),
        .shapes = compiled.shapes,
        .encode_prompt = encode_prompt,
        .encode_canvas = encode_canvas,
        .denoise = try .init(&compiled.denoise, allocator, .{ .weights = buffers.* }),
        .kv_cache = kv_cache,
        .rng = rng,
        .random = .init(seed),
    };
}

pub fn deinit(self: *Session) void {
    self.encode_prompt.deinit(self.allocator);
    self.encode_canvas.deinit(self.allocator);
    self.denoise.deinit(self.allocator);
    KvCache.deinitBuffer(&self.kv_cache);
    zml.Tensor.Rng.deinitBuffer(&self.rng);
}

pub fn tokenizePrompt(self: *const Session, allocator: mem.Allocator, prompt: []const u8) ![]const u32 {
    const prefix: []const u32 = &.{self.config.text_config.bos_token_id};
    return self.tokenizeUserTurn(allocator, prefix, prompt);
}

pub fn tokenizeTurn(self: *const Session, allocator: mem.Allocator, prompt: []const u8) ![]const u32 {
    const prefix: []const u32 = &.{ try self.tokenId("<turn|>"), try self.tokenId("\n") };
    return self.tokenizeUserTurn(allocator, prefix, prompt);
}

fn tokenizeUserTurn(self: *const Session, allocator: mem.Allocator, prefix: []const u32, prompt: []const u8) ![]const u32 {
    var encoder = try self.text_tokenizer.encoder();
    defer encoder.deinit();

    const start_of_turn = try self.tokenId("<|turn>");
    const end_of_turn = try self.tokenId("<turn|>");
    const newline = try self.tokenId("\n");

    var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
    errdefer tokens.deinit();
    const writer = &tokens.writer;

    try encoder.appendTokens(writer, prefix);
    try encoder.appendTokens(writer, &.{start_of_turn});
    try encoder.encode(writer, "user");
    try encoder.appendTokens(writer, &.{newline});
    try encoder.encode(writer, mem.trim(u8, prompt, &ascii.whitespace));
    try encoder.appendTokens(writer, &.{ end_of_turn, newline, start_of_turn });
    try encoder.encode(writer, "model");
    try encoder.appendTokens(writer, &.{newline});
    return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
}

fn tokenId(self: *const Session, text: []const u8) !u32 {
    return self.text_tokenizer.tokenId(text) orelse error.NoSuchToken;
}

fn upload(self: *const Session, shape: zml.Shape, values: anytype) !zml.Buffer {
    return zml.Buffer.fromBytes(self.io, self.platform, shape, mem.sliceAsBytes(values));
}

pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
    const seqlen = self.shapes.seqlen;
    if (all_tokens.len == 0 or all_tokens.len > seqlen) {
        return error.PromptTooLong;
    }

    const padded = try self.allocator.alloc(u32, seqlen);
    defer self.allocator.free(padded);
    @memset(padded, 0);
    @memcpy(padded[0..all_tokens.len], all_tokens);

    const started: std.Io.Timestamp = .now(self.io, .awake);
    try self.encode(&self.encode_prompt, padded, 0);
    log.info("Prefilled {d} tokens [{f}]", .{ all_tokens.len, started.untilNow(self.io, .awake) });
}

pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
    var decoder = try self.text_tokenizer.decoder();
    defer decoder.deinit();
    var chunk: [256]u8 = undefined;

    const canvas = try self.allocator.alloc(u32, self.shapes.canvas_length);
    defer self.allocator.free(canvas);

    var total_steps: u32 = 0;
    var generated: usize = 0;
    generation: while (all_tokens.items.len + canvas.len <= self.shapes.seqlen) {
        const start: u32 = @intCast(all_tokens.items.len);
        total_steps += try self.denoiseCanvas(start, canvas);

        for (canvas) |id| {
            if (self.generation.isEndOfTurn(id)) {
                break :generation;
            }
            try stdout.writeAll(try decoder.feedOne(id, &chunk));
            try all_tokens.append(self.allocator, id);
            generated += 1;
        }
        try stdout.flush();

        try self.encode(&self.encode_canvas, canvas, start);
    } else {
        log.warn("Reached seqlen={d} before an end of turn token", .{self.shapes.seqlen});
    }

    try stdout.writeAll(try decoder.finalize(&chunk));
    try stdout.flush();
    const tokens_per_step = @as(f64, @floatFromInt(generated)) / @as(f64, @floatFromInt(@max(total_steps, 1)));
    log.info("\n{d} tokens in {d} denoising steps, {d:.2} tokens per forward pass", .{ generated, total_steps, tokens_per_step });
}

fn denoiseCanvas(self: *Session, start: u32, canvas: []u32) !u32 {
    const vocab_size = self.config.text_config.vocab_size;
    for (canvas) |*id| {
        id.* = self.random.random().uintLessThan(u32, vocab_size);
    }
    var canvas_buffer = try self.upload(.init(.{canvas.len}, .u32), canvas);
    defer canvas_buffer.deinit();

    const history_length = self.shapes.settings.historyLength();
    const history = try self.allocator.alloc(u32, history_length * canvas.len);
    defer self.allocator.free(history);
    @memset(history, math.maxInt(u32));
    var history_buffer = try self.upload(.init(.{ history_length, canvas.len }, .u32), history);
    defer history_buffer.deinit();

    const hidden_size = self.shapes.hidden_size;
    const soft = try self.allocator.alloc(u16, canvas.len * hidden_size);
    defer self.allocator.free(soft);
    @memset(soft, 0);
    var soft_buffer = try self.upload(.init(.{ canvas.len, hidden_size }, .bf16), soft);
    defer soft_buffer.deinit();

    var start_buffer: zml.Buffer = try .scalar(self.io, self.platform, start, .u32);
    defer start_buffer.deinit();

    var argmax: ?zml.Buffer = null;
    defer {
        if (argmax) |*buffer| {
            buffer.deinit();
        }
    }

    var steps: u32 = 0;
    var step_index = self.generation.max_denoising_steps;
    while (step_index >= 1) : (step_index -= 1) {
        var step_buffer: zml.Buffer = try .scalar(self.io, self.platform, step_index, .u32);
        defer step_buffer.deinit();

        if (argmax) |*buffer| {
            buffer.deinit();
        }
        var next_argmax: zml.Buffer = undefined;
        var done: zml.Buffer = undefined;
        defer done.deinit();
        self.denoise.run(self.io, .{
            .inputs = .{
                .canvas = canvas_buffer,
                .history = history_buffer,
                .soft = soft_buffer,
                .start = start_buffer,
                .step_index = step_buffer,
                .kv_cache = self.kv_cache,
                .rng = self.rng,
            },
            .outputs = .{
                .canvas = &canvas_buffer,
                .history = &history_buffer,
                .soft = &soft_buffer,
                .argmax = &next_argmax,
                .done = &done,
                .rng = &self.rng,
            },
        }, .{});
        argmax = next_argmax;
        steps += 1;

        if (try done.getValue(u8, self.io) != 0) {
            break;
        }
    }

    try argmax.?.toSlice(self.io, .init(argmax.?.shape(), mem.sliceAsBytes(canvas)));
    return steps;
}

fn encode(self: *Session, runner: *EncodeRunner, tokens: []const u32, start: u32) !void {
    var tokens_buffer = try self.upload(.init(.{tokens.len}, .u32), tokens);
    defer tokens_buffer.deinit();
    var start_buffer: zml.Buffer = try .scalar(self.io, self.platform, start, .u32);
    defer start_buffer.deinit();

    runner.run(self.io, .{
        .inputs = .{ .tokens = tokens_buffer, .start = start_buffer, .kv_cache = self.kv_cache },
        .outputs = .{ .kv_cache = &self.kv_cache },
    }, .{});
}
