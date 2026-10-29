const std = @import("std");
const ascii = std.ascii;
const mem = std.mem;
const zml = @import("zml");
const tokenizer = zml.tokenizer;

const Buffers = @import("../diffusiongemma.zig").Buffers;
const Config = @import("Config.zig");
const Generation = @import("Generation.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.diffusiongemma);

const PrefillRunner = inference.PrefillExe.Runner(.{.weights});

const Session = @This();

allocator: mem.Allocator,
io: std.Io,
platform: *const zml.Platform,
text_tokenizer: tokenizer.Tokenizer,
config: Config,
generation: Generation,
shapes: inference.Shapes,

prefill: PrefillRunner,
rng: zml.Bufferized(zml.Tensor.Rng),

next_token: u32 = 0,

pub fn init(
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    text_tokenizer: tokenizer.Tokenizer,
    compiled: *inference.CompiledModel,
    buffers: *Buffers,
) !Session {
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
        .prefill = try .init(&compiled.prefill, allocator, .{ .weights = buffers.* }),
        .rng = rng,
    };
}

pub fn deinit(self: *Session) void {
    self.prefill.deinit(self.allocator);
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
    if (all_tokens.len == 0 or all_tokens.len > self.shapes.seqlen) {
        return error.PromptTooLong;
    }
    const started: std.Io.Timestamp = .now(self.io, .awake);
    self.next_token = try self.sampleNext(all_tokens);
    log.info("Prefilled {d} tokens [{f}]", .{ all_tokens.len, started.untilNow(self.io, .awake) });
}

pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
    var decoder = try self.text_tokenizer.decoder();
    defer decoder.deinit();
    var chunk: [256]u8 = undefined;

    var generated: usize = 0;
    while (all_tokens.items.len < self.shapes.seqlen) {
        const id = self.next_token;
        if (self.generation.isEndOfTurn(id)) {
            break;
        }
        try stdout.writeAll(try decoder.feedOne(id, &chunk));
        try stdout.flush();
        try all_tokens.append(self.allocator, id);
        generated += 1;

        self.next_token = try self.sampleNext(all_tokens.items);
    } else {
        log.warn("Reached seqlen={d} before an end of turn token", .{self.shapes.seqlen});
    }

    try stdout.writeAll(try decoder.finalize(&chunk));
    try stdout.flush();
    log.info("\n{d} tokens, one full sequence pass each", .{generated});
}

fn sampleNext(self: *Session, history: []const u32) !u32 {
    const padded = try self.allocator.alloc(u32, self.shapes.seqlen);
    defer self.allocator.free(padded);
    @memset(padded, 0);
    @memcpy(padded[0..history.len], history);

    var tokens = try self.upload(.init(.{self.shapes.seqlen}, .u32), padded);
    defer tokens.deinit();
    var len: zml.Buffer = try .scalar(self.io, self.platform, history.len, .u32);
    defer len.deinit();

    var next: zml.Buffer = undefined;
    defer next.deinit();
    self.prefill.run(self.io, .{
        .inputs = .{ .tokens = tokens, .len = len, .rng = self.rng },
        .outputs = .{ .token = &next, .rng = &self.rng },
    }, .{});

    var id: [1]u32 = undefined;
    try next.toSlice(self.io, .init(next.shape(), mem.sliceAsBytes(&id)));
    return id[0];
}
