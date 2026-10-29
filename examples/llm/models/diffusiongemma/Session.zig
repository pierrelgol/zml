const std = @import("std");
const ascii = std.ascii;
const mem = std.mem;
const zml = @import("zml");
const tokenizer = zml.tokenizer;

const Buffers = @import("../diffusiongemma.zig").Buffers;
const Config = @import("Config.zig");
const Generation = @import("Generation.zig");
const inference = @import("inference.zig");
const log = std.log.scoped(.diffusiongemma_session);

const PrefillRunner = inference.PrefillExe.Runner(.{.weights});

pub const Session = @This();

allocator: mem.Allocator,
io: std.Io,
platform: *const zml.Platform,
text_tokenizer: tokenizer.Tokenizer,
config: Config,
generation: Generation,
shapes: inference.Shapes,
prefill: PrefillRunner,

pub fn init(
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    text_tokenizer: tokenizer.Tokenizer,
    compiled: *inference.CompiledModel,
    buffers: *Buffers,
) !Session {
    return .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .text_tokenizer = text_tokenizer,
        .config = compiled.loaded_model.config(),
        .generation = compiled.loaded_model.generation(),
        .shapes = compiled.shapes,
        .prefill = try init(&compiled.prefill, allocator, .{ .weights = buffers.* }),
    };
}

pub fn deinit(self: *Session) void {
    self.prefill.deinit(self.allocator);
}

pub fn upload(self: *const Session, shape: zml.Shape, values: anytype) !zml.Buffer {
    return zml.Buffer.fromBytes(self.io, self.platform, shape, mem.sliceAsBytes(values));
}

pub fn tokenizePrompt(self: *const Session, allocator: mem.Allocator, prompt: []const u8) ![]const u32 {
    const prefix: []const u32 = &.{self.config.text_config.bos_token_id};
    return self.tokenizeUserTurn(allocator, prefix, prompt);
}

pub fn tokenizeTurn(self: *const Session, allocator: mem.Allocator, prompt: []const u8) ![]const u32 {
    const prefix: []const u32 = &.{ try self.tokenId("<turn|>"), try self.tokenId("\n") };
    return self.tokenizeUserTurn(allocator, prefix, prompt);
}

pub fn tokenizeUserTurn(self: *const Session, allocator: mem.Allocator, prefix: []const u32, prompt: []const u8) ![]const u32 {
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

pub fn tokenId(self: *const Session, text: []const u8) !u32 {
    return self.text_tokenizer.tokenId(text) orelse error.NoSuchToken;
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

    var tokens = try self.upload(.init(.{seqlen}, .u32), padded);
    defer tokens.deinit();

    var len: zml.Buffer = try .scalar(self.io, self.platform, all_tokens.len, .u32);
    defer len.deinit();

    const started: std.Io.Timestamp = .now(self.io, .awake);

    var candidates: zml.Buffer = undefined;
    defer candidates.deinit();

    var probabilities: zml.Buffer = undefined;
    defer probabilities.deinit();

    self.prefill.run(self.io, .{
        .inputs = .{ .tokens = tokens, .len = len },
        .outputs = .{ .candidate = &candidates, .probabilities = &probabilities },
    }, .{});

    var ids: [inference.num_candidates]u32 = undefined;
    try candidates.toSlice(self.io, .init(candidates.shape(), mem.sliceAsBytes(&ids)));

    var odds: [inference.num_candidates]f32 = undefined;
    try probabilities.toSlice(self.io, .init(probabilities.shape(), mem.sliceAsBytes(&odds)));
    log.info("Prefilled {d} tokens {f}", .{ all_tokens.len, started.untilNow(self.io, .awake) });

    var decoder = try self.text_tokenizer.decoder();
    defer decoder.deinit();

    var chunk: [256]u8 = undefined;
    for (ids, odds) |id, probability| {
        const piece = try decoder.feedOne(id, &chunk);
        log.info("next token {d:>6} {s:<16}, p={d:.4}", .{ id, piece, probability });
    }
}

pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
    _ = all_tokens;
    _ = stdout;
    log.warn("TODO : seqlen : {d}", .{self.shapes.seqlen});
}
