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

pub const Session = @This();

allocator: mem.Allocator,
io: std.Io,
platform: *const zml.Platform,
text_tokenizer: tokenizer.Tokenizer,
config: Config,
generation: Generation,
seqlen: u32,

pub fn init(
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    text_tokenizer: tokenizer.Tokenizer,
    compiled: *inference.CompiledModel,
    buffers: *Buffers,
) !Session {
    _ = buffers;

    return .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .text_tokenizer = text_tokenizer,
        .config = compiled.loaded_model.config(),
        .generation = compiled.loaded_model.generation(),
        .seqlen = compiled.seqlen,
    };
}

pub fn deinit(self: *Session) void {
    _ = self;
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
    if (all_tokens.len > self.seqlen) {
        return error.PromptToolLong;
    }

    var decoder = try self.text_tokenizer.decoder();
    defer decoder.deinit();

    var text: std.Io.Writer.Allocating = .init(self.allocator);
    defer text.deinit();

    var chunk: [256]u8 = undefined;
    for (all_tokens) |id| {
        try text.writer.writeAll(try decoder.feedOne(id, &chunk));
    }

    try text.writer.writeAll(try decoder.finalize(&chunk));
    log.info("Prompt is {d} tokens out of {d}:\n{s}", .{ all_tokens.len, self.seqlen, text.written() });
}

pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
    _ = all_tokens;
    _ = stdout;
    log.warn("TODO : seqlen : {d}", .{self.seqlen});
}
