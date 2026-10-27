const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled_model: *inference.CompiledModel,
    prefill: inference.KernelRunner,
    decode: inference.KernelRunner,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    token_index_buffers: []zml.Buffer,
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    tokenizer: zml.tokenizer.Tokenizer,
    config: *const model.Config,
    seqlen: u32,
    last_generated_token: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const token_index_buffers = try allocator.alloc(zml.Buffer, compiled_model.params.seqlen);
        errdefer allocator.free(token_index_buffers);
        var initialized_token_index_buffers: usize = 0;
        errdefer {
            for (token_index_buffers[0..initialized_token_index_buffers]) |*token_index_buffer| {
                token_index_buffer.deinit();
            }
        }
        for (token_index_buffers, 0..) |*token_index_buffer, i| {
            token_index_buffer.* = try zml.Buffer.scalar(io, platform, i, .u32);
            initialized_token_index_buffers = i + 1;
        }

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.init().initBuffer(io, platform, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        var prefill = try inference.KernelRunner.init(allocator, &compiled_model.prefill, model_buffers);
        errdefer prefill.deinit(allocator);
        const decode = try inference.KernelRunner.init(allocator, &compiled_model.decode, model_buffers);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .compiled_model = compiled_model,
            .prefill = prefill,
            .decode = decode,
            .kv_cache_buffers = kv_cache_buffers,
            .token_index_buffers = token_index_buffers,
            .rng_buffers = rng_buffers,
            .tokenizer = tokenizer,
            .config = &compiled_model.loaded_model.parsed_config.value,
            .seqlen = @intCast(compiled_model.params.seqlen),
        };
    }

    pub fn deinit(self: *Session) void {
        self.prefill.deinit(self.allocator);
        self.decode.deinit(self.allocator);
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        for (self.token_index_buffers) |*token_index_buffer| {
            token_index_buffer.deinit();
        }
        self.allocator.free(self.token_index_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
    }

    /// `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n`
    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        const bos: u32 = @intCast(self.config.text_config.bos_token_id orelse return error.NoSuchToken);
        return self.tokenizeUserTurn(allocator, &.{bos}, prompt);
    }

    /// The previous model turn stopped on `<turn|>` which was not kept, so close it first.
    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        const end_turn = self.tokenizer.tokenId("<turn|>") orelse return error.NoSuchToken;
        const newline = self.tokenizer.tokenId("\n") orelse return error.NoSuchToken;
        return self.tokenizeUserTurn(allocator, &.{ end_turn, newline }, prompt);
    }

    fn tokenizeUserTurn(self: *const Session, allocator: std.mem.Allocator, prefix: []const u32, prompt: []const u8) ![]const u32 {
        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const start_turn = self.tokenizer.tokenId("<|turn>") orelse return error.NoSuchToken;
        const end_turn = self.tokenizer.tokenId("<turn|>") orelse return error.NoSuchToken;
        const newline = self.tokenizer.tokenId("\n") orelse return error.NoSuchToken;

        var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
        try tokens.ensureUnusedCapacity(prompt.len);

        const w: *std.Io.Writer = &tokens.writer;
        try encoder.appendTokens(w, prefix);
        try encoder.appendTokens(w, &.{start_turn});
        try encoder.encode(w, "user");
        try encoder.appendTokens(w, &.{newline});
        try encoder.encode(w, prompt);
        try encoder.appendTokens(w, &.{ end_turn, newline, start_turn });
        try encoder.encode(w, "model");
        try encoder.appendTokens(w, &.{newline});

        return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const prefill_tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{self.seqlen}, .u32));
        defer prefill_tokens_slice.free(self.allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, prefill_tokens_slice);
        defer prefill_tokens_buffer.deinit();
        var active_length_buffer: zml.Buffer = try .scalar(self.io, self.platform, all_tokens.len, .u32);
        defer active_length_buffer.deinit();

        var next_token_buffer: zml.Buffer = undefined;
        inference.run(&self.prefill, .{
            .io = self.io,
            .tokens_buf = &prefill_tokens_buffer,
            .next_token_buf = &next_token_buffer,
            .token_index_buf = &self.token_index_buffers[0],
            .active_length_buf = &active_length_buffer,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .rng_buffers = &self.rng_buffers,
        });

        defer next_token_buffer.deinit();
        self.last_generated_token = try next_token_buffer.getValue(u32, self.io);
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder: zml.tokenizer.Tokenizer.Decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const decoder_out_buffer: []u8 = try self.allocator.alloc(u8, 256);
        defer self.allocator.free(decoder_out_buffer);

        var last_token_id: u32 = self.last_generated_token;
        var current_token_buffer: zml.Buffer = try .fromBytes(self.io, self.platform, .init(.{ .s = 1 }, .u32), @ptrCast(&last_token_id));
        defer current_token_buffer.deinit();
        var active_length_buffer: zml.Buffer = try .scalar(self.io, self.platform, 1, .u32);
        defer active_length_buffer.deinit();

        generation: while (true) {
            if (isEosToken(self.config, last_token_id)) break :generation;

            try stdout.writeAll(try decoder.feedOne(last_token_id, decoder_out_buffer));
            try stdout.flush();

            try all_tokens.append(self.allocator, last_token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            inference.run(&self.decode, .{
                .io = self.io,
                .tokens_buf = &current_token_buffer,
                .next_token_buf = &current_token_buffer,
                .token_index_buf = &self.token_index_buffers[all_tokens.items.len - 1],
                .active_length_buf = &active_length_buffer,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
            });
            last_token_id = try current_token_buffer.getValue(u32, self.io);
        }

        try stdout.writeAll(try decoder.finalize(decoder_out_buffer));
        try stdout.flush();
    }
};

fn isEosToken(config: *const model.Config, token_id: u32) bool {
    return switch (config.eos_token_id) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
    };
}
