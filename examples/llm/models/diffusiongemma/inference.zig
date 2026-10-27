const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.diffusiongemma);
const Phase = common.Phase;

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    moe_backend: zml.moe.Backend,
    seqlen: usize,
    meshes: common.Meshes,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, moe_backend: zml.moe.Backend, meshes: common.Meshes) CompilationParameters {
        return .{
            .prefill_tokens = .init(.{ .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(config, seqlen, mdl.text_model.embed_tokens.weight.dtype(), meshes.model),
            .rng = .init(),
            .moe_backend = moe_backend,
            .seqlen = seqlen,
            .meshes = meshes,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const Args = struct {
    io: std.Io,
    tokens_buf: *zml.Buffer,
    next_token_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    active_length_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill: KernelExe,
    decode: KernelExe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        mdl: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const prefill = try compileKernel(allocator, io, platform, mdl, parameters, @intCast(parameters.prefill_tokens.dim(.s)), .prefill, progress);
        errdefer prefill.deinit();
        const decode = try compileKernel(allocator, io, platform, mdl, parameters, @intCast(parameters.decode_tokens.dim(.s)), .decode, progress);

        return .{
            .loaded_model = loaded_model,
            .prefill = prefill,
            .decode = decode,
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

const Forward = struct {
    pub const Input = struct {
        weights: model.Model,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: model.KvCache,
        rng: zml.Tensor.Rng,
        moe_backend: zml.moe.Backend,
    };

    pub const Output = struct {
        tokens: zml.Tensor,
        kv_cache: model.KvCache,
        rng: zml.Tensor.Rng,
    };

    pub fn forward(input: Input) Output {
        const tokens, const kv_cache, const rng = input.weights.forward(
            input.tokens,
            input.token_index,
            input.active_length,
            input.kv_cache,
            input.rng,
            input.moe_backend,
        );
        return .{
            .tokens = if (input.tokens.dim(.s) == 1) tokens.reuseBuffer(input.tokens) else tokens,
            .kv_cache = kv_cache.reuseBuffer(input.kv_cache),
            .rng = .{ ._state = rng._state.reuseBuffer(input.rng._state) },
        };
    }
};

pub const KernelExe = zml.FnExe(Forward.forward);

pub const KernelRunner = struct {
    forward: KernelExe.Runner(.{.weights}),

    pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const model.Buffers) !KernelRunner {
        return .{ .forward = try .init(exe, allocator, .{ .weights = buffers.* }) };
    }

    pub fn deinit(self: *KernelRunner, allocator: std.mem.Allocator) void {
        self.forward.deinit(allocator);
    }
};

pub fn run(runner: *KernelRunner, args: Args) void {
    runner.forward.run(args.io, .{
        .inputs = .{
            .tokens = args.tokens_buf.*,
            .token_index = args.token_index_buf.*,
            .active_length = args.active_length_buf.*,
            .kv_cache = args.kv_cache_buffers.*,
            .rng = args.rng_buffers.*,
        },
        .outputs = .{
            .tokens = args.next_token_buf,
            .kv_cache = args.kv_cache_buffers,
            .rng = args.rng_buffers,
        },
    }, .{});
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mdl: model.Model,
    parameters: CompilationParameters,
    seqlen: usize,
    phase: Phase,
    progress: *std.Progress.Node,
) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("forward"), 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "forward", io, from);

    return KernelExe.compile(allocator, io, platform, .{
        .meshes = &parameters.meshes.all(),
        .program_name = phase.programName("diffusiongemma", "forward"),
    }, .{.{
        .weights = mdl,
        .tokens = zml.Tensor.init(.{ .s = seqlen }, .u32),
        .token_index = parameters.token_index,
        .active_length = .init(.{}, .u32),
        .kv_cache = parameters.kv_cache,
        .rng = parameters.rng,
        .moe_backend = parameters.moe_backend,
    }});
}
