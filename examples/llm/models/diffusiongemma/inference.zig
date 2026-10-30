const std = @import("std");
const mem = std.mem;
const zml = @import("zml");
const moe = zml.moe;

const common = @import("../common.zig");
const decoder = @import("decoder.zig");
const encoder = @import("encoder.zig");
const KvCache = @import("KvCache.zig");
const LoadedModel = @import("LoadedModel.zig");
const Model = @import("Model.zig");
const sampler = @import("sampler.zig");

const log = std.log.scoped(.diffusiongemma);

pub const CompiledModel = struct {
    loaded_model: *const LoadedModel,
    encode_prompt: EncodeExe,
    encode_canvas: EncodeExe,
    denoise: DenoiseExe,
    shapes: Shapes,

    pub fn deinit(self: *CompiledModel) void {
        self.encode_prompt.deinit();
        self.encode_canvas.deinit();
        self.denoise.deinit();
    }
};

pub const Shapes = struct {
    seqlen: u32,
    canvas_length: u32,
    hidden_size: u32,
    kv_cache: KvCache,
    settings: sampler.Settings,
    moe_backend: moe.Backend,
    meshes: common.Meshes,
};

pub const Encode = struct {
    pub const Input = struct {
        weights: Model,
        tokens: zml.Tensor,
        start: zml.Tensor,
        kv_cache: KvCache,
        moe_backend: moe.Backend,
    };

    pub const Output = struct {
        kv_cache: KvCache,
    };

    pub fn forward(input: Input) Output {
        _, const kv_cache = encoder.forward(input.weights.stack, input.tokens, input.start, input.kv_cache, input.moe_backend);
        return .{
            .kv_cache = kv_cache,
        };
    }
};

pub const Denoise = struct {
    pub const Input = struct {
        weights: Model,
        canvas: zml.Tensor,
        history: zml.Tensor,
        soft: zml.Tensor,
        start: zml.Tensor,
        step_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
        settings: sampler.Settings,
        moe_backend: moe.Backend,
    };

    pub const Output = struct {
        canvas: zml.Tensor,
        history: zml.Tensor,
        soft: zml.Tensor,
        argmax: zml.Tensor,
        done: zml.Tensor,
        rng: zml.Tensor.Rng,
    };

    pub fn forward(input: Input) Output {
        const weights = input.weights;

        const logits = decoder.forward(
            weights.stack,
            weights.self_conditioning,
            input.canvas,
            input.soft,
            input.start,
            input.kv_cache,
            input.moe_backend,
        );

        const noise, const rng = sampler.Noise.draw(input.rng, logits.dim(.s), logits.dim(.voc));
        const next = sampler.step(input.settings, logits, input.history, input.step_index, noise);

        return .{
            .canvas = next.canvas.reuseBuffer(input.canvas),
            .history = next.history.reuseBuffer(input.history),
            .soft = decoder.softEmbeddings(weights.stack, next.processed).reuseBuffer(input.soft),
            .argmax = next.argmax,
            .done = next.done.convert(.u8),
            .rng = .{
                ._state = rng._state.reuseBuffer(input.rng._state),
            },
        };
    }
};

pub const EncodeExe = zml.FnExe(Encode.forward);
pub const DenoiseExe = zml.FnExe(Denoise.forward);

pub fn compile(
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded_model: *const LoadedModel,
    shapes: Shapes,
    progress: *std.Progress.Node,
) !CompiledModel {
    const meshes = shapes.meshes.all();
    const context: CompileContext = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .meshes = &meshes,
        .progress = progress,
    };

    const encode_prompt = try context.compile(EncodeExe, "encode_prompt", encodeInput(loaded_model, shapes, shapes.seqlen));
    errdefer encode_prompt.deinit();

    const encode_canvas = try context.compile(EncodeExe, "encode_canvas", encodeInput(loaded_model, shapes, shapes.canvas_length));
    errdefer encode_canvas.deinit();

    const denoise = try context.compile(DenoiseExe, "denoise", Denoise.Input{
        .weights = loaded_model.model,
        .canvas = .init(.{ .s = shapes.canvas_length }, .u32),
        .history = .init(.{ .hist = shapes.settings.historyLength(), .s = shapes.canvas_length }, .u32),
        .soft = .init(.{ .s = shapes.canvas_length, .d = shapes.hidden_size }, .bf16),
        .start = .init(.{}, .u32),
        .step_index = .init(.{}, .u32),
        .kv_cache = shapes.kv_cache,
        .rng = zml.Tensor.Rng.init(),
        .settings = shapes.settings,
        .moe_backend = shapes.moe_backend,
    });

    return .{
        .loaded_model = loaded_model,
        .encode_prompt = encode_prompt,
        .encode_canvas = encode_canvas,
        .denoise = denoise,
        .shapes = shapes,
    };
}

fn encodeInput(loaded_model: *const LoadedModel, shapes: Shapes, num_tokens: u32) Encode.Input {
    return .{
        .weights = loaded_model.model,
        .tokens = .init(.{ .s = num_tokens }, .u32),
        .start = .init(.{}, .u32),
        .kv_cache = shapes.kv_cache,
        .moe_backend = shapes.moe_backend,
    };
}

const CompileContext = struct {
    allocator: mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    meshes: []const *const zml.Mesh,
    progress: *std.Progress.Node,

    fn compile(self: CompileContext, comptime Exe: type, comptime name: []const u8, input: anytype) !Exe {
        self.progress.increaseEstimatedTotalItems(1);
        var node = self.progress.start("Compiling " ++ name ++ "...", 1);
        defer node.end();

        const started: std.Io.Timestamp = .now(self.io, .awake);
        defer log.info("Compiled " ++ name ++ " [{f}]", .{started.untilNow(self.io, .awake)});

        const options: zml.Compiler.Options = .{ .meshes = self.meshes, .program_name = "llm_diffusiongemma_" ++ name };
        return Exe.compile(self.allocator, self.io, self.platform, options, .{input});
    }
};
