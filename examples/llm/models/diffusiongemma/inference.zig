const std = @import("std");
const mem = std.mem;
const zml = @import("zml");
const moe = zml.moe;

const common = @import("../common.zig");
const encoder = @import("encoder.zig");
const LoadedModel = @import("LoadedModel.zig");
const Model = @import("Model.zig");

const log = std.log.scoped(.diffusiongemma);

pub const num_candidates = 5;

pub const CompiledModel = struct {
    loaded_model: *const LoadedModel,
    prefill: PrefillExe,
    shapes: Shapes,

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
    }
};

pub const Shapes = struct {
    seqlen: u32,
    moe_backend: moe.Backend,
    meshes: common.Meshes,
};

pub const Prefill = struct {
    pub const Input = struct {
        weights: Model,

        tokens: zml.Tensor,
        len: zml.Tensor,
        moe_backend: moe.Backend,
    };

    pub const Output = struct { candidates: zml.Tensor, probabilities: zml.Tensor };

    pub fn forward(input: Input) Output {
        const stack = input.weights.stack;
        const hidden = encoder.forward(stack, input.tokens, input.moe_backend);

        const last = hidden.slice(.s, .dyn(input.len.convert(.i32).addConstant(-1), 1));
        const probabilities = stack.logits(last).softmax(.voc);
        const best = probabilities.topK(.{ .topk = .voc }, num_candidates, .{});
        return .{ .candidates = best.indices.convert(.u32), .probabilities = best.values };
    }
};

pub const PrefillExe = zml.FnExe(Prefill.forward);

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

    const prefill = try context.compile(PrefillExe, "prefill", Prefill.Input{
        .weights = loaded_model.model,
        .tokens = .init(.{ .s = shapes.seqlen }, .u32),
        .len = .init(.{}, .u32),
        .moe_backend = shapes.moe_backend,
    });

    return .{ .loaded_model = loaded_model, .prefill = prefill, .shapes = shapes };
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
