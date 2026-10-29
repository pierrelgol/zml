const std = @import("std");
const zml = @import("zml");
const nn = zml.nn;
const TensorStore = zml.io.TensorStore;
const Config = @import("Config.zig");

pub fn linear(store: TensorStore.View, partitioning: anytype) nn.Linear {
    return .init(
        store.createTensor("weight", .{ .dout, .d }, .model, partitioning),
        null,
        .d,
    );
}

pub const RmsNorm = struct {
    weight: ?zml.Tensor,
    eps: f32,

    pub fn init(store: TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .model, .{}),
            .eps = eps,
        };
    }

    pub fn initWithoutScale(eps: f32) RmsNorm {
        return .{
            .weight = null,
            .eps = eps,
        };
    }
};

pub const Mlp = struct {
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,

    pub fn init(store: TensorStore.View) Mlp {
        return .{
            .gate_proj = linear(store.withPrefix("gate_proj"), .{ .dout = .model }),
            .up_proj = linear(store.withPrefix("up_proj"), .{ .dout = .model }),
            .down_proj = linear(store.withPrefix("down_proj"), .{ .d = .model }),
        };
    }
};

pub const Attention = struct {
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    // full_attention layers don't need value projection
    // they reuse the key to proj
    v_proj: ?nn.Linear,
    o_proj: nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,

    num_heads: u32,
    num_kv_heads: u32,
    rope: nn.RopeOpts,

    pub fn init(store: TensorStore.View, text: Config.Text, kind: Config.LayerType) Attention {
        const num_kv_heads = text.num_key_value_heads;

        const rope: nn.RopeOpts = switch (kind) {
            .sliding_attention => .{
                .layout = .real_im_pass,
                .scaling = text.rope_parameters.sliding_attention,
            },
            .full_attention => .{
                .layout = .real_pass_im_pass,
                .scaling = text.rope_parameters.full_attention,
            },
        };

        var v_proj: ?nn.Linear = null;
        if (store.hasKey("v_proj.weight")) {
            v_proj = linear(store.withPrefix("v_proj"), .{ .dout = .model });
        }

        return .{
            .q_proj = linear(store.withPrefix("q_proj"), .{ .dout = .model }),
            .k_proj = linear(store.withPrefix("k_proj"), .{ .dout = .model }),
            .v_proj = v_proj,
            .o_proj = linear(store.withPrefix("o_proj"), .{ .d = .model }),
            .q_norm = .init(store.withPrefix("q_norm"), text.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), text.rms_norm_eps),
            .v_norm = .initWithoutScale(text.rms_norm_eps),
            .num_heads = text.num_attention_heads,
            .num_kv_heads = num_kv_heads,
            .rope = rope,
        };
    }
};

pub const Router = struct {
    norm: RmsNorm,
    proj: nn.Linear,
    scale: zml.Tensor,
    per_expert_scale: zml.Tensor,
    top_k: u32,

    pub fn init(store: TensorStore.View, text: Config.Text) Router {
        return .{
            .norm = .initWithoutScale(text.rms_norm_eps),
            .proj = .init(store.createTensor("proj.weight", .{ .expert, .d }, .model, .{}), null, .d),
            .scale = store.createTensor("scale", .{.d}, .model, .{}),
            .per_expert_scale = store.createTensor("per_expert_scale", .{.expert}, .model, .{}),
            .top_k = text.top_k_experts,
        };
    }
};

pub const Experts = struct {
    gate_up_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: TensorStore.View) Experts {
        const sharded = .{ .expert = .experts };
        return .{
            .gate_up_proj = store.createTensor("gate_up_proj", .{ .expert, .dout, .d }, .experts, sharded),
            .down_proj = store.createTensor("down_proj", .{ .expert, .d, .dout }, .experts, sharded),
        };
    }
};
