const std = @import("std");
const math = std.math;
const zml = @import("zml");
const moe = zml.moe;
const nn = zml.nn;
const zml_io = zml.io;

const Config = @import("Config.zig");

pub fn linear(store: zml_io.TensorStore.View, partitioning: anytype) nn.Linear {
    return .init(store.createTensor("weight", .{ .dout, .d }, .model, partitioning), null, .d);
}

pub const RmsNorm = struct {
    weight: ?zml.Tensor,
    eps: f32,

    pub fn init(store: zml_io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, .model, .{}), .eps = eps };
    }

    pub fn initWithoutScale(eps: f32) RmsNorm {
        return .{ .weight = null, .eps = eps };
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const normalized = nn.rmsNorm(x.convert(.f32), -1, self.eps);
        if (self.weight == null) {
            return normalized.convert(x.dtype());
        }

        const weight = self.weight.?.convert(.f32).withTags(.{x.shape().tag(-1)});
        return normalized.mul(weight.broad(normalized.shape())).convert(x.dtype());
    }
};

pub const Mlp = struct {
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,

    pub fn init(store: zml_io.TensorStore.View) Mlp {
        return .{
            .gate_proj = linear(store.withPrefix("gate_proj"), .{ .dout = .model }),
            .up_proj = linear(store.withPrefix("up_proj"), .{ .dout = .model }),
            .down_proj = linear(store.withPrefix("down_proj"), .{ .d = .model }),
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const gate = self.gate_proj.forward(x, x.dtype());
        const up = self.up_proj.forward(x, x.dtype());

        const activated = gate.convert(.f32).gelu().convert(x.dtype());
        const hidden = activated.mul(up).rename(.{ .dout = .d });
        return self.down_proj.forward(hidden, x.dtype()).rename(.{ .dout = .d });
    }
};

pub const Attention = struct {
    q_proj: nn.Linear,
    k_proj: nn.Linear,

    v_proj: ?nn.Linear,
    o_proj: nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    num_heads: u32,
    num_kv_heads: u32,
    rope: nn.RopeOpts,

    pub fn init(store: zml_io.TensorStore.View, text: Config.Text, kind: Config.LayerType) Attention {
        var num_kv_heads = text.num_key_value_heads;

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

        if (kind == .full_attention) {
            num_kv_heads = text.num_global_key_value_heads;
        }

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

    pub const Projected = struct {
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
    };

    pub fn project(self: Attention, x: zml.Tensor, positions: zml.Tensor) Projected {
        const q_heads = .{ .h = self.num_heads, .hd = .auto };
        const kv_heads = .{ .h = self.num_kv_heads, .hd = .auto };
        const q = self.q_proj.forward(x, x.dtype()).splitAxis(.dout, q_heads);
        const k = self.k_proj.forward(x, x.dtype()).splitAxis(.dout, kv_heads);

        var v = k;
        if (self.v_proj) |v_proj| {
            v = v_proj.forward(x, x.dtype()).splitAxis(.dout, kv_heads);
        }

        return .{
            .q = nn.rope(self.q_norm.forward(q), positions, self.rope),
            .k = nn.rope(self.k_norm.forward(k), positions, self.rope),
            .v = self.v_norm.forward(v),
        };
    }

    pub fn attend(self: Attention, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, visible: zml.Tensor) zml.Tensor {
        const out = nn.sdpa(q, k, v, .{
            .attn_mask = additiveMask(visible, q.dtype()),

            .scale = .scalar(1.0, q.dtype()),
        });
        const merged = out.merge(.{ .d = .{ .h, .hd } });
        return self.o_proj.forward(merged, q.dtype()).rename(.{ .dout = .d });
    }
};

pub fn additiveMask(visible: zml.Tensor, dtype: zml.DataType) zml.Tensor {
    return visible.select(zml.Tensor.scalar(0, dtype), zml.Tensor.scalar(-math.inf(f32), dtype));
}

pub const Router = struct {
    norm: RmsNorm,
    proj: nn.Linear,
    scale: zml.Tensor,
    per_expert_scale: zml.Tensor,
    top_k: u32,

    pub fn init(store: zml_io.TensorStore.View, text: Config.Text) Router {
        return .{
            .norm = .initWithoutScale(text.rms_norm_eps),
            .proj = .init(store.createTensor("proj.weight", .{ .expert, .d }, .model, .{}), null, .d),
            .scale = store.createTensor("scale", .{.d}, .model, .{}),
            .per_expert_scale = store.createTensor("per_expert_scale", .{.expert}, .model, .{}),
            .top_k = text.top_k_experts,
        };
    }

    pub const Routing = struct {
        weights: zml.Tensor,
        indices: zml.Tensor,
    };

    pub fn forward(self: Router, x: zml.Tensor) Routing {
        const inverse_root = 1.0 / @sqrt(@as(f32, @floatFromInt(x.dim(.d))));
        const normalized = self.norm.forward(x);
        const scaled = normalized.mul(self.scale.convert(x.dtype()).broad(x.shape())).scale(inverse_root);

        const probabilities = self.proj.forward(scaled, x.dtype()).convert(.f32).softmax(.expert);
        const best = probabilities.topK(.{ .topk = .expert }, self.top_k, .{});
        const indices = best.indices.convert(.i32);

        const weights = best.values.div(best.values.sum(.topk).broad(best.values.shape()));
        const gain = self.per_expert_scale.convert(.f32).gather(.{ .expert = indices }, .{});
        return .{ .weights = weights.mul(gain), .indices = indices };
    }
};

pub const Experts = struct {
    gate_up_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml_io.TensorStore.View) Experts {
        const sharded = .{ .expert = .experts };
        return .{
            .gate_up_proj = store.createTensor("gate_up_proj", .{ .expert, .dout, .d }, .experts, sharded),
            .down_proj = store.createTensor("down_proj", .{ .expert, .d, .dout }, .experts, sharded),
        };
    }

    pub fn forward(self: Experts, x: zml.Tensor, routing: Router.Routing, backend: moe.Backend) zml.Tensor {
        const out = moe.forwardMoe(
            x.rename(.{ .s = .t }),
            routing.indices.rename(.{ .s = .t }),
            routing.weights.rename(.{ .s = .t }),
            .init(self.gate_up_proj, null, .d),
            .init(self.down_proj, null, .dout),
            backend,
            .{ .activation = .geglu_tanh, .quantize_input = false, .routing_weight_placement = .after_down },
        );
        return out.rename(.{ .t = .s });
    }
};
