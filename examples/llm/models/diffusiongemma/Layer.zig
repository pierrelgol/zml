const std = @import("std");
const zml = @import("zml");
const blocks = @import("blocks.zig");
const Config = @import("Config.zig");
const TensorStore = zml.io.TensorStore;
pub const Layer = @This();

kind: Config.LayerType,
cache_index: u32,

input_layernorm: blocks.RmsNorm,
self_attn: blocks.Attention,
post_attention_layernorm: blocks.RmsNorm,

pre_feedforward_layernorm: blocks.RmsNorm,
mlp: blocks.Mlp,
post_feedforward_layernorm_1: blocks.RmsNorm,

router: blocks.Router,
pre_feedforward_layernorm_2: blocks.RmsNorm,

experts: blocks.Experts,
post_feedforward_layernorm_2: blocks.RmsNorm,

post_feedforward_layernorm: blocks.RmsNorm,
encoder_scalar: zml.Tensor,

pub fn init(
    store: TensorStore.View,
    encoder_store: TensorStore.View,
    text: Config.Text,
    kind: Config.LayerType,
    cache_index: u32,
) !Layer {
    const epsilon = text.rms_norm_eps;

    return .{
        .kind = kind,
        .cache_index = cache_index,
        .input_layernorm = .init(store.withPrefix("input_layernorm"), epsilon),
        .self_attn = .init(store.withPrefix("self_attn"), text, kind),
        .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), epsilon),
        .pre_feedforward_layernorm = .init(store.withPrefix("pre_feedforward_layernorm"), epsilon),
        .mlp = .init(store.withPrefix("mlp")),
        .post_feedforward_layernorm_1 = .init(store.withPrefix("post_feedforward_layernorm_1"), epsilon),
        .router = .init(store.withPrefix("router"), text),
        .pre_feedforward_layernorm_2 = .init(store.withPrefix("pre_feedforward_layernorm_2"), epsilon),
        .experts = .init(store.withPrefix("experts")),
        .post_feedforward_layernorm_2 = .init(store.withPrefix("post_feedforward_layernorm_2"), epsilon),
        .post_feedforward_layernorm = .init(store.withPrefix("post_feedforward_layernorm"), epsilon),
        .encoder_scalar = encoder_store.createTensor("layer_scalar", .{.one}, .model, .{}),
    };
}
