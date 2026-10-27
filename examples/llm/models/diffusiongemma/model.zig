const std = @import("std");
const json = std.json;

const zml = @import("zml");
const stdx = zml.stdx;

const inference = @import("inference.zig");
const session = @import("session.zig");

pub const Config = struct {
    boi_token_id: u64,
    eoi_token_id: u64,
    text_config: DiffusionGemmaTextConfig,

    pub const DiffusionGemmaTextConfig = struct {
        model_type: enum { diffusion_gema_text },
        vocab_size: u64,
        hidden_size: u64,
        intermediate_size: u64,
        num_hidden_layers: u64,
        num_attention_heads: u64,
        num_key_value_heads: u64,
        head_dim: u64,
        hidden_activation_str: []const u8,
        max_position_embeddings: u64,
        initializer_range: f32,
        rms_norm_eps: f32,
        pad_token_id: ?u64,
        eos_token_id: ?[]u64,
        bos_token_id: ?u64,
        tie_word_embeddings: bool,
        rope_parameters: ?json.ObjectMap,
        attention_bias: bool,
        attention_dropout: ?f32,
        sliding_window: u64,
        layer_types: ?[]const u8,
        final_logit_softcapping: f32,
        use_bidirectional_attention: ?enum { all, vision },
        num_experts: u64,
        top_k_experts: u64,
        moe_intermediate_size: u64,
    };
};

pub const TextModel = struct {};

pub const TransformerLayer = struct {};

pub const SelfAttention = struct {};

pub const Mlp = struct {};

pub const Router = struct {};

pub const Experts = struct {};

pub const RmsNorm = struct {};

pub const LmHead = struct {};

pub const KvCache = struct {};

pub const Buffers = struct {};
pub const Model = struct {};
pub const LoadedModel = struct {};
