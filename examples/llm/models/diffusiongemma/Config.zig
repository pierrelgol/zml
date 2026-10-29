const std = @import("std");
const zml = @import("zml");
const nn = zml.nn;
pub const Config = @This();

text_config: Text,

pub const Text = struct {
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    // querry heads are the same, for each layers, k/v heads are different
    // sliding_attention layers, use 'num_key_value_heads', full layer use
    // the global counts
    num_attention_heads: u32,

    num_key_value_heads: u32,
    head_dim: u32,

    num_global_key_value_heads: u32,
    global_head_dim: u32,

    num_experts: u32,
    top_k_experts: u32,
    // size of the window in tokens
    sliding_window: u32,

    // one entry per layer, which is why it's an hybrid architecture
    layer_types: []const LayerType,
    rope_parameters: struct {
        sliding_attention: nn.RopeOpts.Scaling,
        full_attention: nn.RopeOpts.Scaling,
    },
    rms_norm_eps: f32,
    final_logit_softcapping: f32,
    bos_token_id: u32,

    /// ensures that the values within the config are representatbles
    pub fn validate(text: Text) !void {
        if (text.layer_types.len != text.num_hidden_layers) {
            return error.InvalidConfig;
        }

        if (text.top_k_experts == 0 or text.top_k_experts > text.num_experts) {
            return error.InvalidConfig;
        }
    }
};

pub fn validate(config: Config) !void {
    try config.text_config.validate();
}

/// helper, to count the total numbers of layers per kind
/// for diffusion gemma, it should be 25 sliding attention
/// and 5 full_attention spaced every 6th layer
pub fn countLayers(self: Config, kind: LayerType) u32 {
    var count: u32 = 0;

    for (self.text_config.layer_types) |layer_type| {
        if (layer_type == kind) {
            count += 1;
        }
    }

    return count;
}

pub const LayerType = enum {
    sliding_attention,
    full_attention,
};
