const zml = @import("zml");
const nn = zml.nn;

const Config = @This();

canvas_length: u32,
text_config: Text,

pub const LayerType = enum {
    sliding_attention,
    full_attention,
};

pub const Text = struct {
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    num_global_key_value_heads: u32,
    head_dim: u32,
    global_head_dim: u32,
    num_experts: u32,
    top_k_experts: u32,
    sliding_window: u32,
    layer_types: []const LayerType,
    rope_parameters: struct {
        sliding_attention: nn.RopeOpts.Scaling,
        full_attention: nn.RopeOpts.Scaling,
    },
    rms_norm_eps: f32,
    final_logit_softcapping: f32,
    bos_token_id: u32,
};

pub fn validate(self: Config) !void {
    const text = self.text_config;

    if (self.canvas_length == 0) {
        return error.InvalidConfig;
    }

    if (text.layer_types.len != text.num_hidden_layers) {
        return error.InvalidConfig;
    }

    if (text.top_k_experts == 0 or text.top_k_experts > text.num_experts) {
        return error.InvalidConfig;
    }
}

pub fn countLayers(self: Config, kind: LayerType) u32 {
    var count: u32 = 0;

    for (self.text_config.layer_types) |layer_type| {
        if (layer_type == kind) {
            count += 1;
        }
    }

    return count;
}
