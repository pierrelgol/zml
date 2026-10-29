const std = @import("std");
const mem = std.mem;
const zml = @import("zml");
const nn = zml.nn;
const TensorStore = zml.io.TensorStore;

const blocks = @import("blocks.zig");
const Config = @import("Config.zig");
const Layer = @import("Layer.zig");

pub const Stack = @This();

embed_tokens: nn.TokenEmbedding,
layers: []Layer,
norm: blocks.RmsNorm,
hidden_size: u32,
sliding_window: u32,
softcap: f32,

pub fn init(
    allocator: mem.Allocator,
    store: TensorStore.View,
    encoder_store: TensorStore.View,
    text: Config.Text,
) !Stack {
    const layers = try allocator.alloc(Layer, text.num_hidden_layers);
    errdefer allocator.free(layers);

    var layers_seen: std.EnumArray(Config.LayerType, u32) = .initFill(0);
    for (layers, text.layer_types, 0..) |*layer, kind, index| {
        layer.* = try .init(
            store.withPrefix("layers").withLayer(index),
            encoder_store.withPrefix("layers").withLayer(index),
            text,
            kind,
            layers_seen.get(kind),
        );

        layers_seen.getPtr(kind).* += 1;
    }

    return .{
        .embed_tokens = .{
            .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .model, .{ .d = .model }),
        },
        .layers = layers,
        .norm = .init(store.withPrefix("norm"), text.rms_norm_eps),
        .hidden_size = text.hidden_size,
        .sliding_window = text.sliding_window,
        .softcap = text.final_logit_softcapping,
    };
}

pub fn deinit(self: Stack, allocator: mem.Allocator) void {
    allocator.free(self.layers);
}
