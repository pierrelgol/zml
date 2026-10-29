const std = @import("std");
const mem = std.mem;
const zml = @import("zml");
const floats = zml.floats;
const nn = zml.nn;
const zml_io = zml.io;

const blocks = @import("blocks.zig");
const Config = @import("Config.zig");
const Layer = @import("Layer.zig");

const Stack = @This();

embed_tokens: nn.TokenEmbedding,
layers: []Layer,
norm: blocks.RmsNorm,
hidden_size: u32,
sliding_window: u32,
softcap: f32,

pub fn init(
    allocator: mem.Allocator,
    store: zml_io.TensorStore.View,
    encoder_store: zml_io.TensorStore.View,
    text: Config.Text,
) !Stack {
    const layers = try allocator.alloc(Layer, text.num_hidden_layers);
    errdefer allocator.free(layers);

    var layers_seen: std.EnumArray(Config.LayerType, u32) = .initFill(0);
    for (layers, text.layer_types, 0..) |*layer, kind, index| {
        layer.* = .init(
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

pub fn embedScale(self: Stack) f32 {
    const exact = @sqrt(@as(f32, @floatFromInt(self.hidden_size)));
    return floats.BFloat16.fromF32(exact).toF32();
}

pub fn embed(self: Stack, tokens: zml.Tensor) zml.Tensor {
    const rows = self.embed_tokens.forward(tokens);
    return rows.mul(zml.Tensor.scalar(self.embedScale(), rows.dtype()).broad(rows.shape()));
}

pub fn logits(self: Stack, hidden: zml.Tensor) zml.Tensor {
    const raw = hidden.dot(self.embed_tokens.weight.convert(hidden.dtype()), .d).convert(.f32);
    return raw.scale(1.0 / self.softcap).tanh().scale(self.softcap);
}

pub fn positions(start: zml.Tensor, len: i64) zml.Tensor {
    const offsets = zml.Tensor.arange(.{ .end = len }, .i32).withTags(.{.s});
    return offsets.add(start.convert(.i32).broad(offsets.shape()));
}
