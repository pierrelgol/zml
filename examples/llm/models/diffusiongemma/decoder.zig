const zml = @import("zml");
const moe = zml.moe;
const zml_io = zml.io;

const blocks = @import("blocks.zig");
const Config = @import("Config.zig");
const KvCache = @import("KvCache.zig");
const Layer = @import("Layer.zig");
const Stack = @import("Stack.zig");

pub const SelfConditioning = struct {
    pre_norm: blocks.RmsNorm,
    mlp: blocks.Mlp,
    post_norm: blocks.RmsNorm,

    pub fn init(store: zml_io.TensorStore.View, eps: f32) SelfConditioning {
        return .{
            .pre_norm = .init(store.withPrefix("pre_norm"), eps),
            .mlp = .init(store),
            .post_norm = .initWithoutScale(eps),
        };
    }

    pub fn forward(self: SelfConditioning, embeds: zml.Tensor, soft: zml.Tensor) zml.Tensor {
        const signal = self.mlp.forward(self.pre_norm.forward(soft));
        return self.post_norm.forward(embeds.add(signal));
    }
};

pub fn softEmbeddings(stack: Stack, logits: zml.Tensor) zml.Tensor {
    const weight = stack.embed_tokens.weight;
    const probabilities = logits.convert(weight.dtype()).convert(.f32).softmax(.voc).convert(weight.dtype());
    const soft = probabilities.dot(weight, .voc);
    return soft.mul(zml.Tensor.scalar(stack.embedScale(), soft.dtype()).broad(soft.shape()));
}

pub fn forward(
    stack: Stack,
    self_conditioning: SelfConditioning,
    canvas: zml.Tensor,
    soft: zml.Tensor,
    start: zml.Tensor,
    cache: KvCache,
    backend: moe.Backend,
) zml.Tensor {
    const embeds = stack.embed(canvas);
    var x = self_conditioning.forward(embeds, soft.convert(embeds.dtype()));

    for (stack.layers) |layer| {
        x = block(layer, x, start, cache, stack.sliding_window, backend);
    }

    return stack.logits(stack.norm.forward(x));
}

pub fn block(layer: Layer, x: zml.Tensor, start: zml.Tensor, cache: KvCache, window: u32, backend: moe.Backend) zml.Tensor {
    const attn = attention(layer, layer.input_layernorm.forward(x), start, cache, window);
    return layer.feedForward(x, attn, layer.decoder_scalar, backend);
}

pub fn attention(layer: Layer, hidden: zml.Tensor, start: zml.Tensor, cache: KvCache, window: u32) zml.Tensor {
    const len = hidden.dim(.s);
    const projected = layer.self_attn.project(hidden, Stack.positions(start, len));

    const cached_k, const cached_v = cache.pool(layer.kind).read(layer.cache_index, start);
    const dtype = projected.q.dtype();
    const k = zml.Tensor.concatenate(&.{ cached_k.convert(dtype), projected.k.rename(.{ .s = .k }) }, .k);
    const v = zml.Tensor.concatenate(&.{ cached_v.convert(dtype), projected.v.rename(.{ .s = .k }) }, .k);

    const visible = canvasMask(layer.kind, len, cached_k.dim(.k), start, window);

    return layer.self_attn.attend(projected.q.rename(.{ .s = .q }), k, v, visible).rename(.{ .q = .s });
}

fn canvasMask(kind: Config.LayerType, len: i64, num_slots: i64, start: zml.Tensor, window: u32) zml.Tensor {
    const cache_shape = zml.Shape.init(.{ .q = len, .k = num_slots }, .i32);
    const slot = zml.Tensor.iota(cache_shape, .k);
    const written = start.convert(.i32);
    var cached = slot.cmp(.LT, written);

    if (kind == .sliding_attention) {
        const oldest = written.addConstant(-@as(i64, window - 1));
        cached = cached.logical(.AND, slot.cmp(.GE, oldest));
    }

    const canvas_shape = zml.Shape.init(.{ .q = len, .k = len }, .bool);
    const canvas = zml.Tensor.constant(.{ .bool = true }).broad(canvas_shape);
    return zml.Tensor.concatenate(&.{ cached, canvas }, .k);
}
