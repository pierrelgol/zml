const zml = @import("zml");
const moe = zml.moe;

const Config = @import("Config.zig");
const KvCache = @import("KvCache.zig");
const Layer = @import("Layer.zig");
const Stack = @import("Stack.zig");

pub fn forward(
    stack: Stack,
    tokens: zml.Tensor,
    backend: moe.Backend,
) struct { zml.Tensor, KvCache } {
    var x = stack.embed(tokens);

    for (stack.layers) |layer| {
        x = block(layer, x, stack.sliding_window, backend);
    }

    return stack.norm.forward(x);
}

pub fn block(layer: Layer, x: zml.Tensor, window: u32, backend: moe.Backend) struct { zml.Tensor, KvCache } {
    const attn, const updated = attention(layer, layer.input_layernorm.forward(x), window);

    return .{
        layer.feedForward(x, attn, layer.encoder_scalar, backend),
        updated,
    };
}

pub fn attention(layer: Layer, hidden: zml.Tensor, window: u32) zml.Tensor {
    const len = hidden.dim(.s);
    const positions = Stack.positions(zml.Tensor.scale(0, .i32), len);
    const projected = layer.self_attn.project(hidden, positions);

    const q = projected.q.rename(.{ .s = .q });
    const k = projected.k.rename(.{ .s = .k });
    const v = projected.v.rename(.{ .s = .k });

    const visible = causalMask(layer.kind, len, window);
    return layer.self_attn.attend(q, k, v, visible);
}

fn causalMask(kind: Config.LayerType, len: i64, window: i32) zml.Tensor {
    const shape = zml.Shape.init(.{ .q = len, .k = len }, .i32);
    const query = zml.Tensor.iota(shape, .q);
    const key = zml.Tensor.iota(shape, .k);
    var visible = key.cmp(.LE, query);

    if (kind == .sliding_attention) {
        const recent = query.sub(key).cmp(.LT, .scalar(window, .i32));
        visible = visible.logical(.AND, recent);
    }

    return visible;
}
