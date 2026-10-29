const zml = @import("zml");
const moe = zml.moe;

const Config = @import("Config.zig");
const KvCache = @import("KvCache.zig");
const Layer = @import("Layer.zig");
const Stack = @import("Stack.zig");

pub fn forward(
    stack: Stack,
    tokens: zml.Tensor,
    start: zml.Tensor,
    cache: KvCache,
    backend: moe.Backend,
) struct { zml.Tensor, KvCache } {
    var x = stack.embed(tokens);
    var updated = cache;
    for (stack.layers) |layer| {
        x, updated = block(layer, x, start, updated, stack.sliding_window, backend);
    }
    return .{ stack.norm.forward(x), updated.reuseBuffer(cache) };
}

pub fn block(
    layer: Layer,
    x: zml.Tensor,
    start: zml.Tensor,
    cache: KvCache,
    window: u32,
    backend: moe.Backend,
) struct { zml.Tensor, KvCache } {
    const attn, const updated = attention(layer, layer.input_layernorm.forward(x), start, cache, window);
    return .{ layer.feedForward(x, attn, layer.encoder_scalar, backend), updated };
}

pub fn attention(
    layer: Layer,
    hidden: zml.Tensor,
    start: zml.Tensor,
    cache: KvCache,
    window: u32,
) struct { zml.Tensor, KvCache } {
    const len = hidden.dim(.s);
    const positions = Stack.positions(start, len);
    const projected = layer.self_attn.project(hidden, positions);

    const pool = cache.pool(layer.kind).write(layer.cache_index, projected.k, projected.v, start);
    const k, const v = pool.read(layer.cache_index, start.convert(.i32).addConstant(len));

    const visible = causalMask(layer.kind, positions.rename(.{ .s = .q }), k.dim(.k), window);
    const attn = layer.self_attn.attend(projected.q.rename(.{ .s = .q }), k, v, visible);
    return .{ attn.rename(.{ .q = .s }), cache.withPool(layer.kind, pool) };
}

fn causalMask(kind: Config.LayerType, query_positions: zml.Tensor, num_slots: i64, window: u32) zml.Tensor {
    const shape = zml.Shape.init(.{ .q = query_positions.dim(.q), .k = num_slots }, .i32);
    const query = query_positions.broad(shape);
    const key = zml.Tensor.iota(shape, .k);
    var visible = key.cmp(.LE, query);

    if (kind == .sliding_attention) {
        const recent = query.sub(key).cmp(.LT, .scalar(window, .i32));
        visible = visible.logical(.AND, recent);
    }
    return visible;
}
