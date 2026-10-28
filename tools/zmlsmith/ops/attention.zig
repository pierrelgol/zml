//! zml.attention backends that run on CPU. `attention` with the vanilla backend builds a
//! causal mask over the keys, gathers the rows starting at `token_index`, then runs sdpa.

const std = @import("std");

const zml = @import("zml");

const k = @import("../kit.zig");
const Shape = k.Shape;
const DataType = k.DataType;
const Tensor = k.Tensor;
const Value = k.Value;
const Gen = k.Gen;
const Eval = k.Eval;
const Node = k.Node;
const Writer = k.Writer;
const nn = @import("nn.zig");

const T = struct {
    const b = Shape.toTag(.b);
    const h = Shape.toTag(.h);
    const q = Shape.toTag(.q);
    const kk = Shape.toTag(.k);
    const hd = Shape.toTag(.hd);
};

fn off(sh: Shape, st: [8]usize, b: usize, h: usize, x: usize, d: usize, x_tag: Shape.Tag) usize {
    var o = h * st[sh.hasTag(T.h).?] + x * st[sh.hasTag(x_tag).?] + d * st[sh.hasTag(T.hd).?];
    if (sh.hasTag(T.b)) |ax| o += b * st[ax];
    return o;
}

pub const attention = struct {
    pub const Attrs = struct { batched: bool };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.float) orelse return null;
        const batched = g.chance(0.4);
        const nb = g.intIn(i64, 1, 3);
        const hk = g.intIn(i64, 1, 2);
        const hq = hk * g.intIn(i64, 1, 2);
        const kl = g.intIn(i64, 1, 6);
        // The gathered mask rows must fit in the keys.
        const ql = g.intIn(i64, 1, kl);
        const hd = g.intIn(i64, 1, 4);
        const qsh, const ksh, const vsh = if (batched) .{
            nn.sdpa.permutedShape(g, dt, &.{ nb, hq, ql, hd }, &.{ T.b, T.h, T.q, T.hd }),
            nn.sdpa.permutedShape(g, dt, &.{ nb, hk, kl, hd }, &.{ T.b, T.h, T.kk, T.hd }),
            nn.sdpa.permutedShape(g, dt, &.{ nb, hk, kl, hd }, &.{ T.b, T.h, T.kk, T.hd }),
        } else .{
            nn.sdpa.permutedShape(g, dt, &.{ hq, ql, hd }, &.{ T.h, T.q, T.hd }),
            nn.sdpa.permutedShape(g, dt, &.{ hk, kl, hd }, &.{ T.h, T.kk, T.hd }),
            nn.sdpa.permutedShape(g, dt, &.{ hk, kl, hd }, &.{ T.h, T.kk, T.hd }),
        };
        const idx_dt = g.choose(DataType, &.{ .u32, .i32 });
        const ish: Shape = if (batched) Shape.init(.{}, idx_dt).appendDim(nb, T.b) else .init(.{}, idx_dt);
        var ids: [4]u32 = undefined;
        ids[0] = try g.operandLike(qsh, true);
        ids[1] = try g.freshInput(ksh);
        ids[2] = try g.freshInput(vsh);
        // Mostly in range, sometimes past the last fitting row (gather clamps it).
        ids[3] = try g.freshInputRange(ish, 0, if (g.chance(0.8)) kl - ql else kl);
        return .{ .op = .{ .attention = .{ .batched = batched } }, .inputs = try k.dupeIds(g, &ids), .shape = qsh };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        _ = a;
        const q = ins[0];
        const kv = ins[1];
        const vv = ins[2];
        const tok = ins[3];
        const dt = q.dtype();
        const nb: usize = if (q.shape.hasTag(T.b)) |ax| @intCast(q.shape.dim(ax)) else 1;
        const hq: usize = @intCast(q.shape.dim(T.h));
        const hk: usize = @intCast(kv.shape.dim(T.h));
        const ql: usize = @intCast(q.shape.dim(T.q));
        const kl: usize = @intCast(kv.shape.dim(T.kk));
        const hd: usize = @intCast(q.shape.dim(T.hd));
        const group = hq / hk;
        const s: f64 = 1 / @sqrt(@as(f64, @floatFromInt(hd)));
        var v = try e.alloc(node.shape);
        const qs = Value.strides(q.shape);
        const ks = Value.strides(kv.shape);
        const vs = Value.strides(vv.shape);
        const os = Value.strides(node.shape);
        var w: [8]f64 = undefined;
        const err_in: f64 = q.err + kv.err + vv.err;
        // The softmax runs in f32 whatever the input dtype.
        const sm_eps = @max(Value.eps(dt), Value.eps(.f32));
        // Per row bounds: the row's own score magnitude and its kv head's values.
        const errs = try v.trackErrs(e.allocator);
        for (0..nb) |b| {
            // XLA gather clamps the start so that the slice of `ql` rows fits in `kl`.
            const raw = tok.getI(if (tok.shape.rank() == 0) 0 else b);
            const start: usize = @intCast(std.math.clamp(raw, 0, @as(i128, @intCast(kl - ql))));
            for (0..hq) |h| for (0..ql) |qi| {
                const kh = h / group;
                var mag: f64 = 1;
                var vmax: f64 = 1;
                for (0..kl) |ki| for (0..hd) |d| {
                    vmax = @max(vmax, @abs(vv.getF(off(vv.shape, vs, b, kh, ki, d, T.kk))));
                };
                var m: f64 = -std.math.inf(f64);
                // Causal: query row start + qi sees keys 0..start + qi.
                const last = start + qi;
                for (0..kl) |ki| {
                    if (ki > last) continue;
                    var dot: f64 = 0;
                    for (0..hd) |d| {
                        const qv = q.getF(off(q.shape, qs, b, h, qi, d, T.q));
                        const kx = kv.getF(off(kv.shape, ks, b, kh, ki, d, T.kk)) * s;
                        dot += qv * kx;
                        mag = @max(mag, @abs(qv) * @abs(kx) * @as(f64, @floatFromInt(hd)));
                    }
                    w[ki] = dot;
                    m = @max(m, dot);
                }
                var sum: f64 = 0;
                for (0..kl) |ki| {
                    w[ki] = if (ki > last) 0 else @exp(w[ki] - m);
                    sum += w[ki];
                }
                const row_err = (err_in * 4 + 16 * sm_eps * @as(f64, @floatFromInt(hd + kl))) * mag * vmax;
                for (0..hd) |d| {
                    var acc: f64 = 0;
                    for (0..kl) |ki| acc += w[ki] / sum * vv.getF(off(vv.shape, vs, b, kh, ki, d, T.kk));
                    const o = off(node.shape, os, b, h, qi, d, T.q);
                    _ = k.storeF(v, o, acc);
                    errs[o] = row_err;
                }
            };
        }
        v.rounded = true;
        v.finishErrs();
        return v;
    }

    pub fn lower(_: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return zml.attention.attention(ins[0], ins[1], ins[2], ins[3], .{ .vanilla = {} }, .{ .vanilla = {} });
    }

    pub fn emit(_: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("zml.attention.attention({s}, {s}, {s}, {s}, .{{ .vanilla = {{}} }}, .{{ .vanilla = {{}} }})", .{ ins[0], ins[1], ins[2], ins[3] });
    }
};
