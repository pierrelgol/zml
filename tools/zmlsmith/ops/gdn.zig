//! nn.GatedDeltaNet.forward: the gated delta rule recurrence run with `ops.while`.
//! Per head and step t < active_length (clamped to [0, s]):
//!   v_hat = (S k) a,  delta = (v - v_hat) b,  S = S a + delta k^T,  y_t = S q
//! Outputs after active_length stay zero.

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
    const s = Shape.toTag(.s);
    const h = Shape.toTag(.h);
    const kk = Shape.toTag(.k);
    const v = Shape.toTag(.v);
};

fn at(x: Value, st: [8]usize, tags: []const Shape.Tag, idx: []const usize) usize {
    var o: usize = 0;
    for (tags, idx) |t, i| o += i * st[x.shape.hasTag(t).?];
    return o;
}

fn shapeOf(g: *Gen, permute: bool, dt: DataType, dims: []const i64, tags: []const Shape.Tag) Shape {
    if (permute) return nn.sdpa.permutedShape(g, dt, dims, tags);
    var res: Shape = .init(.{}, dt);
    for (dims, tags) |d, t| res = res.appendDim(d, t);
    return res;
}

pub const gdn = struct {
    pub const Out = enum { outputs, state };
    pub const Attrs = struct { out: Out, length: i64 };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.float) orelse return null;
        const ns = g.intIn(i64, 1, 5);
        const nh = g.intIn(i64, 1, 2);
        const nk = g.intIn(i64, 1, 4);
        const nv = g.intIn(i64, 1, 4);
        const permute = !g.cfg.avoid.contains(.gdn_axis_order) and g.chance(0.5);
        const qsh = shapeOf(g, permute, dt, &.{ ns, nh, nk }, &.{ T.s, T.h, T.kk });
        const ksh = shapeOf(g, permute, dt, &.{ ns, nh, nk }, &.{ T.s, T.h, T.kk });
        const vsh = shapeOf(g, permute, dt, &.{ ns, nh, nv }, &.{ T.s, T.h, T.v });
        const gsh = shapeOf(g, permute, dt, &.{ ns, nh }, &.{ T.s, T.h });
        const ssh = shapeOf(g, permute, dt, &.{ nh, nv, nk }, &.{ T.h, T.v, T.kk });
        var ids: [6]u32 = undefined;
        ids[0] = try g.operandLike(qsh, true);
        ids[1] = try g.operandLike(ksh, true);
        ids[2] = try g.operandLike(vsh, true);
        // Gates in [0, 1], as produced by sigmoid / exp(-softplus) in models.
        ids[3] = try g.freshInputRange(gsh, 0, 1);
        ids[4] = try g.freshInputRange(gsh, 0, 1);
        ids[5] = try g.operandLike(ssh, true);
        // Out of range lengths are clamped by forward.
        const length = if (g.chance(0.2)) g.choose(i64, &.{ -1, ns + 1, 0 }) else g.intIn(i64, 1, ns);
        const out: Out = if (g.chance(0.6)) .outputs else .state;
        return .{ .op = .{ .gdn = .{ .out = out, .length = length } }, .inputs = try k.dupeIds(g, &ids), .shape = if (out == .outputs) vsh else ssh };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const q, const kx, const vx, const al, const be, const s0 = .{ ins[0], ins[1], ins[2], ins[3], ins[4], ins[5] };
        const dt = q.dtype();
        const ns: usize = @intCast(q.shape.dim(T.s));
        const nh: usize = @intCast(q.shape.dim(T.h));
        const nk: usize = @intCast(q.shape.dim(T.kk));
        const nv: usize = @intCast(vx.shape.dim(T.v));
        const len: usize = @intCast(std.math.clamp(a.length, 0, @as(i64, @intCast(ns))));
        const st = .{ Value.strides(q.shape), Value.strides(kx.shape), Value.strides(vx.shape), Value.strides(al.shape), Value.strides(be.shape), Value.strides(s0.shape) };
        var out = try e.alloc(node.shape);
        const os = Value.strides(node.shape);
        const eps = @max(Value.eps(dt), Value.eps(.f32));
        // Per element: each output carries the bound of its head at its step. Outputs past
        // active_length stay exact zeros.
        const errs = try out.trackErrs(e.allocator);
        var S: [4][4]f64 = undefined;
        for (0..nh) |h| {
            for (0..nv) |vi| for (0..nk) |ki| {
                S[vi][ki] = s0.getF(at(s0, st[5], &.{ T.h, T.v, T.kk }, &.{ h, vi, ki }));
            };
            // Running bound on the error of S entries.
            var e_s: f64 = s0.err;
            for (0..len) |t| {
                const alpha = al.getF(at(al, st[3], &.{ T.s, T.h }, &.{ t, h }));
                const beta = be.getF(at(be, st[4], &.{ T.s, T.h }, &.{ t, h }));
                var key: [4]f64 = undefined;
                var qv: [4]f64 = undefined;
                var sum_k: f64 = 0;
                var sum_q: f64 = 0;
                for (0..nk) |ki| {
                    key[ki] = kx.getF(at(kx, st[1], &.{ T.s, T.h, T.kk }, &.{ t, h, ki }));
                    qv[ki] = q.getF(at(q, st[0], &.{ T.s, T.h, T.kk }, &.{ t, h, ki }));
                    sum_k += @abs(key[ki]);
                    sum_q += @abs(qv[ki]);
                }
                var e_next: f64 = 0;
                for (0..nv) |vi| {
                    var sk: f64 = 0;
                    var mag_sk: f64 = 0;
                    for (0..nk) |ki| {
                        sk += S[vi][ki] * key[ki];
                        mag_sk += @abs(S[vi][ki] * key[ki]);
                    }
                    const val = vx.getF(at(vx, st[2], &.{ T.s, T.h, T.v }, &.{ t, h, vi }));
                    const delta = (val - sk * alpha) * beta;
                    const e_sk = e_s * sum_k + kx.err * mag_sk / @max(sum_k, 1e-30) * @as(f64, @floatFromInt(nk)) + eps * @as(f64, @floatFromInt(nk + 1)) * mag_sk;
                    const e_delta = (vx.err + e_sk + @abs(sk) * al.err + eps * (@abs(val) + @abs(sk))) * @abs(beta) + @abs(val - sk * alpha) * be.err + eps * @abs(delta);
                    for (0..nk) |ki| {
                        S[vi][ki] = S[vi][ki] * alpha + delta * key[ki];
                        const e_entry = e_s * @abs(alpha) + @abs(S[vi][ki]) * al.err + e_delta * @abs(key[ki]) + @abs(delta) * kx.err + 2 * eps * (@abs(S[vi][ki]) + @abs(delta * key[ki]));
                        e_next = @max(e_next, e_entry);
                    }
                    var y: f64 = 0;
                    var mag_y: f64 = 0;
                    for (0..nk) |ki| {
                        y += S[vi][ki] * qv[ki];
                        mag_y += @abs(S[vi][ki] * qv[ki]);
                    }
                    if (a.out == .outputs) {
                        const o = at(out, os, &.{ T.s, T.h, T.v }, &.{ t, h, vi });
                        _ = k.storeF(out, o, y);
                        const bound = e_next * sum_q + q.err * mag_y / @max(sum_q, 1e-30) * @as(f64, @floatFromInt(nk)) + eps * @as(f64, @floatFromInt(nk + 1)) * mag_y;
                        // Fused multiply-adds and reassociated dots: a few more roundings than modeled.
                        errs[o] = 4 * bound + k.roundErr(dt, @abs(y));
                        if (y == 0) errs[o] = @max(errs[o], Value.any_zero_sign);
                    }
                }
                e_s = e_next;
            }
            if (a.out == .state) {
                for (0..nv) |vi| for (0..nk) |ki| {
                    const o = at(out, os, &.{ T.h, T.v, T.kk }, &.{ h, vi, ki });
                    _ = k.storeF(out, o, S[vi][ki]);
                    errs[o] = 4 * e_s + k.roundErr(dt, @abs(S[vi][ki]));
                    if (S[vi][ki] == 0) errs[o] = @max(errs[o], Value.any_zero_sign);
                };
            }
        }
        out.rounded = true;
        out.finishErrs();
        return out;
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const len = Tensor.scalar(a.length, .i32);
        const res = zml.nn.GatedDeltaNet.forward(ins[0], ins[1], ins[2], ins[3], ins[4], .{ .s = ins[5] }, len);
        return switch (a.out) {
            .outputs => res.outputs,
            .state => res.state.s,
        };
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("zml.nn.GatedDeltaNet.forward({s}, {s}, {s}, {s}, {s}, .{{ .s = {s} }}, zml.Tensor.scalar({d}, .i32)).{s}", .{ ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], a.length, if (a.out == .outputs) "outputs" else "state.s" });
    }
};
