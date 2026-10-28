//! zml.nn: activations, norms, rope, attention masks, sdpa, embeddings, Linear and the
//! resize/upsample helpers. References follow the doc comments, and the formulas the
//! docs point to (HF rope scaling, torch normalize, ...).

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
const TagId = k.TagId;
const AxisRef = k.AxisRef;
const List = k.List;
const elementwise = @import("elementwise.zig");

const T = struct {
    const s = Shape.toTag(.s);
    const hd = Shape.toTag(.hd);
    const h = Shape.toTag(.h);
    const q = Shape.toTag(.q);
    const kk = Shape.toTag(.k);
};

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// Generic inexact float error: a few ulps of every intermediate of magnitude `mag`.
fn floatErr(dt: DataType, in_err: f64, gain: f64, mag: f64, ops_count: f64) f64 {
    return gain * in_err + ops_count * k.approxErr(dt, mag, 1) + ops_count * 4 * Value.eps(.f32) * mag;
}

// -----------------------------------------------------------------------------
// Activations: nn.elu and nn.Activation.forward.

pub const activation = struct {
    pub const Kind = enum { sigmoid, tanh, relu, leaky_relu, elu, silu, gelu, quick_gelu, elu_fn };
    pub const Attrs = struct { kind: Kind, alpha: f64 = 0 };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .class = .float });
        const alpha: f64 = switch (kind) {
            .leaky_relu, .elu, .elu_fn => g.choose(f64, &.{ 0.5, 1, 0.25 }),
            else => 0,
        };
        return .{ .op = .{ .activation = .{ .kind = kind, .alpha = alpha } }, .inputs = try k.dupeIds(g, &.{x}), .shape = g.shapeOf(x) };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const U = elementwise.unary;
        switch (a.kind) {
            .sigmoid => return U.eval(e, .{ .kind = .sigmoid }, node, ins),
            .tanh => return U.eval(e, .{ .kind = .tanh }, node, ins),
            .relu => return U.eval(e, .{ .kind = .relu }, node, ins),
            .leaky_relu => return U.eval(e, .{ .kind = .leaky_relu, .slope = a.alpha }, node, ins),
            .silu => return U.eval(e, .{ .kind = .silu }, node, ins),
            .gelu => return U.eval(e, .{ .kind = .gelu }, node, ins),
            .quick_gelu => return U.eval(e, .{ .kind = .quick_gelu }, node, ins),
            .elu, .elu_fn => {
                const x = ins[0];
                var v = try e.alloc(node.shape);
                // elu(x) = x if x >= 0 else alpha * (exp(x) - 1); the branch is a discontinuity
                // of the derivative only, but the select on x >= 0 needs a margin.
                // Per element: the positive branch is a copy, the negative one exp(x) - 1.
                const errs = try v.trackErrs(e.allocator);
                const dt = x.dtype();
                for (0..x.count()) |i| {
                    const xi = x.getF(i);
                    const ei = x.inErr(i);
                    if (ei > 0 and @abs(xi) <= 4 * ei) try e.reject();
                    if (xi >= 0) {
                        v.copyElem(i, x, i);
                        // At 0 the other branch gives +0: either sign.
                        errs[i] = if (xi == 0) @max(ei, Value.any_zero_sign) else ei;
                    } else {
                        const r = a.alpha * (@exp(xi) - 1);
                        const exact = k.storeF(v, i, r);
                        // exp is a few ulps off, and exp(x) - 1 cancels near 0 (up to alpha).
                        errs[i] = a.alpha * @exp(xi) * ei + k.approxErr(dt, a.alpha, 0) + 4 * Value.eps(dt) * a.alpha;
                        if (!exact) errs[i] += k.roundErr(dt, @abs(r));
                        if (r == 0) errs[i] = @max(errs[i], Value.any_zero_sign);
                    }
                }
                v.rounded = true;
                v.finishErrs();
                return v;
            },
        }
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const act: zml.nn.Activation = switch (a.kind) {
            .sigmoid => .sigmoid,
            .tanh => .tanh,
            .relu => .relu,
            .leaky_relu => .{ .leakyReLU = @floatCast(a.alpha) },
            .elu => .{ .elu = @floatCast(a.alpha) },
            .silu => .silu,
            .gelu => .gelu,
            .quick_gelu => .quick_gelu,
            .elu_fn => return zml.nn.elu(ins[0], @floatCast(a.alpha)),
        };
        return act.forward(ins[0]);
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (a.kind) {
            .elu_fn => return w.print("zml.nn.elu({s}, {d})", .{ ins[0], a.alpha }),
            .leaky_relu => return w.print("(zml.nn.Activation{{ .leakyReLU = {d} }}).forward({s})", .{ a.alpha, ins[0] }),
            .elu => return w.print("(zml.nn.Activation{{ .elu = {d} }}).forward({s})", .{ a.alpha, ins[0] }),
            else => try w.print("(zml.nn.Activation{{ .{s} = {{}} }}).forward({s})", .{ @tagName(a.kind), ins[0] }),
        }
    }
};

// -----------------------------------------------------------------------------
// Normalizations: LayerNorm, rmsNorm, normalizeVariance, normalizeL2.

pub const norm = struct {
    pub const Kind = enum { layer_norm, rms_norm, normalize_variance, normalize_l2 };
    pub const Attrs = struct { kind: Kind, eps: f64, axis: AxisRef = .{ .index = -1 }, has_bias: bool = false };
    pub const weight = 4;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .class = .float, .min_rank = 1 });
        const sh = g.shapeOf(x);
        if (sh.dim(-1) == 0) return error.Reject;
        var a: Attrs = .{ .kind = kind, .eps = g.choose(f64, &.{ 1e-5, 1e-6, 1e-3, 0.1 }) };
        var ids: [3]u32 = .{ x, 0, 0 };
        var n: usize = 1;
        switch (kind) {
            .layer_norm => {
                // weight and bias are vectors broadcast on the last axis.
                const wsh = Shape.init(.{sh.dim(-1)}, sh.dtype());
                ids[1] = try g.freshInput(wsh);
                n = 2;
                a.has_bias = g.chance(0.6);
                if (a.has_bias) {
                    ids[2] = try g.freshInput(wsh);
                    n = 3;
                }
            },
            .rms_norm => {
                const ax = g.rng.uintLessThan(usize, sh.rank());
                if (sh.dim(ax) == 0) return error.Reject;
                a.axis = .random(g, sh, ax);
            },
            else => {},
        }
        return .{ .op = .{ .norm = a }, .inputs = try k.dupeIds(g, ids[0..n]), .shape = sh };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const dt = x.dtype();
        const ax = if (a.kind == .rms_norm) a.axis.resolve(x.shape) else x.shape.rank() - 1;
        const n: usize = @intCast(x.shape.dim(ax));
        var v = try e.alloc(node.shape);
        const st = Value.strides(x.shape);
        var it: Value.Index = .init(x.shape.setDim(ax, 1));
        var gain: f64 = 1;
        // Per lane bounds: the lane's own scale and magnitudes.
        const errs = try v.trackErrs(e.allocator);
        // The norms compute in f32 whatever the input dtype.
        const cdt: DataType = if (a.kind == .normalize_l2 or dt == .bf16 or dt == .f16) dt else .f32;
        const w_mag = if (a.kind == .layer_norm) @max(1, ins[1].maxAbs()) else 1;
        var w_err: f64 = 0;
        for (ins[1..]) |w| w_err = @max(w_err, w.err);
        while (!it.done) {
            const base = Value.offset(st, it.idx[0..x.shape.rank()]);
            var mean: f64 = 0;
            var sq: f64 = 0;
            for (0..n) |j| {
                const xi = x.getF(base + j * st[ax]);
                mean += xi;
                sq += xi * xi;
            }
            mean /= @floatFromInt(n);
            var variance: f64 = 0;
            for (0..n) |j| {
                const d = x.getF(base + j * st[ax]) - mean;
                variance += d * d;
            }
            variance /= @floatFromInt(n);
            const nf: f64 = @floatFromInt(n);
            const scale: f64 = switch (a.kind) {
                .layer_norm, .normalize_variance => 1 / @sqrt(variance + a.eps),
                .rms_norm => 1 / @sqrt(sq / nf + a.eps),
                // Implemented as x * rsqrt(sum(x^2) + eps).
                .normalize_l2 => 1 / @sqrt(sq + a.eps),
            };
            gain = @max(gain, scale * 2);
            const lane_gain = @max(1, scale * 2);
            var lane_x: f64 = 0;
            var lane_in: f64 = 0;
            var lane_out: f64 = 0;
            for (0..n) |j| {
                const off = base + j * st[ax];
                const xi = x.getF(off);
                var r = switch (a.kind) {
                    .layer_norm, .normalize_variance => (xi - mean) * scale,
                    else => xi * scale,
                };
                if (a.kind == .layer_norm) {
                    r *= ins[1].getF(j);
                    if (a.has_bias) r += ins[2].getF(j);
                }
                _ = k.storeF(v, off, r);
                lane_x = @max(lane_x, @abs(xi));
                lane_in = @max(lane_in, x.inErr(off));
                lane_out = @max(lane_out, @abs(r));
            }
            const err_in = @max(lane_in, w_err * lane_gain);
            var lane_err = floatErr(cdt, err_in, lane_gain * @max(1, lane_x), @max(1, lane_out), nf + 4);
            // x - mean cancels: the rounding of the mean, relative to the inputs, is
            // amplified by the scale (up to 1/sqrt(eps) for a constant lane).
            if (a.kind == .layer_norm or a.kind == .normalize_variance)
                lane_err += lane_gain * w_mag * (nf + 2) * Value.eps(cdt) * lane_x;
            for (0..n) |j| errs[base + j * st[ax]] = lane_err;
            _ = it.next();
        }
        v.rounded = true;
        v.finishErrs();
        // Normalizing a near constant lane amplifies everything.
        if (gain > 1e3) try e.reject();
        return v;
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const eps: f32 = @floatCast(a.eps);
        return switch (a.kind) {
            .layer_norm => (zml.nn.LayerNorm{ .weight = ins[1], .bias = if (a.has_bias) ins[2] else null, .eps = eps }).forward(ins[0]),
            .rms_norm => switch (a.axis.pass()) {
                inline else => |ax| zml.nn.rmsNorm(ins[0], ax, eps),
            },
            .normalize_variance => zml.nn.normalizeVariance(ins[0], eps),
            .normalize_l2 => zml.nn.normalizeL2(ins[0], eps),
        };
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (a.kind) {
            .layer_norm => try w.print("(zml.nn.LayerNorm{{ .weight = {s}, .bias = {s}, .eps = {e} }}).forward({s})", .{ ins[1], if (a.has_bias) ins[2] else "null", a.eps, ins[0] }),
            .rms_norm => {
                try w.print("zml.nn.rmsNorm({s}, ", .{ins[0]});
                try a.axis.print(w);
                try w.print(", {e})", .{a.eps});
            },
            .normalize_variance => try w.print("zml.nn.normalizeVariance({s}, {e})", .{ ins[0], a.eps }),
            .normalize_l2 => try w.print("zml.nn.normalizeL2({s}, {e})", .{ ins[0], a.eps }),
        }
    }
};

// -----------------------------------------------------------------------------
// Rotary position embedding.

pub const rope = struct {
    pub const Layout = zml.nn.RopeOpts.Layout;
    pub const ScalingKind = enum { default, linear, proportional, llama3, yarn };
    pub const Attrs = struct {
        layout: Layout,
        scaling: ScalingKind,
        theta: f64,
        factor: f64,
        partial: f64,
        max_pos: u32,
        has_pos: bool,
    };
    pub const weight = 3;

    fn opts(a: Attrs) zml.nn.RopeOpts {
        const theta: f32 = @floatCast(a.theta);
        return .{
            .layout = a.layout,
            .scaling = switch (a.scaling) {
                .default => .{ .default = .{ .rope_theta = theta } },
                .linear => .{ .linear = .{ .factor = @floatCast(a.factor), .rope_theta = theta } },
                .proportional => .{ .proportional = .{ .partial_rotary_factor = @floatCast(a.partial), .rope_theta = theta } },
                .llama3 => .{ .llama3 = .{ .factor = @floatCast(a.factor), .high_freq_factor = 4, .low_freq_factor = 1, .original_max_position_embeddings = a.max_pos, .rope_theta = theta } },
                .yarn => .{ .yarn = .{ .factor = @floatCast(a.factor), .original_max_position_embeddings = a.max_pos, .rope_theta = theta, .partial_rotary_factor = @floatCast(a.partial) } },
            },
        };
    }

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.float) orelse return null;
        var a: Attrs = .{
            .layout = g.rng.enumValue(Layout),
            .scaling = g.rng.enumValue(ScalingKind),
            .theta = g.choose(f64, &.{ 10000, 500000, 100 }),
            .factor = g.choose(f64, &.{ 2, 8, 32 }),
            .partial = 1,
            .max_pos = g.choose(u32, &.{ 8, 64, 8192 }),
            .has_pos = g.chance(0.5),
        };
        if (a.scaling == .proportional or a.scaling == .yarn) a.partial = g.choose(f64, &.{ 1, 0.5, 0.25, 0.75 });
        const hd = 2 * g.intIn(i64, 1, 6);
        const o = opts(a);
        const rot = o.scaling.partialRotaryDim(hd);
        if (rot == 0 or rot % 2 != 0) return error.Reject;
        // x: .{ .s, .hd } plus optional batch/head axes, in a random order.
        var sh: Shape = .init(.{}, dt);
        sh = sh.appendDim(g.intIn(i64, 1, 5), T.s).appendDim(hd, T.hd);
        if (g.chance(0.5)) sh = sh.appendDim(g.intIn(i64, 1, 3), T.h);
        var perm: [3]u8 = .{ 0, 1, 2 };
        g.rng.shuffle(u8, perm[0..sh.rank()]);
        if (g.cfg.avoid.contains(.rope_hd_not_last)) {
            // Keep .hd (index 1 in `sh`) last.
            var j: usize = 0;
            for (perm[0..sh.rank()]) |p| if (p != 1) {
                perm[j] = p;
                j += 1;
            };
            perm[j] = 1;
        }
        var x_sh: Shape = .init(.{}, dt);
        for (perm[0..sh.rank()]) |p| x_sh = x_sh.appendDim(sh.dim(p), sh.tag(p));
        const x = try g.operandLike(x_sh, true);
        var ids: [2]u32 = .{ x, 0 };
        if (a.has_pos) {
            // Positions along .s, possibly as integers.
            const pdt = g.choose(DataType, &.{ .i32, .f32, .i64 });
            const pv = try Value.alloc(g.allocator, Shape.init(.{}, pdt).appendDim(x_sh.dim(T.s), T.s));
            for (0..pv.count()) |i| {
                const p = g.intIn(i64, 0, 40);
                if (Value.isFloat(pdt)) pv.setF(i, @floatFromInt(p)) else pv.setI(i, p);
            }
            ids[1] = try g.literalInput(pv);
        }
        return .{ .op = .{ .rope = a }, .inputs = try k.dupeIds(g, ids[0..if (a.has_pos) 2 else 1]), .shape = x_sh };
    }

    /// The inverse frequencies, as documented by RopeOpts.Scaling (HF conventions).
    fn invFreq(a: Attrs, out: []f32, half: u32) void {
        const n_f: u32 = @intCast(out.len);
        const theta: f32 = @floatCast(a.theta);
        for (out, 0..) |*f, n| {
            const denom: f32 = @floatFromInt(if (a.scaling == .proportional) half else n_f);
            f.* = @exp(-@log(theta) * @as(f32, @floatFromInt(n)) / denom);
        }
        switch (a.scaling) {
            .default, .proportional => {},
            .linear => for (out) |*f| {
                f.* /= @floatCast(a.factor);
            },
            .llama3 => {
                const m: f64 = @floatFromInt(a.max_pos);
                const f_high = 4 * (2 * std.math.pi) / m;
                const f_low = 1 * (2 * std.math.pi) / m;
                const down = 1.0 / a.factor;
                for (out) |*f| {
                    const ff: f64 = f.*;
                    if (ff > f_high) {} else if (ff < f_low) {
                        f.* = @floatCast(ff * down);
                    } else {
                        const lerp = (ff - f_low) / (f_high - f_low);
                        f.* = @floatCast(ff * (lerp + (1 - lerp) * down));
                    }
                }
            },
            .yarn => {
                const nf: f64 = @floatFromInt(n_f);
                const m: f64 = @floatFromInt(a.max_pos);
                const f_high = 32.0 * (2 * std.math.pi) / m;
                const f_low = 1.0 * (2 * std.math.pi) / m;
                const down = 1.0 / a.factor;
                const n_low = @floor(-@log(f_high) / @log(a.theta) * nf);
                const n_high = @ceil(-@log(f_low) / @log(a.theta) * nf);
                for (out, 0..) |*f, n| {
                    const ff: f64 = f.*;
                    if (ff > f_high) {} else if (ff < f_low) {
                        f.* = @floatCast(ff * down);
                    } else {
                        const lerp = (n_high - @as(f64, @floatFromInt(n))) / (n_high - n_low);
                        f.* = @floatCast(ff * (lerp + (1 - lerp) * down));
                    }
                }
            },
        }
    }

    /// For element j of the head dimension: its role and the frequency index it uses.
    const Role = union(enum) { real: usize, imag: usize, pass };

    fn role(layout: Layout, hd: usize, rot: usize, j: usize) Role {
        const half = hd / 2;
        const hr = rot / 2;
        return switch (layout) {
            .real_im_pass => if (j < hr) .{ .real = j } else if (j < rot) .{ .imag = j - hr } else .pass,
            .real_pass_im_pass => if (rot == hd)
                (if (j < hr) Role{ .real = j } else Role{ .imag = j - hr })
            else if (j < hr) .{ .real = j } else if (j < half) .pass else if (j < half + hr) .{ .imag = j - half } else .pass,
            .interleaved => if (j >= rot) .pass else if (j % 2 == 0) .{ .real = j / 2 } else .{ .imag = j / 2 },
        };
    }

    /// Index of the partner element in the head dim for a rotated element.
    fn partner(layout: Layout, hd: usize, rot: usize, r: Role) usize {
        const half = hd / 2;
        const hr = rot / 2;
        return switch (r) {
            .real => |i| switch (layout) {
                .real_im_pass => hr + i,
                .real_pass_im_pass => if (rot == hd) hr + i else half + i,
                .interleaved => 2 * i + 1,
            },
            .imag => |i| switch (layout) {
                .real_im_pass, .real_pass_im_pass => i,
                .interleaved => 2 * i,
            },
            .pass => unreachable,
        };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const dt = x.dtype();
        const sh = x.shape;
        const hd: usize = @intCast(sh.dim(T.hd));
        const o = opts(a);
        const rot: usize = o.scaling.partialRotaryDim(@intCast(hd));
        const half: u32 = @intCast(hd / 2);
        const nf = o.scaling.partialRotaryDim(half);
        var freqs: [16]f32 = undefined;
        invFreq(a, freqs[0..nf], half);
        const scaling = o.scaling.attentionScaling();
        var v = try e.alloc(node.shape);
        const st = Value.strides(sh);
        const hd_ax = sh.hasTag(T.hd).?;
        const s_ax = sh.hasTag(T.s).?;
        var it: Value.Index = .init(sh);
        var i: usize = 0;
        var mag: f64 = 0;
        while (!it.done) : (i += 1) {
            const idx = it.idx[0..sh.rank()];
            const j: usize = @intCast(idx[hd_ax]);
            const r = role(a.layout, hd, rot, j);
            const xi = x.getF(i);
            if (r == .pass) {
                v.copyElem(i, x, i);
                _ = it.next();
                continue;
            }
            const fi = switch (r) {
                .real, .imag => |f| f,
                .pass => unreachable,
            };
            if (fi >= nf) {
                // ZML's frequency count and rotary dim disagree: there is no defined result.
                try e.reject();
                _ = it.next();
                continue;
            }
            const pos: f32 = if (a.has_pos) @floatCast(ins[1].getF(@intCast(idx[s_ax]))) else @floatFromInt(idx[s_ax]);
            const angle: f32 = pos * freqs[fi];
            // cos and sin are computed in f32, scaled, then converted to the input dtype.
            const c = roundTo(dt, @cos(@as(f64, angle)) * scaling);
            const sn = roundTo(dt, @sin(@as(f64, angle)) * scaling);
            var other_idx: [8]i64 = undefined;
            @memcpy(other_idx[0..sh.rank()], idx);
            other_idx[hd_ax] = @intCast(partner(a.layout, hd, rot, r));
            const xo = x.getF(Value.offset(st, other_idx[0..sh.rank()]));
            const y = switch (r) {
                // y_real = x_real * cos - x_imag * sin, y_imag = x_real * sin + x_imag * cos
                .real => xi * c - xo * sn,
                .imag => xo * sn + xi * c,
                .pass => unreachable,
            };
            mag = @max(mag, @abs(xi) + @abs(xo));
            _ = k.storeF(v, i, y);
            _ = it.next();
        }
        const pos_err: f64 = 64 * Value.eps(.f32) * 64;
        v.err = 2 * x.err * (1 + scaling) + (k.approxErr(dt, 1, 1) * 4 + pos_err + 4 * Value.eps(dt)) * @max(mag, 1) * (1 + scaling);
        return v;
    }

    fn roundTo(dt: DataType, x: f64) f64 {
        const f: f32 = @floatCast(x);
        return switch (dt) {
            .bf16 => Value.bf16ToF32(Value.bf16Rne(f)),
            .f16 => @as(f16, @floatCast(f)),
            .f32 => f,
            else => x,
        };
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return zml.nn.rope(ins[0], if (a.has_pos) ins[1] else null, opts(a));
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("zml.nn.rope({s}, {s}, .{{ .layout = .{s}, .scaling = ", .{ ins[0], if (a.has_pos) ins[1] else "null", @tagName(a.layout) });
        const theta = a.theta;
        switch (a.scaling) {
            .default => try w.print(".{{ .default = .{{ .rope_theta = {d} }} }}", .{theta}),
            .linear => try w.print(".{{ .linear = .{{ .factor = {d}, .rope_theta = {d} }} }}", .{ a.factor, theta }),
            .proportional => try w.print(".{{ .proportional = .{{ .partial_rotary_factor = {d}, .rope_theta = {d} }} }}", .{ a.partial, theta }),
            .llama3 => try w.print(".{{ .llama3 = .{{ .factor = {d}, .high_freq_factor = 4, .low_freq_factor = 1, .original_max_position_embeddings = {d}, .rope_theta = {d} }} }}", .{ a.factor, a.max_pos, theta }),
            .yarn => try w.print(".{{ .yarn = .{{ .factor = {d}, .original_max_position_embeddings = {d}, .rope_theta = {d}, .partial_rotary_factor = {d} }} }}", .{ a.factor, a.max_pos, theta, a.partial }),
        }
        try w.writeAll(" })");
    }
};

// -----------------------------------------------------------------------------

pub const causal_mask = struct {
    pub const Attrs = struct { window: u32 };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.any) orelse return null;
        const q = g.intIn(i64, 1, 6);
        const kk = g.intIn(i64, 1, 6);
        const sh = Shape.init(.{}, dt).appendDim(q, T.q).appendDim(kk, T.kk);
        return .{ .op = .{ .causal_mask = .{ .window = if (g.chance(0.5)) g.intIn(u32, 1, 6) else 0 } }, .shape = sh };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, _: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        const ql = node.shape.dim(0);
        const kl = node.shape.dim(1);
        const dt = node.shape.dtype();
        var i: usize = 0;
        for (0..@intCast(ql)) |qi| for (0..@intCast(kl)) |ki| {
            // Documented: q_idx - window_len < k_idx <= q_idx, the window only applies when
            // one of the lengths reaches it.
            var keep = ki <= qi;
            if (a.window > 0 and (ql >= a.window or kl >= a.window)) keep = keep and qi < ki + a.window;
            if (Value.isFloat(dt)) v.setF(i, if (keep) 0 else -std.math.inf(f64)) else v.setI(i, @intFromBool(keep));
            i += 1;
        };
        v.allow_inf = Value.isFloat(dt);
        return v;
    }

    pub fn lower(a: Attrs, node: *const Node, _: []const Tensor) Tensor {
        return zml.nn.causalAttnMask(.{ .q = node.shape.dim(0), .k = node.shape.dim(1) }, node.shape.dtype(), if (a.window > 0) a.window else null);
    }

    pub fn emit(a: Attrs, node: *const Node, w: *Writer, _: []const []const u8) Writer.Error!void {
        try w.print("zml.nn.causalAttnMask(.{{ .q = {d}, .k = {d} }}, .{s}, ", .{ node.shape.dim(0), node.shape.dim(1), @tagName(node.shape.dtype()) });
        if (a.window > 0) try w.print("{d})", .{a.window}) else try w.writeAll("null)");
    }
};

// -----------------------------------------------------------------------------

pub const sdpa = struct {
    pub const Attrs = struct { has_mask: bool, has_scale: bool, has_bias: bool };
    pub const weight = 3;

    pub fn permutedShape(g: *Gen, dt: DataType, dims: []const i64, tags: []const Shape.Tag) Shape {
        var perm: [4]u8 = .{ 0, 1, 2, 3 };
        g.rng.shuffle(u8, perm[0..dims.len]);
        var res: Shape = .init(.{}, dt);
        for (perm[0..dims.len]) |p| res = res.appendDim(dims[p], tags[p]);
        return res;
    }

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.float) orelse return null;
        const hk = g.intIn(i64, 1, 2);
        const hq = hk * g.intIn(i64, 1, 2);
        const ql = g.intIn(i64, 1, 4);
        const kl = g.intIn(i64, 1, 4);
        const hd = g.intIn(i64, 1, 4);
        const qsh = permutedShape(g, dt, &.{ hq, ql, hd }, &.{ T.h, T.q, T.hd });
        const ksh = permutedShape(g, dt, &.{ hk, kl, hd }, &.{ T.h, T.kk, T.hd });
        const vsh = permutedShape(g, dt, &.{ hk, kl, hd }, &.{ T.h, T.kk, T.hd });
        var a: Attrs = .{ .has_mask = g.chance(0.5), .has_scale = g.chance(0.3), .has_bias = g.chance(0.2) };
        var ids: [6]u32 = undefined;
        ids[0] = try g.operandLike(qsh, true);
        ids[1] = try g.freshInput(ksh);
        ids[2] = try g.freshInput(vsh);
        var n: usize = 3;
        if (a.has_mask) {
            const msh = Shape.init(.{}, dt).appendDim(ql, T.q).appendDim(kl, T.kk);
            ids[n] = g.pickLike(msh, true) orelse blk: {
                // A causal mask when there is one, else any additive mask.
                if (g.chance(0.5)) break :blk try g.commit(.{ .op = .{ .causal_mask = .{ .window = 0 } }, .shape = msh });
                break :blk try g.freshInput(msh);
            };
            n += 1;
        }
        if (a.has_scale) {
            ids[n] = try g.freshInput(.init(.{}, dt));
            n += 1;
        }
        if (a.has_bias) {
            ids[n] = try g.freshInput(Shape.init(.{}, dt).appendDim(hq, T.h));
            n += 1;
        }
        _ = &a;
        return .{ .op = .{ .sdpa = a }, .inputs = try k.dupeIds(g, ids[0..n]), .shape = g.shapeOf(ids[0]) };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const q = ins[0];
        const kv = ins[1];
        const vv = ins[2];
        var next: usize = 3;
        const mask: ?Value = if (a.has_mask) blk: {
            next += 1;
            break :blk ins[next - 1];
        } else null;
        const scale: ?Value = if (a.has_scale) blk: {
            next += 1;
            break :blk ins[next - 1];
        } else null;
        const bias: ?Value = if (a.has_bias) ins[next] else null;
        const dt = q.dtype();
        const hq: usize = @intCast(q.shape.dim(T.h));
        const hk: usize = @intCast(kv.shape.dim(T.h));
        const ql: usize = @intCast(q.shape.dim(T.q));
        const kl: usize = @intCast(kv.shape.dim(T.kk));
        const hd: usize = @intCast(q.shape.dim(T.hd));
        const group = hq / hk;
        const s: f64 = if (scale) |sc| sc.getF(0) else 1 / @sqrt(@as(f64, @floatFromInt(hd)));
        var v = try e.alloc(node.shape);
        const qs = Value.strides(q.shape);
        const ks = Value.strides(kv.shape);
        const vs = Value.strides(vv.shape);
        const os = Value.strides(node.shape);
        var w: [8]f64 = undefined;
        var err_in: f64 = q.err + kv.err + vv.err;
        if (mask) |mk| err_in += mk.err;
        if (scale) |sc| err_in += sc.err;
        if (bias) |bb| err_in += bb.err;
        // The softmax runs in f32 whatever the input dtype.
        const sm_eps = @max(Value.eps(dt), Value.eps(.f32));
        // Per row bounds: the row's own score magnitude and its kv head's values.
        const errs = try v.trackErrs(e.allocator);
        for (0..hq) |h| for (0..ql) |qi| {
            // Documented: q heads are split as .{ .h = k.dim(.h), .hq }: q head h uses kv head h / hq.
            const kh = h / group;
            var mag: f64 = 1;
            var vmax: f64 = 1;
            for (0..kl) |ki| for (0..hd) |d| {
                vmax = @max(vmax, @abs(vv.getF(off3(vs, vv.shape, kh, ki, d, T.kk))));
            };
            var m: f64 = -std.math.inf(f64);
            for (0..kl) |ki| {
                var dot: f64 = 0;
                for (0..hd) |d| {
                    const qo = off3(qs, q.shape, h, qi, d, T.q);
                    const ko = off3(ks, kv.shape, kh, ki, d, T.kk);
                    dot += q.getF(qo) * (kv.getF(ko) * s);
                    mag = @max(mag, @abs(q.getF(qo)) * @abs(kv.getF(ko) * s) * @as(f64, @floatFromInt(hd)));
                }
                if (mask) |mk| dot += mk.getF(qi * kl + ki);
                w[ki] = dot;
                m = @max(m, dot);
            }
            const b: ?f64 = if (bias) |bb| bb.getF(h) else null;
            if (b) |bb| m = @max(m, bb);
            if (!std.math.isFinite(m)) {
                try e.reject();
                continue;
            }
            var sum: f64 = 0;
            for (0..kl) |ki| {
                w[ki] = @exp(w[ki] - m);
                sum += w[ki];
            }
            if (b) |bb| sum += @exp(bb - m);
            const row_err = (err_in * 4 + 16 * sm_eps * @as(f64, @floatFromInt(hd + kl))) * mag * vmax;
            for (0..hd) |d| {
                var acc: f64 = 0;
                for (0..kl) |ki| acc += w[ki] / sum * vv.getF(off3(vs, vv.shape, kh, ki, d, T.kk));
                const o = off3(os, node.shape, h, qi, d, T.q);
                _ = k.storeF(v, o, acc);
                errs[o] = row_err;
            }
        };
        v.rounded = true;
        v.finishErrs();
        return v;
    }

    fn off3(st: [8]usize, sh: Shape, h: usize, x: usize, d: usize, x_tag: Shape.Tag) usize {
        return h * st[sh.hasTag(T.h).?] + x * st[sh.hasTag(x_tag).?] + d * st[sh.hasTag(T.hd).?];
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        var next: usize = 3;
        var o: zml.nn.SdpaOpts = .{};
        if (a.has_mask) {
            o.attn_mask = ins[next];
            next += 1;
        }
        if (a.has_scale) {
            o.scale = ins[next];
            next += 1;
        }
        if (a.has_bias) o.softmax_bias = ins[next];
        return zml.nn.sdpa(ins[0], ins[1], ins[2], o);
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        var next: usize = 3;
        try w.print("zml.nn.sdpa({s}, {s}, {s}, .{{ ", .{ ins[0], ins[1], ins[2] });
        if (a.has_mask) {
            try w.print(".attn_mask = {s}, ", .{ins[next]});
            next += 1;
        }
        if (a.has_scale) {
            try w.print(".scale = {s}, ", .{ins[next]});
            next += 1;
        }
        if (a.has_bias) try w.print(".softmax_bias = {s}, ", .{ins[next]});
        try w.writeAll("})");
    }
};

// -----------------------------------------------------------------------------
// TokenEmbedding and Linear.

pub const layers = struct {
    pub const Kind = enum { embedding, linear };
    pub const Attrs = struct { kind: Kind, tag: TagId = .none, has_bias: bool = false, out_dtype: DataType = .f32 };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        switch (kind) {
            .embedding => {
                const dt = g.randDtype(.any) orelse return null;
                const voc = g.intIn(i64, 1, 6);
                const d = g.intIn(i64, 1, 4);
                var t: [2]Shape.Tag = undefined;
                g.randTags(&t);
                // TokenEmbedding retags its weight .{ .voc, .d }.
                const w = try g.freshInput(k.ir.makeShape(dt, &.{ voc, d }, if (g.chance(0.5)) &t else &.{ Shape.TagUnknown, Shape.TagUnknown }));
                var ish: Shape = .init(.{}, g.choose(DataType, &.{ .i32, .i64, .u32, .u8 }));
                // TokenEmbedding tags its weight .{ .voc, .d }: indices tagged .d would batch.
                const taken = Shape.init(.{}, .i32).appendDim(1, Shape.toTag(.d));
                for (0..g.intIn(usize, 0, 2)) |_| {
                    var tt = g.freshTag(taken);
                    while (ish.hasTag(tt) != null) tt = g.freshTag(taken);
                    ish = ish.appendDim(g.intIn(i64, 1, 4), if (g.chance(0.7)) tt else null);
                }
                const iv = try Value.alloc(g.allocator, ish);
                for (0..iv.count()) |i| iv.setI(i, g.intIn(i64, 0, voc - 1));
                const idx = try g.literalInput(iv);
                return .{ .op = .{ .layers = .{ .kind = .embedding } }, .inputs = try k.dupeIds(g, &.{ w, idx }), .shape = ish.withDtype(dt).appendDim(d, Shape.toTag(.d)) };
            },
            .linear => {
                const x = try g.operand(.{ .class = .numeric, .min_rank = 1, .max_rank = 3, .tagged = true });
                const sh = g.shapeOf(x);
                const ax = g.rng.uintLessThan(usize, sh.rank());
                const t = sh.tag(ax);
                const out_t = g.freshTag(sh);
                const o = g.intIn(i64, 1, 4);
                const wdt = g.randDtype(.numeric) orelse return null;
                // weight .{ .out, .in }: forward is x.dot(weight.convert(x.dtype()), tag).
                const w = try g.freshInput(Shape.init(.{}, wdt).appendDim(o, out_t).appendDim(sh.dim(ax), t));
                const has_bias = g.chance(0.6);
                const out_dt = g.randDtype(.float) orelse return null;
                var ids: [3]u32 = .{ x, w, 0 };
                if (has_bias) ids[2] = try g.freshInput(Shape.init(.{}, g.randDtype(.numeric) orelse wdt).appendDim(o, out_t));
                var out = sh.remove(ax).withDtype(out_dt).appendDim(o, out_t);
                _ = &out;
                return .{ .op = .{ .layers = .{ .kind = .linear, .tag = TagId.of(t), .has_bias = has_bias, .out_dtype = out_dt } }, .inputs = try k.dupeIds(g, ids[0..if (has_bias) 3 else 2]), .shape = out };
            },
        }
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        switch (a.kind) {
            .embedding => {
                const w = ins[0];
                const idx = ins[1];
                var v = try e.alloc(node.shape);
                const d: usize = @intCast(w.shape.dim(1));
                for (0..idx.count()) |i| {
                    const row: usize = @intCast(idx.getI(i));
                    for (0..d) |j| v.copyElem(i * d + j, w, row * d + j);
                }
                v.err = w.err;
                return v;
            },
            .linear => {
                // Composed from the documented building blocks.
                const x = ins[0];
                const conv = @import("elementwise.zig").convert;
                var wnode: Node = .{ .op = .{ .convert = .{ .bitcast = false } }, .shape = ins[1].shape.withDtype(x.dtype()) };
                const w = try conv.eval(e, .{ .bitcast = false }, &wnode, ins[1..2]);
                const t = a.tag.tag();
                const xs = x.shape;
                const ax = xs.hasTag(t).?;
                var dot_sh = xs.remove(ax).appendDim(w.shape.dim(0), w.shape.tag(0));
                dot_sh = dot_sh.withDtype(x.dtype());
                var dot_node: Node = .{ .op = .{ .dot = .{ .general = false } }, .shape = dot_sh };
                dot_node.op.dot.contracting.append(a.tag);
                const y0 = try @import("linalg.zig").dot.eval(e, dot_node.op.dot, &dot_node, &.{ x, w });
                var cnode: Node = .{ .op = .{ .convert = .{ .bitcast = false } }, .shape = dot_sh.withDtype(a.out_dtype) };
                var y = try conv.eval(e, .{ .bitcast = false }, &cnode, &.{y0});
                if (a.has_bias) {
                    var bnode: Node = .{ .op = .{ .convert = .{ .bitcast = false } }, .shape = ins[2].shape.withDtype(a.out_dtype) };
                    const b = try conv.eval(e, .{ .bitcast = false }, &bnode, ins[2..3]);
                    var bb = try e.alloc(y.shape);
                    const st = Value.strides(y.shape);
                    var it: Value.Index = .init(y.shape);
                    var i: usize = 0;
                    while (!it.done) : (i += 1) {
                        bb.setF(i, b.getF(@intCast(it.idx[y.shape.rank() - 1])));
                        _ = it.next();
                    }
                    _ = st;
                    bb.err = b.err;
                    var anode: Node = .{ .op = .{ .binary = .{ .kind = .add, .bcast = .same } }, .shape = y.shape };
                    y = try @import("elementwise.zig").binary.eval(e, anode.op.binary, &anode, &.{ y, bb });
                    _ = &anode;
                }
                return y;
            },
        }
    }

    pub fn lower(a: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        switch (a.kind) {
            .embedding => return (zml.nn.TokenEmbedding{ .weight = ins[0] }).forward(ins[1]),
            .linear => {
                const l: zml.nn.Linear = .{ .weight = ins[1], .bias = if (a.has_bias) ins[2] else null, .tag = a.tag.tag() };
                return l.forward(ins[0], node.shape.dtype());
            },
        }
    }

    pub fn emit(a: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (a.kind) {
            .embedding => try w.print("(zml.nn.TokenEmbedding{{ .weight = {s} }}).forward({s})", .{ ins[0], ins[1] }),
            .linear => try w.print("zml.nn.Linear.init({s}, {s}, .{s}).forward({s}, .{s})", .{ ins[1], if (a.has_bias) ins[2] else "null", a.tag.name(), ins[0], @tagName(node.shape.dtype()) }),
        }
    }
};

// -----------------------------------------------------------------------------
// Resizing: nn.upsample (nearest), resizeLinear1d / resizeBilinear, resizeCubic1d /
// resizeBicubic. The interpolation conventions are ZML's: scaled = i * len / new_len.

pub const resize = struct {
    pub const Kind = enum { nearest, upsample, linear1d, bilinear, cubic1d, bicubic };
    pub const Attrs = struct { kind: Kind, axes: List(u8, 3), lens: List(i64, 3), scales: List(f64, 3) };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        var a: Attrs = .{ .kind = kind, .axes = .{}, .lens = .{}, .scales = .{} };
        switch (kind) {
            .nearest, .upsample => {
                // Spatial axes are the ones after the two leading batch/channel axes.
                var x = try g.operand(.{ .min_rank = 3, .max_rank = 5, .class = .numeric });
                var sh = g.shapeOf(x);
                if (g.cfg.avoid.contains(.nearest_drops_tags)) {
                    // Spatial axes untagged, so the known tag loss doesn't show up.
                    var plain = sh;
                    for (2..sh.rank()) |ax| plain._tags.set(ax, Shape.TagUnknown);
                    if (!std.mem.eql(Shape.Tag, plain.tags(), sh.tags())) {
                        x = try g.freshInput(plain);
                        sh = plain;
                    }
                }
                var out = sh;
                const uniform = kind == .upsample and g.chance(0.5);
                for (2..sh.rank()) |ax| try g.avoid(.nearest_drops_tags, sh.tag(ax) != Shape.TagUnknown);
                const sf0 = g.choose(f64, &.{ 2, 0.5, 1.5, 3, 1 });
                for (2..sh.rank()) |ax| {
                    const sf = if (uniform) sf0 else g.choose(f64, &.{ 2, 0.5, 1.5, 3, 1 });
                    a.scales.append(sf);
                    const n: i64 = @intFromFloat(@floor(@as(f64, @floatFromInt(sh.dim(ax))) * sf));
                    if (n == 0) return error.Reject;
                    out = out.setDim(ax, n);
                }
                if (uniform) a.scales.len = 1;
                if (out.count() > 4 * g.cfg.max_elems) return error.Reject;
                return .{ .op = .{ .resize = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
            },
            else => {
                const two = kind == .bilinear or kind == .bicubic;
                const x = try g.operand(.{ .min_rank = 1, .max_rank = 4, .class = .float, .tagged = two, .exact = true });
                const sh = g.shapeOf(x);
                var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
                g.rng.shuffle(u8, perm[0..sh.rank()]);
                const n: usize = if (two) g.intIn(usize, 1, @min(2, sh.rank())) else 1;
                var out = sh;
                for (perm[0..n]) |ax| {
                    if (sh.dim(ax) == 0) return error.Reject;
                    if (kind == .cubic1d or kind == .bicubic) {
                        try g.avoid(.cubic_small_axis, sh.dim(ax) < 4);
                        try g.avoid(.cubic_axis_order, ax >= 2);
                    }
                    const len = g.intIn(i64, 1, 2 * sh.dim(ax) + 1);
                    const half = sh.dtype() == .f16 or sh.dtype() == .bf16;
                    try g.avoid(.resize_half_coords, half and @max(sh.dim(ax), len) > 32);
                    a.axes.append(ax);
                    a.lens.append(len);
                    out = out.setDim(ax, len);
                }
                return .{ .op = .{ .resize = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
            },
        }
    }

    fn compute(dt: DataType) DataType {
        return if (Value.isFloat(dt)) dt else .f32;
    }

    fn round(dt: DataType, x: f64) f64 {
        return switch (dt) {
            .bf16 => Value.bf16ToF32(Value.bf16Rne(@floatCast(x))),
            .f16 => @as(f16, @floatCast(x)),
            .f32 => @as(f32, @floatCast(x)),
            else => x,
        };
    }

    /// 1d linear or cubic resize along `ax`, computing in `cdt` like ZML does.
    fn resize1d(e: *Eval, x: Value, ax: usize, len: i64, cubic: bool) Eval.Error!Value {
        const d = x.shape.dim(ax);
        const cdt = compute(x.dtype());
        const out_sh = x.shape.setDim(ax, len);
        var v = try e.alloc(out_sh);
        const st = Value.strides(x.shape);
        var it: Value.Index = .init(out_sh);
        var i: usize = 0;
        const ratio = round(cdt, @as(f64, @floatFromInt(d)) * round(.f32, 1.0 / @as(f64, @floatFromInt(len))));
        while (!it.done) : (i += 1) {
            var idx: [8]i64 = undefined;
            @memcpy(idx[0..x.shape.rank()], it.idx[0..x.shape.rank()]);
            const o = idx[ax];
            const scaled = round(cdt, @as(f64, @floatFromInt(o)) * ratio);
            const fl = @floor(scaled);
            var r: f64 = 0;
            if (!cubic) {
                // Indices out of range are clamped by stablehlo.gather.
                const li = std.math.clamp(@as(i64, @intFromFloat(fl)), 0, d - 1);
                const ri = std.math.clamp(@as(i64, @intFromFloat(fl)) + 1, 0, d - 1);
                idx[ax] = li;
                const lv = x.getF(Value.offset(st, idx[0..x.shape.rank()]));
                idx[ax] = ri;
                const rv = x.getF(Value.offset(st, idx[0..x.shape.rank()]));
                r = lv * (fl + 1 - scaled) + rv * (scaled - fl);
            } else {
                const t = scaled - fl;
                const pos = [4]f64{ 1, t, t * t, t * t * t };
                const wts = [4][4]f64{ .{ 0, 1, 0, 0 }, .{ -0.5, 0, 0.5, 0 }, .{ 1, -2.5, 2, -0.5 }, .{ -0.5, 1.5, -1.5, 0.5 } };
                const start = @max(@as(i64, @intFromFloat(fl)) - 1, 0);
                // gatherSlices of 4 neighbors, start clamped so the slice fits.
                const s0 = std.math.clamp(start, 0, @max(d - 4, 0));
                for (0..4) |p| for (0..4) |n| {
                    idx[ax] = @min(s0 + @as(i64, @intCast(n)), d - 1);
                    r += pos[p] * wts[p][n] * x.getF(Value.offset(st, idx[0..x.shape.rank()]));
                };
            }
            if (Value.isFloat(x.dtype())) _ = k.storeF(v, i, r) else v.setI(i, @intFromFloat(@trunc(r)));
            _ = it.next();
        }
        v.err = 0;
        return v;
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        switch (a.kind) {
            .nearest, .upsample => {
                // Documented by its implementation: src = floor((i + 0.5) * d / n).
                return k.remap(e, x, node.shape, .{ .in = x.shape, .out = node.shape }, struct {
                    fn map(c: anytype, out: []const i64, in: []i64) bool {
                        @memcpy(in, out);
                        for (2..c.in.rank()) |ax| {
                            const d = c.in.dim(ax);
                            const n = c.out.dim(ax);
                            if (d == n) continue;
                            const ratio: f32 = @as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(n));
                            const src: f32 = @floor((@as(f32, @floatFromInt(out[ax])) + 0.5) * ratio);
                            in[ax] = @min(@as(i64, @intFromFloat(src)), d - 1);
                        }
                        return true;
                    }
                }.map, null);
            },
            else => {
                if (!Value.isFloat(x.dtype()) or x.err > 0) {
                    // Integer results are truncated from floats: too sensitive to rounding.
                    try e.reject();
                }
                const cubic = a.kind == .cubic1d or a.kind == .bicubic;
                var cur = x;
                for (a.axes.slice(), a.lens.slice()) |ax, len| cur = try resize1d(e, cur, ax, len, cubic);
                cur.shape = node.shape;
                cur.err = (x.err + 16 * Value.eps(x.dtype())) * @max(1, x.maxAbs()) * (if (cubic) @as(f64, 8) else 2) * @as(f64, @floatFromInt(a.axes.len));
                return cur;
            },
        }
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        switch (a.kind) {
            .nearest => return zml.nn.nearest(x, a.scales.slice()),
            .upsample => return zml.nn.upsample(x, .{ .mode = .nearest, .scale_factor = a.scales.slice() }),
            .linear1d => return zml.nn.resizeLinear1d(x, @intCast(a.axes.slice()[0]), @intCast(a.lens.slice()[0]), .{}),
            .cubic1d => return zml.nn.resizeCubic1d(x, @intCast(a.axes.slice()[0]), @intCast(a.lens.slice()[0]), .{}),
            .bilinear, .bicubic => {
                const sh = x.shape();
                const ax = a.axes.slice();
                const ln = a.lens.slice();
                const Body = struct {
                    x: Tensor,
                    cubic: bool,
                    pub const Result = Tensor;
                    pub fn call(b: @This(), s: anytype) Tensor {
                        return if (b.cubic) zml.nn.resizeBicubic(b.x, s, .{}) else zml.nn.resizeBilinear(b.x, s, .{});
                    }
                };
                const body: Body = .{ .x = x, .cubic = a.kind == .bicubic };
                if (ax.len == 1) return k.callWithTagStruct(TagId.of(sh.tag(ax[0])), @as(u63, @intCast(ln[0])), body);
                return k.callWithTagStruct2(TagId.of(sh.tag(ax[0])), @as(u63, @intCast(ln[0])), TagId.of(sh.tag(ax[1])), @as(u63, @intCast(ln[1])), body);
            },
        }
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (a.kind) {
            .nearest => try w.print("zml.nn.nearest({s}, &[_]f64{{ ", .{ins[0]}),
            .upsample => try w.print("zml.nn.upsample({s}, .{{ .mode = .nearest, .scale_factor = &[_]f64{{ ", .{ins[0]}),
            .linear1d, .cubic1d => return w.print("zml.nn.resize{s}1d({s}, {d}, {d}, .{{}})", .{ if (a.kind == .linear1d) "Linear" else "Cubic", ins[0], a.axes.slice()[0], a.lens.slice()[0] }),
            .bilinear, .bicubic => {
                try w.print("zml.nn.resize{s}({s}, .{{ ", .{ if (a.kind == .bilinear) "Bilinear" else "Bicubic", ins[0] });
                for (a.axes.slice(), a.lens.slice(), 0..) |ax, len, i| try w.print("{s}.{s} = @as(u63, {d})", .{ if (i > 0) ", " else "", k.tagLiteral(k.emit_shapes[0].tag(ax)), len });
                return w.writeAll(" }, .{})");
            },
        }
        for (a.scales.slice(), 0..) |s, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", s });
        try w.writeAll(if (a.kind == .upsample) " } })" else " })");
    }
};

// -----------------------------------------------------------------------------

/// nn.splitRealImg / mergeRealImg and their `Pass` variants for partial rotary dims.
pub const realimg = struct {
    pub const Kind = enum { split_real, split_imag, merge_roundtrip, pass_roundtrip };
    pub const Attrs = struct { kind: Kind, layout: zml.nn.RopeOpts.Layout, rotary: u32 = 0 };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const layout = g.rng.enumValue(zml.nn.RopeOpts.Layout);
        const dt = g.randDtype(.any) orelse return null;
        // `.hd` last: the documented layout, and KnownBug.rope_hd_not_last otherwise.
        var sh: Shape = .init(.{}, dt);
        if (g.chance(0.6)) sh = sh.appendDim(g.intIn(i64, 1, 3), Shape.toTag(.s));
        const hd = 2 * g.intIn(i64, 1, 4);
        sh = sh.appendDim(hd, T.hd);
        const x = try g.operandLike(sh, true);
        var a: Attrs = .{ .kind = kind, .layout = layout };
        var out = sh;
        switch (kind) {
            .split_real, .split_imag => out = sh.setDim(-1, @divExact(hd, 2)),
            .merge_roundtrip => {},
            .pass_roundtrip => {
                // A rotary dim, even and at most hd; the passthrough part is the rest.
                a.rotary = @intCast(2 * g.intIn(i64, 1, @divExact(hd, 2)));
            },
        }
        return .{ .op = .{ .realimg = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        switch (a.kind) {
            .merge_roundtrip, .pass_roundtrip => return x.dupe(e.allocator),
            .split_real, .split_imag => {
                const hd = x.shape.dim(-1);
                const half = @divExact(hd, 2);
                // Documented layouts: [re..., im...] or interleaved [re, im, re, im, ...].
                return k.remap(e, x, node.shape, .{ .rank = x.shape.rank(), .half = half, .layout = a.layout, .imag = a.kind == .split_imag }, struct {
                    fn map(c: anytype, out: []const i64, in: []i64) bool {
                        @memcpy(in, out);
                        const j = out[c.rank - 1];
                        in[c.rank - 1] = switch (c.layout) {
                            .interleaved => 2 * j + @intFromBool(c.imag),
                            else => j + if (c.imag) c.half else 0,
                        };
                        return true;
                    }
                }.map, null);
            },
        }
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        switch (a.kind) {
            .split_real => return zml.nn.splitRealImg(x, a.layout)[0],
            .split_imag => return zml.nn.splitRealImg(x, a.layout)[1],
            .merge_roundtrip => {
                const re, const im = zml.nn.splitRealImg(x, a.layout);
                return zml.nn.mergeRealImg(re, im, a.layout);
            },
            .pass_roundtrip => {
                const re, const im, const pass = zml.nn.splitRealImgPass(x, a.layout, a.rotary);
                return zml.nn.mergeRealImgPass(re, im, pass, a.layout);
            },
        }
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (a.kind) {
            .split_real, .split_imag => try w.print("zml.nn.splitRealImg({s}, .{s})[{d}]", .{ ins[0], @tagName(a.layout), @intFromBool(a.kind == .split_imag) }),
            .merge_roundtrip => try w.print("blk: {{ const ri = zml.nn.splitRealImg({s}, .{s}); break :blk zml.nn.mergeRealImg(ri[0], ri[1], .{s}); }}", .{ ins[0], @tagName(a.layout), @tagName(a.layout) }),
            .pass_roundtrip => try w.print("blk: {{ const re, const im, const pass = zml.nn.splitRealImgPass({s}, .{s}, {d}); break :blk zml.nn.mergeRealImgPass(re, im, pass, .{s}); }}", .{ ins[0], @tagName(a.layout), a.rotary, @tagName(a.layout) }),
        }
    }
};
