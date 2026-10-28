//! Reductions. ZML keeps reduced axes with size 1 so tags survive.

const std = @import("std");

const k = @import("../kit.zig");
const Shape = k.Shape;
const DataType = k.DataType;
const Tensor = k.Tensor;
const Value = k.Value;
const Gen = k.Gen;
const Eval = k.Eval;
const Node = k.Node;
const Writer = k.Writer;
const AxisRef = k.AxisRef;

/// Iterates the elements along `ax` for each position of the reduced output.
const Lanes = struct {
    in: Shape,
    ax: usize,
    st: [Shape.MAX_RANK]usize,
    it: Value.Index,
    out_i: usize = 0,

    fn init(in: Shape, ax: usize) Lanes {
        return .{ .in = in, .ax = ax, .st = Value.strides(in), .it = .init(in.setDim(ax, 1)) };
    }

    /// Flat offset of the first element of the current lane and the stride along it.
    const Lane = struct { base: usize, stride: usize, out: usize };

    fn next(l: *Lanes) ?Lane {
        if (l.it.done) return null;
        const base = Value.offset(l.st, l.it.idx[0..l.in.rank()]);
        const res: Lane = .{ .base = base, .stride = l.st[l.ax], .out = l.out_i };
        l.out_i += 1;
        _ = l.it.next();
        return res;
    }
};

/// Sum of a lane, plus whether the float result is exact in `dt`.
fn sumBound(dt: DataType, n: usize, abs_sum: f64, all_integral: bool, in_exact: bool) f64 {
    if (in_exact and all_integral and abs_sum < std.math.ldexp(@as(f64, 1), Value.precisionBits(dt))) return 0;
    // Any summation order: n roundings bounded by eps times the running absolute sum.
    return @as(f64, @floatFromInt(n)) * Value.eps(dt) * abs_sum;
}

/// Set to true to generate Tensor.any and Tensor.all, which currently don't compile.
pub const compile_broken_apis = false;

pub const reduce = struct {
    pub const Kind = enum { sum, mean, max, min, any, all };
    pub const Attrs = struct { kind: Kind, axis: AxisRef };
    pub const weight = 9;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        // Tensor.any and Tensor.all don't compile (KnownBug.any_compile/all_compile): they
        // are only generated when `compile_broken_apis` is set.
        if (kind == .any or kind == .all) {
            if (!compile_broken_apis) return error.Reject;
            try g.avoid(if (kind == .any) .any_compile else .all_compile, true);
        }
        const class: k.gen.Class = if (kind == .any or kind == .all) .any else .numeric;
        const x = try g.operand(.{ .class = class, .min_rank = 1 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        if (sh.dim(ax) == 0) return error.Reject;
        try g.avoid(.mean_int_len, kind == .mean and Value.isInt(sh.dtype()) and sh.dim(ax) > Value.maxInt(sh.dtype()));
        const out = if (kind == .any or kind == .all) sh.setDim(ax, 1).withDtype(.bool) else sh.setDim(ax, 1);
        return .{ .op = .{ .reduce = .{ .kind = kind, .axis = .random(g, sh, ax) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = attrs.axis.resolve(x.shape);
        const n: usize = @intCast(x.shape.dim(ax));
        var v = try e.alloc(node.shape);
        var lanes: Lanes = .init(x.shape, ax);
        const dt = x.dtype();
        const float = Value.isFloat(dt);
        var err: f64 = 0;
        // Bounds per output lane: one large lane no longer loosens the others.
        const errs = try v.trackErrs(e.allocator);
        while (lanes.next()) |l| {
            if (attrs.kind == .any or attrs.kind == .all) {
                // Documented: whether any (all) element is non zero.
                var r = attrs.kind == .all;
                for (0..n) |j| {
                    const off = l.base + j * l.stride;
                    const nz = if (float) blk: {
                        const a = x.getF(off);
                        if (x.err > 0 and @abs(a) <= 4 * x.err) try e.reject();
                        break :blk a != 0;
                    } else x.getI(off) != 0;
                    r = if (attrs.kind == .all) r and nz else r or nz;
                }
                v.setI(l.out, @intFromBool(r));
                continue;
            }
            if (!float) {
                var acc: i128 = switch (attrs.kind) {
                    .sum, .mean => 0,
                    .max => Value.minInt(dt),
                    .min => Value.maxInt(dt),
                    else => unreachable,
                };
                for (0..n) |j| {
                    const a = x.getI(l.base + j * l.stride);
                    acc = switch (attrs.kind) {
                        .sum, .mean => Value.wrapInt(dt, acc + a),
                        .max => @max(acc, a),
                        .min => @min(acc, a),
                        else => unreachable,
                    };
                }
                // mean is sum().divByConst(n): an integer division for integer dtypes.
                if (attrs.kind == .mean) acc = @divTrunc(acc, @as(i128, @intCast(n)));
                v.setI(l.out, acc);
                continue;
            }
            var acc: f64 = switch (attrs.kind) {
                .sum, .mean => 0,
                .max => -std.math.inf(f64),
                .min => std.math.inf(f64),
                else => unreachable,
            };
            var abs_sum: f64 = 0;
            var integral = true;
            var in_err_sum: f64 = 0;
            var in_err_max: f64 = 0;
            for (0..n) |j| {
                const off = l.base + j * l.stride;
                const a = x.getF(off);
                abs_sum += @abs(a);
                integral = integral and a == @round(a);
                in_err_sum += x.inErr(off);
                in_err_max = @max(in_err_max, x.inErr(off));
                acc = switch (attrs.kind) {
                    .sum, .mean => acc + a,
                    .max => @max(acc, a),
                    .min => @min(acc, a),
                    else => unreachable,
                };
            }
            var lane_err = in_err_max;
            if (attrs.kind == .sum or attrs.kind == .mean) {
                lane_err = in_err_sum + sumBound(dt, n, abs_sum, integral, in_err_sum == 0);
                if (attrs.kind == .mean) {
                    acc /= @floatFromInt(n);
                    lane_err /= @floatFromInt(n);
                    lane_err += Value.eps(dt) * 2 * @abs(acc);
                }
            }
            if (!k.storeF(v, l.out, acc)) lane_err += k.roundErr(dt, @abs(acc));
            // The sign of a zero result depends on the reduction order and on whether XLA
            // folds the init value (a size-1 reduction can return -0).
            if (acc == 0) lane_err = @max(lane_err, Value.any_zero_sign);
            errs[l.out] = lane_err;
            err = @max(err, lane_err);
        }
        v.finishErrs();
        _ = &err;
        v.rounded = float;
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        return switch (attrs.axis.pass()) {
            inline else => |a| switch (attrs.kind) {
                .sum => x.sum(a),
                .mean => x.mean(a),
                .max => x.max(a),
                .min => x.min(a),
                .any => if (comptime compile_broken_apis) x.any(a) else unreachable,
                .all => if (comptime compile_broken_apis) x.all(a) else unreachable,
            },
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.{s}(", .{ ins[0], @tagName(attrs.kind) });
        try attrs.axis.print(w);
        try w.writeAll(")");
    }
};

pub const cumsum = struct {
    pub const Attrs = struct { axis: AxisRef };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .class = .numeric, .min_rank = 1 });
        try g.avoid(.xla_reduce_window_chain, k.gen.paddedWindow(g.prog.nodes.items[x]));
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        return .{ .op = .{ .cumsum = .{ .axis = .random(g, sh, ax) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = sh };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = attrs.axis.resolve(x.shape);
        const n: usize = @intCast(x.shape.dim(ax));
        var v = try e.alloc(node.shape);
        const dt = x.dtype();
        var lanes: Lanes = .init(x.shape, ax);
        var err: f64 = 0;
        const errs = try v.trackErrs(e.allocator);
        while (lanes.next()) |l| {
            var acc_i: i128 = 0;
            var acc_f: f64 = 0;
            var abs_sum: f64 = 0;
            var in_err_sum: f64 = 0;
            var integral = true;
            for (0..n) |j| {
                const off = l.base + j * l.stride;
                if (Value.isFloat(dt)) {
                    const a = x.getF(off);
                    acc_f += a;
                    abs_sum += @abs(a);
                    in_err_sum += x.inErr(off);
                    integral = integral and a == @round(a);
                    var lane_err = in_err_sum + sumBound(dt, j + 1, abs_sum, integral, in_err_sum == 0);
                    if (!k.storeF(v, off, acc_f)) lane_err += k.roundErr(dt, @abs(acc_f));
                    if (acc_f == 0) lane_err = @max(lane_err, Value.any_zero_sign);
                    errs[off] = lane_err;
                    err = @max(err, lane_err);
                } else {
                    acc_i = Value.wrapInt(dt, acc_i + x.getI(off));
                    v.setI(off, acc_i);
                }
            }
        }
        v.err = err;
        v.rounded = Value.isFloat(dt);
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return switch (attrs.axis.pass()) {
            inline else => |a| ins[0].cumulativeSum(a),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.cumulativeSum(", .{ins[0]});
        try attrs.axis.print(w);
        try w.writeAll(")");
    }
};

pub const argmax = struct {
    pub const Attrs = struct { axis: AxisRef, indices: bool };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .class = .numeric, .min_rank = 1 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        const indices = g.chance(0.6);
        const out = sh.setDim(ax, 1);
        return .{ .op = .{ .argmax = .{ .axis = .random(g, sh, ax), .indices = indices } }, .inputs = try k.dupeIds(g, &.{x}), .shape = if (indices) out.withDtype(.i32) else out };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = attrs.axis.resolve(x.shape);
        const n: usize = @intCast(x.shape.dim(ax));
        var v = try e.alloc(node.shape);
        var lanes: Lanes = .init(x.shape, ax);
        const float = Value.isFloat(x.dtype());
        while (lanes.next()) |l| {
            // Documented: the first maximum wins.
            var best: usize = 0;
            for (1..n) |j| {
                const off = l.base + j * l.stride;
                const best_off = l.base + best * l.stride;
                const gt = if (float) x.getF(off) > x.getF(best_off) else x.getI(off) > x.getI(best_off);
                if (gt) best = j;
            }
            if (float and attrs.indices and x.err > 0) {
                // The winner must be unambiguous given the input error.
                const m = x.getF(l.base + best * l.stride);
                for (0..n) |j| if (j != best and m - x.getF(l.base + j * l.stride) <= 4 * x.err) try e.reject();
            }
            if (attrs.indices) v.setI(l.out, @intCast(best)) else v.copyElem(l.out, x, l.base + best * l.stride);
        }
        if (!attrs.indices) v.err = x.err;
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const res = switch (attrs.axis.pass()) {
            inline else => |a| ins[0].argMax(a),
        };
        return if (attrs.indices) res.indices else res.values;
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.argMax(", .{ins[0]});
        try attrs.axis.print(w);
        try w.print(").{s}", .{if (attrs.indices) "indices" else "values"});
    }
};

pub const softmax = struct {
    pub const Kind = enum { softmax, log_sum_exp };
    pub const Attrs = struct { kind: Kind, axis: AxisRef };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .class = .float, .min_rank = 1 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        if (sh.dim(ax) == 0) return error.Reject;
        const out = if (kind == .softmax) sh else sh.setDim(ax, 1);
        return .{ .op = .{ .softmax = .{ .kind = kind, .axis = .random(g, sh, ax) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = attrs.axis.resolve(x.shape);
        const n: usize = @intCast(x.shape.dim(ax));
        var v = try e.alloc(node.shape);
        var lanes: Lanes = .init(x.shape, ax);
        const dt = x.dtype();
        const errs = try v.trackErrs(e.allocator);
        const nf: f64 = @floatFromInt(n);
        // Tensor.softmax computes in f32 whatever the dtype, f64 included; logSumExp in the
        // dtype. Relative rounding: exp, a lane sum and a division.
        const cdt: DataType = if (attrs.kind == .softmax and dt == .f64) .f32 else dt;
        // Results below the smallest normal of the compute dtype may be flushed to zero.
        const floor: f64 = 2 * switch (cdt) {
            .f16 => std.math.floatMin(f16),
            .f64 => std.math.floatMin(f64),
            else => std.math.floatMin(f32),
        };
        const rel = (if (cdt == .f32 or cdt == .f64) @as(f64, 16) else 4) * Value.eps(cdt) * (nf + 4);
        while (lanes.next()) |l| {
            var m: f64 = -std.math.inf(f64);
            var lane_in: f64 = 0;
            for (0..n) |j| {
                const off = l.base + j * l.stride;
                m = @max(m, x.getF(off));
                lane_in = @max(lane_in, x.inErr(off));
            }
            var s: f64 = 0;
            for (0..n) |j| s += @exp(x.getF(l.base + j * l.stride) - m);
            if (attrs.kind == .log_sum_exp) {
                const r = m + @log(s);
                _ = k.storeF(v, l.out, r);
                // 1-Lipschitz in the max norm.
                errs[l.out] = lane_in + k.approxErr(dt, @abs(r), 1) + 4 * Value.eps(.f32) * nf;
                continue;
            }
            for (0..n) |j| {
                const off = l.base + j * l.stride;
                const p = @exp(x.getF(off) - m) / s;
                _ = k.storeF(v, off, p);
                // dp_j <= p_j (|dx_j| + sum_k p_k |dx_k|) <= 2 p_j max |dx|.
                // x - max is rounded to cdt before exp: relative error |x - max| eps.
                errs[off] = p * (2 * lane_in + rel + @abs(x.getF(off) - m) * Value.eps(cdt)) + floor + k.roundErr(dt, p);
            }
        }
        v.rounded = true;
        v.finishErrs();
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return switch (attrs.axis.pass()) {
            inline else => |a| switch (attrs.kind) {
                .softmax => ins[0].softmax(a),
                .log_sum_exp => ins[0].logSumExp(a),
            },
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.{s}(", .{ ins[0], if (attrs.kind == .softmax) "softmax" else "logSumExp" });
        try attrs.axis.print(w);
        try w.writeAll(")");
    }
};
