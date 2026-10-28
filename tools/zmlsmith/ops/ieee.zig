//! IEEE special values: NaN, +-inf and +-0 through the ops whose semantics are fully
//! specified for them (StableHLO follows IEEE 754: maximum and minimum propagate NaN,
//! comparisons with NaN are false except NE). Inputs are literals mixing special and small
//! finite values; results keep `allow_inf`, so only this family consumes them.

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

pub const ieee = struct {
    pub const Kind = enum {
        negate,
        abs,
        relu,
        sign,
        floor,
        ceil,
        round_even,
        sqrt,
        exp,
        log,
        tanh,
        is_finite,
        add,
        sub,
        mul,
        div,
        maximum,
        minimum,
        lt,
        le,
        eq,
        ne,
        select,
        clamp,
        max,
        min,
        sum,
        /// argMax returns NaN when the row has one, and an index pointing at the max (any
        /// NaN for a NaN row: the reduction order isn't specified).
        argmax_ok,
        convert,
    };
    pub const Attrs = struct { kind: Kind, axis: i8 = 0, lo: f64 = -1, hi: f64 = 1 };
    pub const weight = 2;

    fn isBinary(kind: Kind) bool {
        return switch (kind) {
            .add, .sub, .mul, .div, .maximum, .minimum, .lt, .le, .eq, .ne, .select => true,
            else => false,
        };
    }

    /// A literal argument mixing special and small finite values.
    fn specialInput(g: *Gen, sh: Shape) k.GenError!u32 {
        var v = try Value.alloc(g.allocator, sh);
        const big: f64 = if (sh.dtype() == .f16) 3 else 1e30;
        const pool = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 0, -0.0, 1, -1, 0.5, -2, 3, big, -big };
        for (0..v.count()) |i| v.setF(i, g.choose(f64, &pool));
        v.allow_inf = true;
        return g.literalInput(v);
    }

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const dt = g.choose(DataType, &.{ .f32, .f64, .f16, .bf16 });
        const rank = g.intIn(usize, 1, 2);
        var dims: [2]i64 = undefined;
        for (0..rank) |i| dims[i] = g.intIn(i64, 1, 5);
        const sh = Shape.init(dims[0..rank], dt);
        // Inputs: an existing special value of this shape, or a new one.
        const x = blk: {
            if (g.chance(0.5)) if (g.pick(.{ .dtype = dt, .min_rank = @intCast(rank), .max_rank = @intCast(rank), .allow_inf = true })) |id| {
                if (g.values.items[id].allow_inf and g.shapeOf(id).eqlDims(sh)) break :blk id;
            };
            break :blk try specialInput(g, sh);
        };
        const xs = g.shapeOf(x);
        var a: Attrs = .{ .kind = kind };
        switch (kind) {
            .max, .min, .sum, .argmax_ok => {
                if (kind == .max or kind == .min) try g.avoid(.cpu_fast_min_max, hasNan(g.valueOf(x)));
                a.axis = @intCast(g.rng.uintLessThan(usize, xs.rank()));
                // Sums of large values overflow depending on the order.
                if (kind == .sum) {
                    const xv = g.valueOf(x);
                    for (0..xv.count()) |i| if (std.math.isFinite(xv.getF(i)) and @abs(xv.getF(i)) > 1e3) return error.Reject;
                }
                const out = if (kind == .argmax_ok) xs.setDim(a.axis, 1).withDtype(.bool) else xs.setDim(a.axis, 1);
                return .{ .op = .{ .ieee = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
            },
            .clamp => {
                try g.avoid(.cpu_fast_min_max, hasNan(g.valueOf(x)));
                a.lo = g.choose(f64, &.{ -std.math.inf(f64), -1, 0 });
                a.hi = g.choose(f64, &.{ 0.5, 2, std.math.inf(f64) });
            },
            .convert => {
                const to = g.choose(DataType, &.{ .f32, .f64, .f16, .bf16 });
                return .{ .op = .{ .ieee = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = xs.withDtype(to) };
            },
            else => {},
        }
        const nan_sensitive = switch (kind) {
            .maximum, .minimum, .max, .min, .clamp => true,
            else => false,
        };
        if (nan_sensitive) try g.avoid(.cpu_fast_min_max, hasNan(g.valueOf(x)));
        if (isBinary(kind)) {
            const y = try specialInput(g, xs);
            if (nan_sensitive) try g.avoid(.cpu_fast_min_max, hasNan(g.valueOf(y)));
            const out = switch (kind) {
                .lt, .le, .eq, .ne => xs.withDtype(.bool),
                else => xs,
            };
            return .{ .op = .{ .ieee = a }, .inputs = try k.dupeIds(g, &.{ x, y }), .shape = out };
        }
        const out = if (kind == .is_finite) xs.withDtype(.bool) else xs;
        return .{ .op = .{ .ieee = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    fn printF64(w: *Writer, x: f64) Writer.Error!void {
        if (std.math.isInf(x)) return w.writeAll(if (x > 0) "std.math.inf(f64)" else "-std.math.inf(f64)");
        try w.print("@as(f64, {d})", .{x});
    }

    fn hasNan(v: Value) bool {
        for (0..v.count()) |i| if (std.math.isNan(v.getF(i))) return true;
        return false;
    }

    fn nanMax(x: f64, y: f64) f64 {
        if (std.math.isNan(x) or std.math.isNan(y)) return std.math.nan(f64);
        return @max(x, y);
    }

    fn nanMin(x: f64, y: f64) f64 {
        if (std.math.isNan(x) or std.math.isNan(y)) return std.math.nan(f64);
        return @min(x, y);
    }

    fn unaryF(kind: Kind, x: f64) f64 {
        return switch (kind) {
            .negate => -x,
            .abs => @abs(x),
            .relu => nanMax(x, 0),
            .sign => if (std.math.isNan(x)) x else if (x > 0) 1 else if (x < 0) -1 else x,
            .floor => @floor(x),
            .ceil => @ceil(x),
            .round_even => if (!std.math.isFinite(x)) x else blk: {
                const r = @round(x);
                break :blk if (@abs(x - @trunc(x)) == 0.5) 2 * @round(x / 2) else r;
            },
            .sqrt => @sqrt(x),
            .exp => @exp(x),
            .log => @log(x),
            .tanh => std.math.tanh(x),
            else => unreachable,
        };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        const dt = node.shape.dtype();
        const x = ins[0];
        switch (a.kind) {
            .is_finite => {
                for (0..v.count()) |i| v.setI(i, @intFromBool(std.math.isFinite(x.getF(i))));
                return v;
            },
            .lt, .le, .eq, .ne => {
                for (0..v.count()) |i| {
                    const p = x.getF(i);
                    const q = ins[1].getF(i);
                    v.setI(i, @intFromBool(switch (a.kind) {
                        .lt => p < q,
                        .le => p <= q,
                        .eq => p == q,
                        else => p != q,
                    }));
                }
                return v;
            },
            .argmax_ok => {
                @memset(v.bytes, 1);
                return v;
            },
            else => {},
        }
        const errs = try v.trackErrs(e.allocator);
        if (a.kind == .max or a.kind == .min or a.kind == .sum) {
            const ax: usize = @intCast(a.axis);
            const st = Value.strides(x.shape);
            var it: Value.Index = .init(node.shape);
            var o: usize = 0;
            while (!it.done) : (o += 1) {
                const base = Value.offset(st, it.idx[0..x.shape.rank()]);
                var acc: f64 = switch (a.kind) {
                    .max => -std.math.inf(f64),
                    .min => std.math.inf(f64),
                    else => 0,
                };
                var abs_sum: f64 = 0;
                var zero_sign_free = false;
                for (0..@intCast(x.shape.dim(ax))) |j| {
                    const p = x.getF(base + j * st[ax]);
                    acc = switch (a.kind) {
                        .max => nanMax(acc, p),
                        .min => nanMin(acc, p),
                        else => acc + p,
                    };
                    if (std.math.isFinite(p)) abs_sum += @abs(p);
                    if (p == 0) zero_sign_free = true;
                }
                v.setF(o, acc);
                if (a.kind == .sum and std.math.isFinite(acc)) errs[o] = (@as(f64, @floatFromInt(x.shape.dim(ax))) + 1) * Value.eps(dt) * abs_sum;
                // The init value and the order decide the sign of a zero result.
                if (acc == 0 and errs[o] == 0 and (zero_sign_free or a.kind == .sum)) errs[o] = Value.any_zero_sign;
                _ = it.next();
            }
            v.allow_inf = !v.allFinite();
            v.finishErrs();
            return v;
        }
        for (0..v.count()) |i| {
            const p = x.getF(i);
            const q = if (ins.len > 1) ins[1].getF(i) else 0;
            const r: f64 = switch (a.kind) {
                .add => p + q,
                .sub => p - q,
                .mul => p * q,
                .div => p / q,
                .maximum => nanMax(p, q),
                .minimum => nanMin(p, q),
                .select => if (p > q) p else q,
                .clamp => nanMin(nanMax(p, a.lo), a.hi),
                .convert => p,
                else => unaryF(a.kind, p),
            };
            const exact = k.storeF(v, i, r);
            var bound: f64 = 0;
            const transcendental = a.kind == .exp or a.kind == .log or a.kind == .tanh;
            // Exact operations round once (bit exact); transcendentals are a few ulps off.
            if (std.math.isFinite(v.getF(i)) and transcendental) bound = k.approxErr(dt, @abs(r), 0);
            _ = exact;
            // maximum / minimum of +0 and -0, and relu / clamp at 0, can return either sign.
            const tie = switch (a.kind) {
                .maximum, .minimum => p == 0 and q == 0,
                .relu, .clamp => r == 0,
                else => false,
            };
            if (r == 0 and bound == 0 and tie) bound = Value.any_zero_sign;
            errs[i] = bound;
        }
        v.allow_inf = !v.allFinite();
        v.finishErrs();
        return v;
    }

    pub fn lower(a: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const ax: i8 = a.axis;
        return switch (a.kind) {
            .negate => x.negate(),
            .abs => x.abs(),
            .relu => x.relu(),
            .sign => x.sign(),
            .floor => x.floor(),
            .ceil => x.ceil(),
            .round_even => x.roundNearestEven(),
            .sqrt => x.sqrt(),
            .exp => x.exp(),
            .log => x.log(),
            .tanh => x.tanh(),
            .is_finite => x.isFinite(),
            .add => x.add(ins[1]),
            .sub => x.sub(ins[1]),
            .mul => x.mul(ins[1]),
            .div => x.div(ins[1]),
            .maximum => x.maximum(ins[1]),
            .minimum => x.minimum(ins[1]),
            .lt => x.cmp(.LT, ins[1]),
            .le => x.cmp(.LE, ins[1]),
            .eq => x.cmp(.EQ, ins[1]),
            .ne => x.cmp(.NE, ins[1]),
            .select => x.cmp(.GT, ins[1]).select(x, ins[1]),
            .clamp => x.clamp(Tensor.scalar(a.lo, x.dtype()).broad(x.shape()), Tensor.scalar(a.hi, x.dtype()).broad(x.shape())),
            .max => x.max(ax),
            .min => x.min(ax),
            .sum => x.sum(ax),
            .argmax_ok => argmaxOk(x, ax, x.argMax(ax)),
            .convert => x.convert(node.shape.dtype()),
        };
    }

    /// The value at the returned index equals the returned max (both NaN counts), and the
    /// max is NaN exactly when the row has a NaN.
    fn argmaxOk(x: Tensor, ax: i8, r: Tensor.ArgMaxRes) Tensor {
        const iota = Tensor.arange(.{ .end = x.dim(ax) }, r.indices.dtype()).broadcast(x.shape().withDtype(r.indices.dtype()), &.{@as(i64, ax)});
        const hit = iota.cmp(.EQ, r.indices.broad(iota.shape()));
        // The selected value: the only element of the row where hit is true.
        const zero = Tensor.scalar(0, x.dtype()).broad(x.shape());
        const sel = hit.select(x, zero).sum(ax);
        const sel_nan = hit.logical(.AND, x.cmp(.NE, x)).convert(.i32).sum(ax).cmp(.GT, Tensor.scalar(0, .i32).broad(sel.shape().withDtype(.i32)));
        const row_nan = x.cmp(.NE, x).convert(.i32).sum(ax).cmp(.GT, Tensor.scalar(0, .i32).broad(sel.shape().withDtype(.i32)));
        const max_nan = r.values.cmp(.NE, r.values);
        const same = sel.cmp(.EQ, r.values).logical(.OR, sel_nan.logical(.AND, max_nan));
        return same.logical(.AND, max_nan.cmp(.EQ, row_nan));
    }

    pub fn emit(a: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const x = ins[0];
        switch (a.kind) {
            .negate, .abs, .relu, .sign, .floor, .ceil, .sqrt, .exp, .log, .tanh => try w.print("{s}.{s}()", .{ x, switch (a.kind) {
                .negate => "negate",
                .abs => "abs",
                .relu => "relu",
                .sign => "sign",
                .floor => "floor",
                .ceil => "ceil",
                .sqrt => "sqrt",
                .exp => "exp",
                .log => "log",
                else => "tanh",
            } }),
            .round_even => try w.print("{s}.roundNearestEven()", .{x}),
            .is_finite => try w.print("{s}.isFinite()", .{x}),
            .add, .sub, .mul, .div, .maximum, .minimum => try w.print("{s}.{t}({s})", .{ x, a.kind, ins[1] }),
            .lt, .le, .eq, .ne => try w.print("{s}.cmp(.{s}, {s})", .{ x, switch (a.kind) {
                .lt => "LT",
                .le => "LE",
                .eq => "EQ",
                else => "NE",
            }, ins[1] }),
            .select => try w.print("{s}.cmp(.GT, {s}).select({s}, {s})", .{ x, ins[1], x, ins[1] }),
            .clamp => {
                try w.print("{s}.clamp(zml.Tensor.scalar(", .{x});
                try printF64(w, a.lo);
                try w.print(", {s}.dtype()).broad({s}.shape()), zml.Tensor.scalar(", .{ x, x });
                try printF64(w, a.hi);
                try w.print(", {s}.dtype()).broad({s}.shape()))", .{ x, x });
            },
            .max, .min, .sum => try w.print("{s}.{t}({d})", .{ x, a.kind, a.axis }),
            .argmax_ok => try w.print(
                \\blk: {{
                \\                const r = {s}.argMax({d});
                \\                const iota = zml.Tensor.arange(.{{ .end = {s}.dim({d}) }}, r.indices.dtype()).broadcast({s}.shape().withDtype(r.indices.dtype()), &.{{{d}}});
                \\                const hit = iota.cmp(.EQ, r.indices.broad(iota.shape()));
                \\                const sel = hit.select({s}, zml.Tensor.scalar(0, {s}.dtype()).broad({s}.shape())).sum({d});
                \\                const zero = zml.Tensor.scalar(0, .i32).broad(sel.shape().withDtype(.i32));
                \\                const sel_nan = hit.logical(.AND, {s}.cmp(.NE, {s})).convert(.i32).sum({d}).cmp(.GT, zero);
                \\                const row_nan = {s}.cmp(.NE, {s}).convert(.i32).sum({d}).cmp(.GT, zero);
                \\                const max_nan = r.values.cmp(.NE, r.values);
                \\                break :blk sel.cmp(.EQ, r.values).logical(.OR, sel_nan.logical(.AND, max_nan)).logical(.AND, max_nan.cmp(.EQ, row_nan));
                \\            }}
            , .{ x, a.axis, x, a.axis, x, a.axis, x, x, x, a.axis, x, x, a.axis, x, x, a.axis }),
            .convert => try w.print("{s}.convert(.{s})", .{ x, @tagName(node.shape.dtype()) }),
        }
    }
};
