//! Elementwise ops: unary math, binary arithmetic with ZML's restricted broadcasting,
//! ops with a scalar constant, comparisons, select, clamp and dtype conversion.

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

fn isUnsigned(dt: DataType) bool {
    return Value.isInt(dt) and !dt.isSignedInt();
}

/// Distance from `x` to the closest integer.
fn distToInt(x: f64) f64 {
    return @abs(x - @round(x));
}

/// Distance from `x` to the closest half-integer.
fn distToHalf(x: f64) f64 {
    return @abs(x - (@floor(x) + 0.5));
}

// -----------------------------------------------------------------------------
// Unary

pub const unary = struct {
    pub const Kind = enum {
        abs,
        negate,
        sign,
        floor,
        ceil,
        round_afz,
        round_even,
        exp,
        expm1,
        log,
        sqrt,
        rsqrt,
        tanh,
        sin,
        cos,
        sigmoid,
        relu,
        leaky_relu,
        silu,
        gelu,
        quick_gelu,
        not,
        popcnt,
        clz,
        is_finite,

        fn class(kind: Kind) k.gen.Class {
            return switch (kind) {
                .abs, .sign => .signed,
                .negate, .relu => .numeric,
                .not => .int_or_bool,
                .popcnt, .clz => .int,
                else => .float,
            };
        }
    };

    pub const Attrs = struct { kind: Kind, slope: f64 = 0 };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 14;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        var x = try g.operand(.{ .class = kind.class() });
        if (kind == .log or kind == .sqrt or kind == .rsqrt) {
            // Keep the domain positive, otherwise these are almost always rejected.
            const v = g.valueOf(x);
            var min: f64 = std.math.inf(f64);
            for (0..v.count()) |i| min = @min(min, v.getF(i));
            if (!(min > 0.1)) {
                // Repair rather than replace: abs(x) + 1 keeps x's data flow.
                if (g.chance(0.6) and g.valueOf(x).err == 0) {
                    const ab = try g.commit(.{ .op = .{ .unary = .{ .kind = .abs } }, .inputs = try g.allocator.dupe(u32, &.{x}), .shape = g.shapeOf(x) });
                    x = try g.commit(.{ .op = .{ .binary_const = .{ .kind = .add, .value = 1 } }, .inputs = try g.allocator.dupe(u32, &.{ab}), .shape = g.shapeOf(ab) });
                } else x = try g.freshInputRange(g.shapeOf(x), 1, 6);
            }
        }
        const sh = g.shapeOf(x);
        const out = if (kind == .is_finite) sh.withDtype(.bool) else sh;
        const slope: f64 = if (kind == .leaky_relu) g.choose(f64, &.{ 0.25, 0.5, 0.125, 2, -0.5, -0.25 }) else 0;
        try g.avoid(.leaky_negative_slope, slope < 0);
        return .{ .op = .{ .unary = .{ .kind = kind, .slope = slope } }, .inputs = try g.allocator.dupe(u32, &.{x}), .shape = out };
    }

    fn sigmoidF(x: f64) f64 {
        return 1 / (1 + @exp(-x));
    }

    fn geluF(x: f64) f64 {
        const beta = @sqrt(2.0 / std.math.pi);
        return 0.5 * x * (1 + std.math.tanh(beta * (x + 0.044715 * x * x * x)));
    }

    /// The function and a bound on its derivative at `x`.
    fn floatFn(kind: Kind, slope: f64, x: f64) struct { f64, f64 } {
        return switch (kind) {
            .abs => .{ @abs(x), 1 },
            .negate => .{ -x, 1 },
            .relu => .{ @max(x, 0), 1 },
            .leaky_relu => .{ @max(x, 0) + slope * @min(x, 0), @max(1, @abs(slope)) },
            .exp => .{ @exp(x), @exp(x) },
            .expm1 => .{ std.math.expm1(x), @exp(x) },
            .log => .{ @log(x), 1 / x },
            .sqrt => .{ @sqrt(x), if (x > 0) 0.5 / @sqrt(x) else std.math.inf(f64) },
            .rsqrt => .{ 1 / @sqrt(x), 0.5 / (x * @sqrt(x)) },
            .tanh => .{ std.math.tanh(x), 1 },
            .sin => .{ @sin(x), 1 },
            .cos => .{ @cos(x), 1 },
            .sigmoid => .{ sigmoidF(x), 0.25 },
            .silu => .{ x * sigmoidF(x), 1.1 },
            .gelu => .{ geluF(x), 1.2 },
            .quick_gelu => .{ x * sigmoidF(1.702 * x), 1.2 },
            else => unreachable,
        };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        var v = try e.alloc(node.shape);
        const n = x.count();
        const dt = x.dtype();
        const kind = attrs.kind;

        if (kind == .is_finite) {
            for (0..n) |i| v.setI(i, @intFromBool(std.math.isFinite(x.getF(i))));
            return v;
        }

        if (!Value.isFloat(dt)) {
            const bits = Value.bitWidth(dt);
            for (0..n) |i| {
                const a = x.getI(i);
                const r: i128 = switch (kind) {
                    .abs => if (a < 0) -a else a,
                    .negate => -a,
                    .sign => if (a > 0) 1 else if (a < 0) -1 else 0,
                    .relu => @max(a, 0),
                    .not => if (dt == .bool) @intFromBool(a == 0) else ~a,
                    .popcnt, .clz => blk: {
                        const u: u64 = @truncate(@as(u128, @bitCast(a)));
                        const masked = if (bits == 64) u else u & ((@as(u64, 1) << @intCast(bits)) - 1);
                        if (kind == .popcnt) break :blk @popCount(masked);
                        break :blk @as(i128, @clz(masked)) - (64 - @as(i128, bits));
                    },
                    else => unreachable,
                };
                v.setI(i, r);
            }
            return v;
        }

        const err_in = x.err;
        switch (kind) {
            .floor, .ceil, .round_afz, .round_even, .sign => {
                // Discontinuous: only well defined if the input error can't cross a step.
                for (0..n) |i| {
                    const a = x.getF(i);
                    if (err_in > 0) {
                        const d = switch (kind) {
                            .floor, .ceil => distToInt(a),
                            .sign => @abs(a),
                            else => distToHalf(a),
                        };
                        if (d <= 4 * err_in) try e.reject();
                    }
                    const r: f64 = switch (kind) {
                        .floor => @floor(a),
                        .ceil => @ceil(a),
                        .round_afz => @round(a),
                        .round_even => roundEven(a),
                        .sign => if (a > 0) 1 else if (a < 0) -1 else a,
                        else => unreachable,
                    };
                    v.setF(i, r);
                }
                // Exact results, sign of zero included (floor(-0.0) is -0.0).
                if (err_in == 0 and x.errs != null) _ = try v.trackErrs(e.allocator);
                return v;
            },
            else => {},
        }

        // Per element bounds. abs, negate, relu, leaky_relu and sqrt are correctly rounded:
        // exact inputs give bit exact results. Transcendentals are a few ulps off.
        const transcendental = switch (kind) {
            .abs, .negate, .relu, .leaky_relu, .sqrt => false,
            else => true,
        };
        const errs = try v.trackErrs(e.allocator);
        for (0..n) |i| {
            const a = x.getF(i);
            const ea = x.inErr(i);
            switch (kind) {
                .log, .rsqrt => if (a <= 1e-2 or a <= 4 * ea) try e.reject(),
                .sqrt => if (a < 0 or (ea > 0 and a <= 4 * ea)) try e.reject(),
                .sin, .cos => if (@abs(a) > 8) try e.reject(),
                else => {},
            }
            const r, const d = floatFn(kind, attrs.slope, a);
            const exact = k.storeF(v, i, r);
            v.rounded = v.rounded or !exact;
            var bound: f64 = 0;
            if (ea > 0) {
                // The derivative over the input interval, then the final rounding.
                _, const d_lo = floatFn(kind, attrs.slope, a - ea);
                _, const d_hi = floatFn(kind, attrs.slope, a + ea);
                bound = @max(@abs(d), @max(@abs(d_lo), @abs(d_hi))) * ea * 1.01 + Value.ulp(dt, r);
            }
            if (transcendental) {
                const floor_: f64 = switch (kind) {
                    // XLA's log is accurate in absolute terms near 1.
                    .sin, .cos, .tanh, .sigmoid, .expm1, .log => 1,
                    .silu, .gelu, .quick_gelu => @max(1, @abs(a)),
                    else => 0,
                };
                bound += k.approxErr(dt, @abs(r), floor_);
            }
            // The sign of maximum(-0, +0) isn't specified, and producers that don't track
            // bounds don't track the sign of zero either.
            if (bound == 0 and r == 0 and (kind == .relu or kind == .leaky_relu or x.errs == null)) bound = Value.any_zero_sign;
            errs[i] = bound;
        }
        v.finishErrs();
        return v;
    }

    fn roundEven(a: f64) f64 {
        const r = @round(a);
        if (@abs(a - @trunc(a)) == 0.5) return 2 * @round(a / 2);
        return r;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        return switch (attrs.kind) {
            .abs => x.abs(),
            .negate => if (k.ops.mutation == .drop_neg_zero and x.dtype().isFloat()) blk: {
                const r = x.negate();
                break :blk r.cmp(.EQ, Tensor.scalar(0, x.dtype()).broad(x.shape())).select(r.abs(), r);
            } else x.negate(),
            .sign => x.sign(),
            .floor => x.floor(),
            .ceil => x.ceil(),
            .round_afz => x.roundNearestAfz(),
            .round_even => x.roundNearestEven(),
            .exp => x.exp(),
            .expm1 => x.exponentialMinusOne(),
            .log => x.log(),
            .sqrt => if (k.ops.mutation == .sqrt_ulp and x.dtype() == .f32) x.sqrt().bitCast(.u32).add(Tensor.scalar(1, .u32).broad(x.shape().withDtype(.u32))).bitCast(.f32) else x.sqrt(),
            .rsqrt => x.rsqrt(),
            .tanh => x.tanh(),
            .sin => x.sin(),
            .cos => x.cos(),
            .sigmoid => x.sigmoid(),
            .relu => x.relu(),
            .leaky_relu => x.leakyReLU(@floatCast(attrs.slope)),
            .silu => x.silu(),
            .gelu => x.gelu(),
            .quick_gelu => x.quickGelu(),
            .not => x.not(),
            .popcnt => x.popcnt(),
            .clz => x.countLeadingZeros(),
            .is_finite => x.isFinite(),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const name = switch (attrs.kind) {
            .abs => "abs",
            .negate => "negate",
            .sign => "sign",
            .floor => "floor",
            .ceil => "ceil",
            .round_afz => "roundNearestAfz",
            .round_even => "roundNearestEven",
            .exp => "exp",
            .expm1 => "exponentialMinusOne",
            .log => "log",
            .sqrt => "sqrt",
            .rsqrt => "rsqrt",
            .tanh => "tanh",
            .sin => "sin",
            .cos => "cos",
            .sigmoid => "sigmoid",
            .relu => "relu",
            .leaky_relu => return w.print("{s}.leakyReLU({d})", .{ ins[0], attrs.slope }),
            .silu => "silu",
            .gelu => "gelu",
            .quick_gelu => "quickGelu",
            .not => "not",
            .popcnt => "popcnt",
            .clz => "countLeadingZeros",
            .is_finite => "isFinite",
        };
        try w.print("{s}.{s}()", .{ ins[0], name });
    }
};

// -----------------------------------------------------------------------------
// Binary

/// How the second operand relates to the first one. ZML only broadcasts scalars, and
/// size-1 axes when both operands carry exactly the same tags.
pub const Broadcast = enum { same, lhs_scalar, rhs_scalar, rhs_ones };

/// Picks the operands of a binary op according to a random broadcast mode.
fn binaryOperands(g: *Gen, class: k.gen.Class) k.GenError!struct { u32, u32, Broadcast, Shape } {
    const mode = g.rng.enumValue(Broadcast);
    switch (mode) {
        .same => {
            const a = try g.operand(.{ .class = class });
            const sh = g.shapeOf(a);
            // Same dims, possibly different tags: the result takes the lhs tags.
            const b = if (g.chance(0.7)) try g.operandLike(sh, true) else blk: {
                var other = sh;
                var tags: [Shape.MAX_RANK]Shape.Tag = undefined;
                g.randTags(tags[0..sh.rank()]);
                for (0..sh.rank()) |ax| other._tags.set(ax, tags[ax]);
                break :blk try g.operandLike(other, false);
            };
            return .{ a, b, mode, sh };
        },
        .rhs_scalar, .lhs_scalar => {
            const a = try g.operand(.{ .class = class, .min_rank = 1 });
            const sh = g.shapeOf(a);
            const s = try g.operandLike(Shape.init(.{}, sh.dtype()), false);
            return if (mode == .rhs_scalar) .{ a, s, mode, sh } else .{ s, a, mode, sh };
        },
        .rhs_ones => {
            const a = try g.operand(.{ .class = class, .min_rank = 1 });
            const sh = g.shapeOf(a);
            var other = sh;
            var changed = false;
            for (0..sh.rank()) |ax| {
                if (sh.dim(ax) != 1 and g.chance(0.5)) {
                    other = other.setDim(ax, 1);
                    changed = true;
                }
            }
            if (!changed) return error.Reject;
            const b = try g.operandLike(other, true);
            return .{ a, b, mode, sh };
        },
    }
}

/// Index of the element of `v` (of shape `vs`) read when computing element `idx` of the
/// broadcast result: scalars and size-1 axes are repeated.
fn bcastOffset(v: Value, idx: []const i64) usize {
    if (v.shape.rank() == 0) return 0;
    const st = Value.strides(v.shape);
    var off: usize = 0;
    for (idx, 0..) |x, ax| {
        const d = v.shape.dim(ax);
        off += @as(usize, @intCast(if (d == 1) 0 else x)) * st[ax];
    }
    return off;
}

pub const binary = struct {
    pub const Kind = enum {
        add,
        sub,
        mul,
        div,
        maximum,
        minimum,
        remainder,
        pow,
        @"and",
        @"or",
        xor,
        shl,
        shr_arithmetic,
        shr_logical,

        fn class(kind: Kind) k.gen.Class {
            return switch (kind) {
                .@"and", .@"or", .xor => .int_or_bool,
                .shl, .shr_logical => .int,
                .shr_arithmetic => .signed_int,
                .pow => .float,
                else => .numeric,
            };
        }
    };

    pub const Attrs = struct { kind: Kind, bcast: Broadcast };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 16;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        var a, var b, const mode, const sh = try binaryOperands(g, kind.class());
        if (kind == .pow) {
            // A positive base and a small exponent.
            a = try g.freshInputRange(g.shapeOf(a), 1, 4);
            b = try g.freshInputRange(g.shapeOf(b), -2, 2);
        }
        try g.avoid(.xla_bool_mul_sign, kind == .mul and (k.gen.fromBool(&g.prog, a) or k.gen.fromBool(&g.prog, b)));
        if (kind == .maximum or kind == .minimum) {
            const nodes = g.prog.nodes.items;
            for ([2][2]u32{ .{ a, b }, .{ b, a } }) |p| {
                if (nodes[p[0]].op == .clamp) for (nodes[p[0]].inputs.slice()[1..]) |bound| try g.avoid(.xla_clamp_simplify, bound == p[1]);
            }
        }
        // Integer divisors and shift amounts in their domain, instead of rejecting most
        // tensors with one zero divisor or one oversized shift.
        const bdt = g.shapeOf(b).dtype();
        if (Value.isInt(bdt) and g.chance(0.6) and g.scope == 0) switch (kind) {
            .div, .remainder => b = try g.freshInputRange(g.shapeOf(b), 1, 7),
            .shl, .shr_arithmetic, .shr_logical => b = try g.freshInputRange(g.shapeOf(b), 0, @as(i64, Value.bitWidth(bdt)) - 1),
            else => {},
        };
        return .{ .op = .{ .binary = .{ .kind = kind, .bcast = mode } }, .inputs = try g.allocator.dupe(u32, &.{ a, b }), .shape = sh };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const a = ins[0];
        const b = ins[1];
        var v = try e.alloc(node.shape);
        const dt = node.shape.dtype();
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;

        if (!Value.isFloat(dt)) {
            const bits = Value.bitWidth(dt);
            while (!it.done) : (i += 1) {
                const x = a.getI(bcastOffset(a, it.idx[0..node.shape.rank()]));
                const y = b.getI(bcastOffset(b, it.idx[0..node.shape.rank()]));
                const r: i128 = switch (attrs.kind) {
                    .add => x + y,
                    .sub => x - y,
                    .mul => x *% y,
                    .div, .remainder => blk: {
                        if (y == 0 or (dt.isSignedInt() and y == -1 and x == Value.minInt(dt))) {
                            try e.reject();
                            break :blk 0;
                        }
                        break :blk if (attrs.kind == .div) @divTrunc(x, y) else @rem(x, y);
                    },
                    .maximum => @max(x, y),
                    .minimum => @min(x, y),
                    .@"and" => x & y,
                    .@"or" => x | y,
                    .xor => x ^ y,
                    .shl, .shr_arithmetic, .shr_logical => blk: {
                        if (y < 0 or y >= bits) {
                            try e.reject();
                            break :blk 0;
                        }
                        const s: u7 = @intCast(y);
                        const ux: u128 = @as(u128, @bitCast(x)) & ((@as(u128, 1) << bits) - 1);
                        break :blk switch (attrs.kind) {
                            .shl => @bitCast(ux << s),
                            .shr_logical => @intCast(ux >> s),
                            else => x >> s,
                        };
                    },
                    .pow => unreachable,
                };
                v.setI(i, r);
                _ = it.next();
            }
            return v;
        }

        // Per element bounds: +, -, *, / and fmod are correctly rounded, so exact operands
        // give bit exact results.
        const errs = try v.trackErrs(e.allocator);
        // Divisors XLA may turn into a broadcast, then into a multiply by the reciprocal:
        // broadcast scalars, constants, and any divisor uniform along an axis (XLA sinks
        // broadcasts through elementwise ops).
        const recip = attrs.kind == .div and (attrs.bcast == .rhs_scalar or attrs.bcast == .rhs_ones or constDivisor(e, node) or (fromBroadcast(e, node, 1, 3) and uniformAxis(b)));
        while (!it.done) : (i += 1) {
            const oa = bcastOffset(a, it.idx[0..node.shape.rank()]);
            const ob = bcastOffset(b, it.idx[0..node.shape.rank()]);
            const x = a.getF(oa);
            const y = b.getF(ob);
            const ea = a.inErr(oa);
            const eb = b.inErr(ob);
            var r: f64 = undefined;
            var local_err: f64 = 0;
            var approx = false;
            switch (attrs.kind) {
                .add, .sub => {
                    r = if (attrs.kind == .add) x + y else x - y;
                    local_err = ea + eb;
                    // A rounded product operand may be fused into an fma, unrounded.
                    if (a.contractible and (dt == .f32 or dt == .f64)) local_err += 0.5 * Value.ulp(dt, x);
                    if (b.contractible and (dt == .f32 or dt == .f64)) local_err += 0.5 * Value.ulp(dt, y);
                },
                .mul => {
                    r = x * y;
                    local_err = @abs(x) * eb + @abs(y) * ea + ea * eb;
                },
                .div => {
                    if (@abs(y) < 0.05 or @abs(y) <= 4 * eb) try e.reject();
                    r = x / y;
                    local_err = (ea + @abs(r) * eb) / @max(@abs(y) - eb, 1e-300);
                },
                .maximum, .minimum => {
                    r = if (attrs.kind == .maximum) @max(x, y) else @min(x, y);
                    local_err = @max(ea, eb);
                },
                .remainder => {
                    if (@abs(y) < 0.05 or @abs(y) <= 4 * eb) try e.reject();
                    // fmod: truncated division, the result has the sign of the dividend.
                    r = @rem(x, y);
                    const q = @trunc(x / y);
                    if ((ea > 0 or eb > 0) and distToInt(x / y) * @abs(y) <= 4 * (ea + @abs(q) * eb)) try e.reject();
                    if ((ea > 0 or eb > 0) and @abs(r) <= 4 * (ea + @abs(q) * eb)) try e.reject();
                    local_err = ea + @abs(q) * eb;
                },
                .pow => {
                    if (x <= 0.1 or @abs(y) > 4 or x <= 4 * ea) try e.reject();
                    r = std.math.pow(f64, x, y);
                    local_err = @abs(y * std.math.pow(f64, x, y - 1)) * ea + @abs(@log(x) * r) * eb;
                    approx = true;
                },
                else => unreachable,
            }
            const exact = k.storeF(v, i, r);
            v.rounded = v.rounded or !exact;
            // f64 products are rounded by the f64 multiply itself, which storeF can't see:
            // check with an error free product.
            const rounded_mul = !exact or (dt == .f64 and @mulAdd(f64, x, y, -r) != 0);
            if (attrs.kind == .mul and rounded_mul) v.contractible = true;
            var bound: f64 = if (local_err > 0) local_err + Value.ulp(dt, v.getF(i)) else 0;
            if (approx) bound += k.approxErr(dt, @abs(r), 0);
            // XLA rewrites x / broadcast(y), and x / C for a constant C, to x * (1 / y): two
            // roundings.
            if (attrs.kind == .div and recip and @abs(std.math.frexp(y).significand) != 0.5) bound += 2 * Value.ulp(dt, r);
            // The sign of maximum(-0, +0) isn't specified.
            // Also when an operand's own sign of zero isn't known (any nonzero bound).
            if (bound == 0 and r == 0 and ((x == 0 and y == 0 and (attrs.kind == .maximum or attrs.kind == .minimum)) or a.errs == null or b.errs == null or ea > 0 or eb > 0)) bound = Value.any_zero_sign;
            errs[i] = bound;
            _ = it.next();
        }
        v.finishErrs();
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const a = ins[0];
        const b = ins[1];
        return switch (attrs.kind) {
            .add => a.add(b),
            .sub => a.sub(b),
            .mul => a.mul(b),
            .div => a.div(b),
            .maximum => a.maximum(b),
            .minimum => a.minimum(b),
            .remainder => a.remainder(b),
            .pow => a.pow(b),
            .@"and" => a.logical(.AND, b),
            .@"or" => a.logical(.OR, b),
            .xor => a.logical(.XOR, b),
            .shl => a.shiftLeft(b),
            .shr_arithmetic => a.shiftRightArithmetic(b),
            .shr_logical => a.shiftRightLogical(b),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (attrs.kind) {
            .@"and" => return w.print("{s}.logical(.AND, {s})", .{ ins[0], ins[1] }),
            .@"or" => return w.print("{s}.logical(.OR, {s})", .{ ins[0], ins[1] }),
            .xor => return w.print("{s}.logical(.XOR, {s})", .{ ins[0], ins[1] }),
            else => {},
        }
        const name = switch (attrs.kind) {
            .add => "add",
            .sub => "sub",
            .mul => "mul",
            .div => "div",
            .maximum => "maximum",
            .minimum => "minimum",
            .remainder => "remainder",
            .pow => "pow",
            .shl => "shiftLeft",
            .shr_arithmetic => "shiftRightArithmetic",
            .shr_logical => "shiftRightLogical",
            else => unreachable,
        };
        try w.print("{s}.{s}({s})", .{ ins[0], name, ins[1] });
    }
};

// -----------------------------------------------------------------------------
// Ops with a scalar constant: addConstant, subConstant, scale, divByConst, ...

/// Whether input `slot` of `node` comes from a broadcast through at most `depth`
/// elementwise ops, which XLA can move past them.
fn fromBroadcast(e: *const Eval, node: *const Node, slot: usize, depth: u8) bool {
    const prog = e.prog orelse return false;
    if (slot >= node.inputs.len) return false;
    const n = &prog.nodes.items[node.inputs.slice()[slot]];
    switch (n.op) {
        .broadcast => return true,
        .binary => |a| if (a.bcast != .same) return true,
        .unary, .binary_const, .convert => {},
        else => return false,
    }
    if (depth == 0) return false;
    for (0..n.inputs.len) |i| if (fromBroadcast(e, n, i, depth - 1)) return true;
    return false;
}

/// Whether node `id` is a constant add or subtract, possibly behind layout ops that XLA
/// looks through (KnownBug.xla_const_reassoc).
fn constAddBefore(g: *Gen, id: u32) bool {
    var cur = id;
    for (0..6) |_| {
        const n = g.prog.nodes.items[cur];
        switch (n.op) {
            .binary_const => |a| return a.kind == .add or a.kind == .sub,
            .axes, .reshape, .transpose, .broadcast, .convert => if (n.inputs.len > 0) {
                cur = n.inputs.slice()[0];
            } else return false,
            else => return false,
        }
    }
    return false;
}

/// Whether `v` holds the same values along some axis longer than 1.
fn uniformAxis(v: Value) bool {
    const st = Value.strides(v.shape);
    axes: for (0..v.shape.rank()) |ax| {
        if (v.shape.dim(ax) < 2) continue;
        var it: Value.Index = .init(v.shape);
        var i: usize = 0;
        while (!it.done) : (i += 1) {
            if (it.idx[ax] > 0 and v.getF(i) != v.getF(i - @as(usize, @intCast(it.idx[ax])) * st[ax])) continue :axes;
            _ = it.next();
        }
        return true;
    }
    return false;
}

/// Whether the divisor of a binary node is a compile-time constant.
fn constDivisor(e: *const Eval, node: *const Node) bool {
    const prog = e.prog orelse return false;
    if (node.inputs.len < 2) return false;
    return prog.nodes.items[node.inputs.slice()[1]].op == .constant;
}

pub const binary_const = struct {
    pub const Kind = enum { add, sub, scale, div, remainder, pow };
    pub const Attrs = struct { kind: Kind, value: f64 };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 6;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const class: k.gen.Class = if (kind == .pow) .float else .numeric;
        const x = try g.operand(.{ .class = class });
        const dt = g.shapeOf(x).dtype();
        var value: f64 = @floatFromInt(g.intIn(i64, -4, 4));
        if (Value.isFloat(dt) and g.chance(0.4)) value = g.choose(f64, &.{ 0.5, -0.25, 1.5, 0.75 });
        if (isUnsigned(dt) and value < 0) value = -value;
        if ((kind == .div or kind == .remainder) and value == 0) value = 2;
        try g.avoid(.xla_add_zero_sign, kind == .add and value == 0);
        try g.avoid(.xla_bool_mul_sign, kind == .scale and k.gen.fromBool(&g.prog, x));
        try g.avoid(.xla_const_reassoc, Value.isFloat(dt) and (kind == .add or kind == .sub) and constAddBefore(g, x));
        if (kind == .pow) value = g.choose(f64, &.{ 2, 3, 0.5, -1, 1 });
        return .{ .op = .{ .binary_const = .{ .kind = kind, .value = value } }, .inputs = try g.allocator.dupe(u32, &.{x}), .shape = g.shapeOf(x) };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        // Equivalent to the binary op with a scalar of the same dtype.
        const dt = node.shape.dtype();
        var c = try e.alloc(.init(.{}, dt));
        if (Value.isFloat(dt)) {
            c.setF(0, attrs.value);
            _ = try c.trackErrs(e.allocator);
        } else c.setI(0, @intFromFloat(attrs.value));
        const kind: binary.Kind = switch (attrs.kind) {
            .add => .add,
            .sub => .sub,
            .scale => .mul,
            .div => .div,
            .remainder => .remainder,
            .pow => .pow,
        };
        if (kind == .pow) {
            // pow with a constant exponent; the base must stay positive.
            return binary.eval(e, .{ .kind = kind, .bcast = .rhs_scalar }, node, &.{ ins[0], c });
        }
        return binary.eval(e, .{ .kind = kind, .bcast = .rhs_scalar }, node, &.{ ins[0], c });
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        if (Value.isFloat(node.shape.dtype())) {
            const c = attrs.value;
            return switch (attrs.kind) {
                .add => x.addConstant(c),
                .sub => x.subConstant(c),
                .scale => x.scale(c),
                .div => x.divByConst(c),
                .remainder => x.remainderConst(c),
                .pow => x.powByConst(c),
            };
        }
        const c: i64 = @intFromFloat(attrs.value);
        return switch (attrs.kind) {
            .add => x.addConstant(c),
            .sub => x.subConstant(c),
            .scale => x.scale(c),
            .div => x.divByConst(c),
            .remainder => x.remainderConst(c),
            .pow => x.powByConst(c),
        };
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const name = switch (attrs.kind) {
            .add => "addConstant",
            .sub => "subConstant",
            .scale => "scale",
            .div => "divByConst",
            .remainder => "remainderConst",
            .pow => "powByConst",
        };
        if (Value.isFloat(node.shape.dtype())) return w.print("{s}.{s}({d})", .{ ins[0], name, attrs.value });
        try w.print("{s}.{s}({d})", .{ ins[0], name, @as(i64, @intFromFloat(attrs.value)) });
    }
};

// -----------------------------------------------------------------------------
// Comparisons

pub const compare = struct {
    pub const Direction = enum { EQ, NE, GE, GT, LE, LT };
    pub const Attrs = struct { dir: Direction, bcast: Broadcast };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 6;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dir = g.rng.enumValue(Direction);
        // cmp only broadcasts scalars.
        var a: u32 = undefined;
        var b: u32 = undefined;
        var sh: Shape = undefined;
        var mode: Broadcast = .same;
        while (true) {
            a, b, mode, sh = try binaryOperands(g, .any);
            if (mode != .rhs_ones) break;
        }
        // A scalar lhs is broadcast to the rhs shape, keeping the rhs tags.
        return .{ .op = .{ .compare = .{ .dir = dir, .bcast = mode } }, .inputs = try g.allocator.dupe(u32, &.{ a, b }), .shape = sh.withDtype(.bool) };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const a = ins[0];
        const b = ins[1];
        const v = try e.alloc(node.shape);
        const float = Value.isFloat(a.dtype());
        const margin = 4 * (a.err + b.err);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        while (!it.done) : (i += 1) {
            const oa = bcastOffset(a, it.idx[0..node.shape.rank()]);
            const ob = bcastOffset(b, it.idx[0..node.shape.rank()]);
            const ord: std.math.Order = if (float) blk: {
                const x = a.getF(oa);
                const y = b.getF(ob);
                if (margin > 0 and @abs(x - y) <= margin) try e.reject();
                break :blk std.math.order(x, y);
            } else std.math.order(a.getI(oa), b.getI(ob));
            const r = switch (attrs.dir) {
                .EQ => ord == .eq,
                .NE => ord != .eq,
                .GE => ord != .lt,
                .GT => ord == .gt,
                .LE => ord != .gt,
                .LT => ord == .lt,
            };
            v.setI(i, @intFromBool(r));
            _ = it.next();
        }
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const Dir = @typeInfo(@TypeOf(Tensor.cmp)).@"fn".params[1].type.?;
        return switch (attrs.dir) {
            inline else => |d| ins[0].cmp(@field(Dir, @tagName(d)), ins[1]),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.cmp(.{s}, {s})", .{ ins[0], @tagName(attrs.dir), ins[1] });
    }
};

// -----------------------------------------------------------------------------
// select(pred, on_true, on_false)

pub const select = struct {
    pub const Mode = enum { same, true_scalar, false_scalar, both_scalar };
    pub const Attrs = struct { mode: Mode };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 5;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const mode = g.rng.enumValue(Mode);
        const pred = try g.operand(.{ .dtype = .bool, .min_rank = if (mode == .same) 0 else 1 });
        const psh = g.shapeOf(pred);
        const dt = g.randDtype(.any) orelse return null;
        // Dims must match the predicate, tags may differ: the result keeps on_true's tags.
        var vsh = psh.withDtype(dt);
        if (g.chance(0.4)) {
            var tags: [Shape.MAX_RANK]Shape.Tag = undefined;
            g.randTags(tags[0..psh.rank()]);
            for (0..psh.rank()) |ax| vsh._tags.set(ax, tags[ax]);
        }
        const scalar_sh: Shape = .init(.{}, dt);
        const t = try g.operandLike(if (mode == .true_scalar or mode == .both_scalar) scalar_sh else vsh, false);
        const f = try g.operandLike(if (mode == .false_scalar or mode == .both_scalar) scalar_sh else vsh, false);
        const out = switch (mode) {
            .same, .false_scalar => g.shapeOf(t),
            .true_scalar, .both_scalar => psh.withDtype(dt),
        };
        return .{ .op = .{ .select = .{ .mode = mode } }, .inputs = try g.allocator.dupe(u32, &.{ pred, t, f }), .shape = out };
    }

    pub fn eval(e: *Eval, _: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const p = ins[0];
        const t = ins[1];
        const f = ins[2];
        var v = try e.alloc(node.shape);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        while (!it.done) : (i += 1) {
            const idx = it.idx[0..node.shape.rank()];
            if (p.getI(bcastOffset(p, idx)) != 0) v.copyElem(i, t, bcastOffset(t, idx)) else v.copyElem(i, f, bcastOffset(f, idx));
            _ = it.next();
        }
        v.err = @max(t.err, f.err);
        return v;
    }

    pub fn lower(_: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return ins[0].select(ins[1], ins[2]);
    }

    pub fn emit(_: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.select({s}, {s})", .{ ins[0], ins[1], ins[2] });
    }
};

// -----------------------------------------------------------------------------
// clamp(x, min, max)

pub const clamp = struct {
    pub const Attrs = struct { scalar_bounds: bool };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .class = .numeric });
        const sh = g.shapeOf(x);
        const scalar_bounds = g.chance(0.6);
        const bsh = if (scalar_bounds) Shape.init(.{}, sh.dtype()) else sh;
        const lo = try g.operandLike(bsh, false);
        const hi = try g.operandLike(bsh, false);
        return .{ .op = .{ .clamp = .{ .scalar_bounds = scalar_bounds } }, .inputs = try g.allocator.dupe(u32, &.{ x, lo, hi }), .shape = sh };
    }

    pub fn eval(e: *Eval, _: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const lo = ins[1];
        const hi = ins[2];
        var v = try e.alloc(node.shape);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        const float = Value.isFloat(x.dtype());
        while (!it.done) : (i += 1) {
            const idx = it.idx[0..node.shape.rank()];
            // stablehlo.clamp is min(max(x, lo), hi).
            if (float) {
                v.setF(i, @min(@max(x.getF(i), lo.getF(bcastOffset(lo, idx))), hi.getF(bcastOffset(hi, idx))));
            } else {
                v.setI(i, @min(@max(x.getI(i), lo.getI(bcastOffset(lo, idx))), hi.getI(bcastOffset(hi, idx))));
            }
            _ = it.next();
        }
        v.err = @max(x.err, @max(lo.err, hi.err));
        return v;
    }

    pub fn lower(_: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return ins[0].clamp(ins[1], ins[2]);
    }

    pub fn emit(_: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.clamp({s}, {s})", .{ ins[0], ins[1], ins[2] });
    }
};

// -----------------------------------------------------------------------------
// convert and bitCast

pub const convert = struct {
    pub const Attrs = struct { bitcast: bool };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 6;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const bitcast = g.chance(0.2);
        const x = try g.operand(.{ .allow_storage = !bitcast });
        const sh = g.shapeOf(x);
        const src = sh.dtype();
        var dst = g.randDtype(.any) orelse return null;
        // Sometimes into a storage-only dtype: sub-byte ints, f8 and f4 floats.
        if (!bitcast and !Value.isStorage(src) and src != .bool and g.chance(0.25)) dst = g.choose(DataType, &k.gen.Config.storage_dtypes);
        try g.avoid(.xla_f4_to_int, !bitcast and src == .f4e2m1 and Value.isInt(dst));
        if (bitcast) {
            // Same-width bitcasts only; bool has no defined bit layout.
            if (src == .bool) return error.Reject;
            var cands: [16]DataType = undefined;
            var n: usize = 0;
            for (g.cfg.dtypes) |dt| {
                if (dt != .bool and dt.bitSizeOf() == src.bitSizeOf() and n < cands.len) {
                    cands[n] = dt;
                    n += 1;
                }
            }
            dst = cands[g.rng.uintLessThan(usize, n)];
        }
        return .{ .op = .{ .convert = .{ .bitcast = bitcast } }, .inputs = try g.allocator.dupe(u32, &.{x}), .shape = sh.withDtype(dst) };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        var v = try e.alloc(node.shape);
        const src = x.dtype();
        const dst = node.shape.dtype();
        if (attrs.bitcast) {
            @memcpy(v.bytes, x.bytes);
            // Reinterpreting imprecise floats amplifies the error arbitrarily.
            if (x.err > 0) try e.reject();
            if (Value.isFloat(dst) and !v.allFinite()) try e.reject();
            return v;
        }
        const n = x.count();
        if (Value.isFloat(dst)) {
            // Round to nearest even: exact inputs give bit exact results, except where the
            // reference itself may round twice (f64 or large integers through f32).
            const errs = try v.trackErrs(e.allocator);
            for (0..n) |i| {
                const a = x.getF(i);
                // Overflowing conversions are not specified (NaN, inf or saturation).
                if (@abs(a) > 0.9 * Value.floatLimit(dst)) try e.reject();
                // e8m0 only holds positive powers of two.
                if (dst == .f8e8m0 and !(a > 0)) try e.reject();
                // XLA rounds ties to e8m0 up, to the larger power of two.
                const exact = if (dst == .f8e8m0) blk: {
                    const fr = std.math.frexp(a);
                    v.setF(i, std.math.ldexp(@as(f64, 1), fr.exponent - @as(i32, if (fr.significand < 0.75) 1 else 0)));
                    break :blk v.getF(i) == a;
                } else k.storeF(v, i, a);
                v.rounded = v.rounded or !exact;
                const ea = x.inErr(i);
                const twice = (src == .f64 and dst.bitSizeOf() < 32 and @as(f32, @floatCast(a)) != a) or
                    (!Value.isFloat(src) and @abs(a) > 0x1p53);
                errs[i] = if (ea > 0 or twice) ea + Value.ulp(dst, v.getF(i)) else if (v.getF(i) == 0 and x.errs == null and Value.isFloat(src)) Value.any_zero_sign else 0;
            }
            v.finishErrs();
            return v;
        }
        if (Value.isFloat(src)) {
            // Float to integer truncates toward zero; out of range values are not specified.
            for (0..n) |i| {
                const a = x.getF(i);
                const r: f64 = if (dst == .bool) (if (a != 0) 1 else 0) else @trunc(a);
                if (x.err > 0) {
                    const d = if (dst == .bool) @abs(a) else distToInt(a);
                    if (d <= 4 * x.err and !(dst != .bool and @trunc(a - 4 * x.err) == @trunc(a + 4 * x.err))) try e.reject();
                }
                if (dst != .bool and (r < @as(f64, @floatFromInt(Value.minInt(dst))) or r > @as(f64, @floatFromInt(Value.maxInt(dst))))) {
                    try e.reject();
                    continue;
                }
                v.setI(i, @intFromFloat(r));
            }
            return v;
        }
        // Integer to integer: values that don't fit are implementation defined.
        for (0..n) |i| {
            const a = x.getI(i);
            if (dst == .bool) {
                v.setI(i, @intFromBool(a != 0));
                continue;
            }
            if (a < Value.minInt(dst) or a > Value.maxInt(dst)) try e.reject();
            v.setI(i, a);
        }
        return v;
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        if (attrs.bitcast) return ins[0].bitCast(node.shape.dtype());
        const x = ins[0];
        if (k.ops.mutation == .truncating_convert and x.dtype() == .f32 and node.shape.dtype() == .bf16) {
            const hi = x.bitCast(.u32).shiftRightLogical(Tensor.scalar(16, .u32).broad(x.shape().withDtype(.u32)));
            return hi.convert(.u16).bitCast(.bf16);
        }
        return x.convert(node.shape.dtype());
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.{s}(.{s})", .{ ins[0], if (attrs.bitcast) "bitCast" else "convert", @tagName(node.shape.dtype()) });
    }
};
