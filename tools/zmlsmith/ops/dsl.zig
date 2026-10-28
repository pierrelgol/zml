//! The generic zml.ops building blocks: reduce over several inputs and axes with custom
//! bodies, and sort with custom comparators over several inputs.

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
const List = k.List;

const ReduceArgs = zml.ops.ReduceArgs;
const SortArgs = zml.ops.SortArgs;

pub const reduce_n = struct {
    /// Reducer bodies. `pair_sum_max` reduces two inputs independently; `argmax` is the
    /// classic (value, index) pair keeping the first maximum.
    pub const Body = enum { sum, prod, max, min, bit_and, bit_or, bit_xor, pair_sum_max, argmax };
    pub const Attrs = struct { body: Body, axes: List(u8, 8), pick: u8 };
    pub const weight = 4;

    fn arity(b: Body) usize {
        return switch (b) {
            .pair_sum_max, .argmax => 2,
            else => 1,
        };
    }

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const body = g.rng.enumValue(Body);
        const class: k.gen.Class = switch (body) {
            .bit_and, .bit_or, .bit_xor => .int_or_bool,
            .prod => .int,
            else => .numeric,
        };
        try g.avoid(.xla_spmd_bitwise_reduce, g.prog.sharded and (body == .bit_and or body == .bit_or or body == .bit_xor));
        const x = try g.operand(.{ .class = class, .min_rank = 1 });
        const sh = g.shapeOf(x);
        var a: Attrs = .{ .body = body, .axes = .{}, .pick = 0 };
        var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        g.rng.shuffle(u8, perm[0..sh.rank()]);
        const n = g.intIn(usize, 1, sh.rank());
        for (perm[0..n]) |ax| {
            if (sh.dim(ax) == 0) return error.Reject;
            a.axes.append(ax);
        }
        // Axes are given in increasing order.
        std.mem.sort(u8, a.axes.mut(), {}, std.sort.asc(u8));
        var ids: [2]u32 = .{ x, 0 };
        var out = sh;
        for (a.axes.slice()) |ax| out = out.setDim(ax, 1);
        switch (body) {
            .pair_sum_max => {
                ids[1] = try g.operandLike(sh.withDtype(g.randDtype(.numeric) orelse sh.dtype()), true);
                a.pick = g.intIn(u8, 0, 1);
                if (a.pick == 1) out = out.withDtype(g.shapeOf(ids[1]).dtype());
            },
            .argmax => {
                // Indices along the single reduced axis.
                if (a.axes.len != 1) return error.Reject;
                const iv = try Value.alloc(g.allocator, sh.withDtype(.i32));
                var it: Value.Index = .init(sh);
                var i: usize = 0;
                while (!it.done) : (i += 1) {
                    iv.setI(i, it.idx[a.axes.slice()[0]]);
                    _ = it.next();
                }
                ids[1] = try g.literalInput(iv);
                a.pick = g.intIn(u8, 0, 1);
                if (a.pick == 1) out = out.withDtype(.i32);
            },
            else => {},
        }
        return .{ .op = .{ .reduce_n = a }, .inputs = try k.dupeIds(g, ids[0..arity(body)]), .shape = out };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const which = ins[a.pick];
        const dt = which.dtype();
        var v = try e.alloc(node.shape);
        const st = Value.strides(x.shape);
        var reduced: [8]bool = @splat(false);
        for (a.axes.slice()) |ax| reduced[ax] = true;
        var lane_sh: Shape = .init(.{}, .i32);
        for (a.axes.slice()) |ax| lane_sh = lane_sh.appendDim(x.shape.dim(ax), null);
        var out_it: Value.Index = .init(node.shape);
        var o: usize = 0;
        var err: f64 = 0;
        while (!out_it.done) : (o += 1) {
            var acc_i: i128 = 0;
            var acc_f: f64 = 0;
            var best: ?usize = null;
            var abs_sum: f64 = 0;
            var integral = true;
            var count: usize = 0;
            const float = Value.isFloat(dt);
            switch (a.body) {
                .prod => acc_i = 1,
                .bit_and => acc_i = -1,
                .max => if (float) {
                    acc_f = -std.math.inf(f64);
                } else {
                    acc_i = Value.minInt(dt);
                },
                .min => if (float) {
                    acc_f = std.math.inf(f64);
                } else {
                    acc_i = Value.maxInt(dt);
                },
                .pair_sum_max => if (a.pick == 1) {
                    if (float) acc_f = -std.math.inf(f64) else acc_i = Value.minInt(dt);
                },
                else => {},
            }
            var lit: Value.Index = .init(lane_sh);
            while (!lit.done) {
                var idx: [8]i64 = undefined;
                var r: usize = 0;
                for (0..x.shape.rank()) |ax| {
                    if (reduced[ax]) {
                        idx[ax] = lit.idx[r];
                        r += 1;
                    } else idx[ax] = out_it.idx[ax];
                }
                const off = Value.offset(st, idx[0..x.shape.rank()]);
                count += 1;
                const body: Body = if (a.body == .pair_sum_max) (if (a.pick == 0) .sum else .max) else a.body;
                switch (body) {
                    .argmax => {
                        // Documented ArgMaxRes behavior: first maximum.
                        const gt = if (best) |b|
                            (if (Value.isFloat(x.dtype())) x.getF(off) > x.getF(b) else x.getI(off) > x.getI(b))
                        else
                            true;
                        if (gt) best = off;
                    },
                    else => if (float) {
                        const xv = which.getF(off);
                        abs_sum += @abs(xv);
                        integral = integral and xv == @round(xv);
                        acc_f = switch (body) {
                            .sum => acc_f + xv,
                            .max => @max(acc_f, xv),
                            .min => @min(acc_f, xv),
                            else => unreachable,
                        };
                    } else {
                        const xv = which.getI(off);
                        acc_i = switch (body) {
                            .sum => Value.wrapInt(dt, acc_i + xv),
                            .prod => Value.wrapInt(dt, acc_i *% xv),
                            .max => @max(acc_i, xv),
                            .min => @min(acc_i, xv),
                            .bit_and => acc_i & xv,
                            .bit_or => acc_i | xv,
                            .bit_xor => acc_i ^ xv,
                            else => unreachable,
                        };
                    },
                }
                _ = lit.next();
            }
            if (a.body == .argmax) {
                const b = best.?;
                if (Value.isFloat(x.dtype()) and x.err > 0) try e.reject();
                if (a.pick == 0) v.copyElem(o, x, b) else v.setI(o, ins[1].getI(b));
            } else if (float) {
                var lane_err = @as(f64, @floatFromInt(count)) * which.err;
                if (!(which.err == 0 and integral and abs_sum < std.math.ldexp(@as(f64, 1), Value.precisionBits(dt))))
                    lane_err += @as(f64, @floatFromInt(count + 1)) * Value.eps(dt) * abs_sum;
                if (!k.storeF(v, o, acc_f)) lane_err += k.roundErr(dt, @abs(acc_f));
                err = @max(err, lane_err);
            } else v.setI(o, acc_i);
            _ = out_it.next();
        }
        v.err = if (a.body == .argmax and a.pick == 0) x.err else err;
        return v;
    }

    fn init(b: Body, dt: DataType) Tensor {
        return switch (b) {
            .sum, .pair_sum_max, .bit_or, .bit_xor => .constant(dt.zero()),
            .prod => .constant(dt.one()),
            .max, .argmax => .constant(dt.minValue()),
            .min => .constant(dt.maxValue()),
            .bit_and => if (dt == .bool) Tensor.scalar(true, .bool) else if (dt.isSignedInt()) Tensor.scalar(-1, dt) else Tensor.constant(dt.maxValue()),
        };
    }

    const Bodies = struct {
        fn sum(a: ReduceArgs) struct { Tensor } {
            return .{a.left.add(a.right)};
        }
        fn prod(a: ReduceArgs) struct { Tensor } {
            return .{a.left.mul(a.right)};
        }
        fn max(a: ReduceArgs) struct { Tensor } {
            return .{a.left.maximum(a.right)};
        }
        fn min(a: ReduceArgs) struct { Tensor } {
            return .{a.left.minimum(a.right)};
        }
        fn bitAnd(a: ReduceArgs) struct { Tensor } {
            return .{a.left.logical(.AND, a.right)};
        }
        fn bitOr(a: ReduceArgs) struct { Tensor } {
            return .{a.left.logical(.OR, a.right)};
        }
        fn bitXor(a: ReduceArgs) struct { Tensor } {
            return .{a.left.logical(.XOR, a.right)};
        }
        fn sumMax(a: ReduceArgs, b: ReduceArgs) struct { Tensor, Tensor } {
            return .{ a.left.add(a.right), b.left.maximum(b.right) };
        }
    };

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        var axes_: [8]i64 = undefined;
        for (a.axes.slice(), 0..) |ax, i| axes_[i] = ax;
        const ax = axes_[0..a.axes.len];
        const x = ins[0];
        const dt = x.dtype();
        return switch (a.body) {
            .sum => zml.ops.reduce(.{x}, .{init(.sum, dt)}, ax, Bodies.sum, .{})[0],
            .prod => zml.ops.reduce(.{x}, .{init(.prod, dt)}, ax, Bodies.prod, .{})[0],
            .max => zml.ops.reduce(.{x}, .{init(.max, dt)}, ax, Bodies.max, .{})[0],
            .min => zml.ops.reduce(.{x}, .{init(.min, dt)}, ax, Bodies.min, .{})[0],
            .bit_and => zml.ops.reduce(.{x}, .{init(.bit_and, dt)}, ax, Bodies.bitAnd, .{})[0],
            .bit_or => zml.ops.reduce(.{x}, .{init(.bit_or, dt)}, ax, Bodies.bitOr, .{})[0],
            .bit_xor => zml.ops.reduce(.{x}, .{init(.bit_xor, dt)}, ax, Bodies.bitXor, .{})[0],
            .pair_sum_max => blk: {
                const r = zml.ops.reduce(.{ x, ins[1] }, .{ init(.sum, dt), init(.max, ins[1].dtype()) }, ax, Bodies.sumMax, .{});
                break :blk if (a.pick == 0) r[0] else r[1];
            },
            .argmax => blk: {
                const r = zml.ops.reduce(.{ x, ins[1] }, .{ init(.max, dt), Tensor.scalar(0, .i32) }, ax, argmaxBody, .{});
                break :blk r[a.pick];
            },
        };
    }

    /// The same comparator as Tensor.ArgMaxRes: NaN wins, ties keep the smallest index.
    fn argmaxBody(values: ReduceArgs, indices: ReduceArgs) [2]Tensor {
        const gt = values.left.cmp(.GT, values.right);
        const is_nan = values.left.cmp(.NE, values.left);
        const gt_or_nan = gt.logical(.OR, is_nan);
        const same = values.left.cmp(.EQ, values.right);
        const first = indices.left.cmp(.LT, indices.right);
        const keep_left = gt_or_nan.logical(.OR, same.logical(.AND, first));
        return .{ gt_or_nan.select(values.left, values.right), keep_left.select(indices.left, indices.right) };
    }

    pub fn emit(a: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const dt = @tagName(k.emit_shapes[0].dtype());
        try w.print("zml.ops.reduce(.{{ {s}", .{ins[0]});
        if (ins.len > 1) try w.print(", {s}", .{ins[1]});
        try w.writeAll(" }, .{ ");
        switch (a.body) {
            .sum, .bit_or, .bit_xor => try w.print("zml.Tensor.constant(zml.DataType.zero(.{s}))", .{dt}),
            .prod => try w.print("zml.Tensor.constant(zml.DataType.one(.{s}))", .{dt}),
            .max => try w.print("zml.Tensor.constant(zml.DataType.minValue(.{s}))", .{dt}),
            .min => try w.print("zml.Tensor.constant(zml.DataType.maxValue(.{s}))", .{dt}),
            .bit_and => if (k.emit_shapes[0].dtype() == .bool) try w.writeAll("zml.Tensor.scalar(true, .bool)") else if (k.emit_shapes[0].dtype().isSignedInt()) try w.print("zml.Tensor.scalar(-1, .{s})", .{dt}) else try w.print("zml.Tensor.constant(zml.DataType.maxValue(.{s}))", .{dt}),
            .pair_sum_max => try w.print("zml.Tensor.constant(zml.DataType.zero(.{s})), zml.Tensor.constant(zml.DataType.minValue(.{s}))", .{ dt, @tagName(k.emit_shapes[1].dtype()) }),
            .argmax => try w.print("zml.Tensor.constant(zml.DataType.minValue(.{s})), zml.Tensor.scalar(0, .i32)", .{dt}),
        }
        try w.writeAll(" }, &[_]i64{ ");
        for (a.axes.slice(), 0..) |ax, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", ax });
        try w.writeAll(" }, struct {\n            fn f(");
        switch (a.body) {
            .pair_sum_max => try w.writeAll("a: zml.ops.ReduceArgs, b: zml.ops.ReduceArgs) struct { zml.Tensor, zml.Tensor } { return .{ a.left.add(a.right), b.left.maximum(b.right) }; }"),
            .argmax => try w.writeAll("v: zml.ops.ReduceArgs, i: zml.ops.ReduceArgs) [2]zml.Tensor { const gt = v.left.cmp(.GT, v.right).logical(.OR, v.left.cmp(.NE, v.left)); const keep = gt.logical(.OR, v.left.cmp(.EQ, v.right).logical(.AND, i.left.cmp(.LT, i.right))); return .{ gt.select(v.left, v.right), keep.select(i.left, i.right) }; }"),
            else => {
                const op = switch (a.body) {
                    .sum => "add(a.right)",
                    .prod => "mul(a.right)",
                    .max => "maximum(a.right)",
                    .min => "minimum(a.right)",
                    .bit_and => "logical(.AND, a.right)",
                    .bit_or => "logical(.OR, a.right)",
                    .bit_xor => "logical(.XOR, a.right)",
                    else => unreachable,
                };
                try w.print("a: zml.ops.ReduceArgs) struct {{ zml.Tensor }} {{ return .{{a.left.{s}}}; }}", .{op});
            },
        }
        try w.print("\n        }}.f, .{{}})[{d}]", .{a.pick});
        _ = node;
    }
};

// -----------------------------------------------------------------------------

pub const sort_n = struct {
    pub const Comparator = enum { first_asc, first_desc, lexicographic, second_desc };
    pub const Attrs = struct { cmp: Comparator, axis: u8, stable: bool, n: u8, pick: u8 };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const cmp = g.rng.enumValue(Comparator);
        const x = try g.operand(.{ .class = .numeric, .min_rank = 1, .exact = true });
        const sh = g.shapeOf(x);
        const n: u8 = if (cmp == .lexicographic or cmp == .second_desc) 2 else g.intIn(u8, 1, 3);
        var ids: [3]u32 = undefined;
        ids[0] = x;
        for (1..n) |i| {
            const dt = g.randDtype(.any) orelse return null;
            const want: Shape = sh.withDtype(dt);
            ids[i] = g.pickLike(want, true) orelse try g.freshInput(want);
            if (g.valueOf(ids[i]).err > 0) return error.Reject;
        }
        if ((cmp == .lexicographic or cmp == .second_desc) and g.shapeOf(ids[1]).dtype() == .bool) return error.Reject;
        const ax = g.rng.uintLessThan(usize, sh.rank());
        if (sh.dim(ax) == 0) return error.Reject;
        const pick = g.intIn(u8, 0, n - 1);
        return .{
            .op = .{ .sort_n = .{ .cmp = cmp, .axis = @intCast(ax), .stable = g.chance(0.5), .n = n, .pick = pick } },
            .inputs = try k.dupeIds(g, ids[0..n]),
            .shape = g.shapeOf(ids[pick]),
        };
    }

    /// Compares elements `l` and `r` of `v`, exactly for integers.
    fn cmpElem(v: Value, l: usize, r: usize) std.math.Order {
        if (Value.isFloat(v.dtype())) return std.math.order(v.getF(l), v.getF(r));
        return std.math.order(v.getI(l), v.getI(r));
    }

    fn order(a: Attrs, ins: []const Value, l: usize, r: usize) std.math.Order {
        return switch (a.cmp) {
            .first_asc => cmpElem(ins[0], l, r),
            .first_desc => cmpElem(ins[0], r, l),
            .lexicographic => blk: {
                const o = cmpElem(ins[0], l, r);
                break :blk if (o != .eq) o else cmpElem(ins[1], l, r);
            },
            .second_desc => cmpElem(ins[1], r, l),
        };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const sh = ins[0].shape;
        const ax = a.axis;
        const d: usize = @intCast(sh.dim(ax));
        var v = try e.alloc(node.shape);
        const st = Value.strides(sh);
        const idx = try e.allocator.alloc(usize, d);
        var it: Value.Index = .init(sh.setDim(ax, 1));
        while (!it.done) {
            const base = Value.offset(st, it.idx[0..sh.rank()]);
            for (idx, 0..) |*p, j| p.* = base + j * st[ax];
            const C = struct { a: Attrs, ins: []const Value };
            // The comparator is a strict "less than": stable sort keeps ties in order.
            std.sort.block(usize, idx, C{ .a = a, .ins = ins }, struct {
                fn lt(c: C, l: usize, r: usize) bool {
                    return order(c.a, c.ins, l, r) == .lt;
                }
            }.lt);
            if (!a.stable) {
                // Unstable sorts may permute ties.
                for (1..d) |j| if (order(a, ins, idx[j - 1], idx[j]) == .eq) try e.reject();
            }
            for (idx, 0..) |src, j| v.copyElem(base + j * st[ax], ins[a.pick], src);
            _ = it.next();
        }
        return v;
    }

    const Cmp = struct {
        fn firstAsc(x: SortArgs, _: SortArgs, _: SortArgs) Tensor {
            return x.left.cmp(.LT, x.right);
        }
        fn firstDesc(x: SortArgs, _: SortArgs, _: SortArgs) Tensor {
            return x.left.cmp(.GT, x.right);
        }
        fn lexicographic(x: SortArgs, y: SortArgs) Tensor {
            const lt = x.left.cmp(.LT, x.right);
            const eq = x.left.cmp(.EQ, x.right);
            return lt.logical(.OR, eq.logical(.AND, y.left.cmp(.LT, y.right)));
        }
        fn secondDesc(_: SortArgs, y: SortArgs) Tensor {
            return y.left.cmp(.GT, y.right);
        }
    };

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const ax: i64 = a.axis;
        switch (a.n) {
            1 => return switch (a.cmp) {
                .first_asc => zml.ops.sort(.{ins[0]}, ax, struct {
                    fn f(x: SortArgs) Tensor {
                        return x.left.cmp(.LT, x.right);
                    }
                }.f, .{}, a.stable)[a.pick],
                else => zml.ops.sort(.{ins[0]}, ax, struct {
                    fn f(x: SortArgs) Tensor {
                        return x.left.cmp(.GT, x.right);
                    }
                }.f, .{}, a.stable)[a.pick],
            },
            2 => return switch (a.cmp) {
                .first_asc => zml.ops.sort(.{ ins[0], ins[1] }, ax, struct {
                    fn f(x: SortArgs, _: SortArgs) Tensor {
                        return x.left.cmp(.LT, x.right);
                    }
                }.f, .{}, a.stable)[a.pick],
                .first_desc => zml.ops.sort(.{ ins[0], ins[1] }, ax, struct {
                    fn f(x: SortArgs, _: SortArgs) Tensor {
                        return x.left.cmp(.GT, x.right);
                    }
                }.f, .{}, a.stable)[a.pick],
                .lexicographic => zml.ops.sort(.{ ins[0], ins[1] }, ax, Cmp.lexicographic, .{}, a.stable)[a.pick],
                .second_desc => zml.ops.sort(.{ ins[0], ins[1] }, ax, Cmp.secondDesc, .{}, a.stable)[a.pick],
            },
            else => return switch (a.cmp) {
                .first_asc => zml.ops.sort(.{ ins[0], ins[1], ins[2] }, ax, Cmp.firstAsc, .{}, a.stable)[a.pick],
                else => zml.ops.sort(.{ ins[0], ins[1], ins[2] }, ax, Cmp.firstDesc, .{}, a.stable)[a.pick],
            },
        }
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.writeAll("zml.ops.sort(.{ ");
        try k.printOperands(w, ins);
        try w.print(" }}, {d}, struct {{\n            fn f(", .{a.axis});
        for (0..a.n) |i| try w.print("{s}{s}: zml.ops.SortArgs", .{ if (i > 0) ", " else "", ([_][]const u8{ "x", "y", "z" })[i] });
        try w.writeAll(") zml.Tensor {\n                ");
        for (1..a.n) |i| try w.print("_ = &{s};\n                ", .{([_][]const u8{ "x", "y", "z" })[i]});
        switch (a.cmp) {
            .first_asc => try w.writeAll("return x.left.cmp(.LT, x.right);"),
            .first_desc => try w.writeAll("return x.left.cmp(.GT, x.right);"),
            .lexicographic => try w.writeAll("return x.left.cmp(.LT, x.right).logical(.OR, x.left.cmp(.EQ, x.right).logical(.AND, y.left.cmp(.LT, y.right)));"),
            .second_desc => try w.writeAll("_ = &x;\n                return y.left.cmp(.GT, y.right);"),
        }
        try w.print("\n            }}\n        }}.f, .{{}}, {})[{d}]", .{ a.stable, a.pick });
    }
};
