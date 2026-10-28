//! Data dependent indexing: gather, dynamic slices and updates, sort and topK.

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

const index_dtypes = [_]DataType{ .i32, .i32, .i64, .u32, .i8, .u8, .u16 };

fn indexDtype(g: *Gen, max_value: i64) DataType {
    while (true) {
        const dt = g.choose(DataType, &index_dtypes);
        if (Value.maxInt(dt) < max_value) continue;
        if (g.prog.sharded and dt.bitSizeOf() < 32 and g.cfg.avoid.contains(.xla_spmd_small_indices)) continue;
        return dt;
    }
}

/// A literal index tensor of shape `sh` with values in [0, bound).
fn indexLiteral(g: *Gen, sh: Shape, bound: i64) k.GenError!u32 {
    return indexLiteralSorted(g, sh, bound, false);
}

/// Random indices in [0, bound), ascending in flat order when `sorted`.
fn indexLiteralSorted(g: *Gen, sh: Shape, bound: i64, sorted: bool) k.GenError!u32 {
    const v = try Value.alloc(g.allocator, sh);
    var lo: i64 = 0;
    for (0..v.count()) |i| {
        const x = g.intIn(i64, if (sorted) lo else 0, bound - 1);
        if (sorted) lo = x;
        v.setI(i, x);
    }
    return g.literalInput(v);
}

// -----------------------------------------------------------------------------

pub const gather = struct {
    pub const Attrs = struct {
        /// Gathered axes of the operand, consecutive.
        axes: List(u8, 2),
        /// Use `Tensor.gather(.{ .a = idx })` instead of `zml.ops.gather`.
        tagged_api: bool,
        /// GatherOpts.indices_are_sorted, set when the indices are ascending.
        sorted: bool = false,
    };
    pub const weight = 5;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 1, .max_rank = 4 });
        const sh = g.shapeOf(x);
        try g.avoid(.gather_coord_tag, sh.hasTag(.coord) != null);
        const n_axes: usize = if (sh.rank() >= 2 and g.chance(0.3)) 2 else 1;
        const first = g.rng.uintLessThan(usize, sh.rank() - n_axes + 1);
        var tagged_api = g.chance(0.6);
        for (first..first + n_axes) |ax| {
            if (sh.dim(ax) == 0) return error.Reject;
            if (sh.tag(ax) == Shape.TagUnknown) tagged_api = false;
        }
        // Indices: batching axes first (tags shared with the operand, same order), then new
        // axes with fresh tags.
        var ish: Shape = .init(.{}, indexDtype(g, sh.dim(first) + 8));
        for (0..sh.rank()) |ax| {
            if (ax >= first and ax < first + n_axes) continue;
            const t = sh.tag(ax);
            if (t != Shape.TagUnknown and g.chance(0.25)) ish = ish.appendDim(sh.dim(ax), t);
        }
        const n_new = g.intIn(usize, 0, 2);
        for (0..n_new) |_| {
            if (ish.rank() + sh.rank() >= Shape.MAX_RANK) break;
            var t = g.freshTag(sh);
            while (ish.hasTag(t) != null) t = g.freshTag(sh);
            ish = ish.appendDim(g.intIn(i64, 1, 4), if (g.chance(0.8)) t else Shape.TagUnknown);
        }
        for (0..ish.rank()) |iax| {
            if (sh.hasTag(ish.tag(iax))) |sax| try g.avoid(.gather_batching_after, sax > first);
        }
        var ids: [3]u32 = undefined;
        ids[0] = x;
        var a: Attrs = .{ .axes = .{}, .tagged_api = tagged_api, .sorted = n_axes == 1 and g.chance(0.3) };
        for (0..n_axes) |j| {
            ids[1 + j] = try indexLiteralSorted(g, ish, sh.dim(first + j), a.sorted);
            a.axes.append(@intCast(first + j));
        }
        return .{ .op = .{ .gather = a }, .inputs = try k.dupeIds(g, ids[0 .. 1 + n_axes]), .shape = resultShape(sh, a.axes.slice(), ish) };
    }

    const Kind = union(enum) { offset: u8, batching: struct { self_ax: u8, idx_ax: u8 }, indices: u8 };

    /// Documented: the operand shape where the gathered axes are replaced by the
    /// non-batching axes of the indices, inserted at the first gathered axis.
    fn layout(sh: Shape, axes_: []const u8, ish: Shape, out: *[8]Kind) usize {
        var n: usize = 0;
        for (0..sh.rank()) |ax| {
            if (ax == axes_[0]) {
                for (0..ish.rank()) |iax| {
                    const t = ish.tag(iax);
                    if (t != Shape.TagUnknown and sh.hasTag(t) != null) continue;
                    out[n] = .{ .indices = @intCast(iax) };
                    n += 1;
                }
            }
            if (std.mem.indexOfScalar(u8, axes_, @intCast(ax)) != null) continue;
            const t = sh.tag(ax);
            if (t != Shape.TagUnknown) if (ish.hasTag(t)) |iax| {
                out[n] = .{ .batching = .{ .self_ax = @intCast(ax), .idx_ax = iax } };
                n += 1;
                continue;
            };
            out[n] = .{ .offset = @intCast(ax) };
            n += 1;
        }
        return n;
    }

    fn resultShape(sh: Shape, axes_: []const u8, ish: Shape) Shape {
        var kinds: [8]Kind = undefined;
        const n = layout(sh, axes_, ish, &kinds);
        var res: Shape = .init(.{}, sh.dtype());
        for (kinds[0..n]) |kd| switch (kd) {
            .offset => |ax| res = res.appendDim(sh.dim(ax), sh.tag(ax)),
            .batching => |b| res = res.appendDim(sh.dim(b.self_ax), sh.tag(b.self_ax)),
            .indices => |iax| res = res.appendDim(ish.dim(iax), ish.tag(iax)),
        };
        return res;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const idx = ins[1..];
        const ish = idx[0].shape;
        var kinds: [8]Kind = undefined;
        const n = layout(x.shape, attrs.axes.slice(), ish, &kinds);
        std.debug.assert(n == node.shape.rank());
        var v = try e.alloc(node.shape);
        const xs = Value.strides(x.shape);
        const is = Value.strides(ish);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        var xi: [8]i64 = @splat(0);
        var ii: [8]i64 = @splat(0);
        while (!it.done) : (i += 1) {
            for (kinds[0..n], 0..) |kd, o| switch (kd) {
                .offset => |ax| xi[ax] = it.idx[o],
                .batching => |b| {
                    xi[b.self_ax] = it.idx[o];
                    ii[b.idx_ax] = it.idx[o];
                },
                .indices => |iax| ii[iax] = it.idx[o],
            };
            const ioff = Value.offset(is, ii[0..ish.rank()]);
            for (attrs.axes.slice(), 0..) |ax, j| xi[ax] = @intCast(idx[j].getI(ioff));
            v.copyElem(i, x, Value.offset(xs, xi[0..x.shape.rank()]));
            _ = it.next();
        }
        v.err = x.err;
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const axes_ = attrs.axes.slice();
        if (attrs.tagged_api) {
            const Body = struct {
                x: Tensor,
                o: zml.ops.GatherOpts,
                pub const Result = Tensor;
                pub fn call(b: @This(), s: anytype) Tensor {
                    return b.x.gather(s, b.o);
                }
            };
            const o: zml.ops.GatherOpts = .{ .indices_are_sorted = attrs.sorted };
            if (axes_.len == 1) return k.callWithTagStruct(TagId.of(x.shape().tag(axes_[0])), ins[1], Body{ .x = x, .o = o });
            return k.callWithTagStruct2(TagId.of(x.shape().tag(axes_[0])), ins[1], TagId.of(x.shape().tag(axes_[1])), ins[2], Body{ .x = x, .o = o });
        }
        var ax: [2]u3 = undefined;
        for (axes_, 0..) |a, i| ax[i] = @intCast(a);
        return zml.ops.gather(x, ax[0..axes_.len], ins[1..], .{ .indices_are_sorted = attrs.sorted });
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        _ = node;
        const axes_ = attrs.axes.slice();
        if (attrs.tagged_api) {
            try w.print("{s}.gather(.{{ ", .{ins[0]});
            for (axes_, 0..) |a, i| try w.print("{s}.{s} = {s}", .{ if (i > 0) ", " else "", k.tagLiteral(k.emit_shapes[0].tag(a)), ins[1 + i] });
            return w.print(" }}, .{{ .indices_are_sorted = {} }})", .{attrs.sorted});
        }
        try w.print("zml.ops.gather({s}, &.{{ ", .{ins[0]});
        for (axes_, 0..) |a, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", a });
        try w.writeAll(" }, &.{ ");
        try k.printOperands(w, ins[1..]);
        try w.print(" }}, .{{ .indices_are_sorted = {} }})", .{attrs.sorted});
    }
};

// -----------------------------------------------------------------------------

pub const dynamic_slice = struct {
    pub const Attrs = struct { axis: AxisRef, len: i64, single: bool };
    pub const weight = 4;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 1 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        const d = sh.dim(ax);
        if (d == 0) return error.Reject;
        const single = g.chance(0.3);
        const len = if (single) 1 else g.intIn(i64, 1, d);
        // In range start: dynamic_slice clamps otherwise, which ZML doesn't document.
        const start = try indexLiteral(g, .init(.{}, indexDtype(g, d)), d - len + 1);
        const out = if (single) sh.remove(ax) else sh.setDim(ax, len);
        return .{ .op = .{ .dynamic_slice = .{ .axis = .random(g, sh, ax), .len = len, .single = single } }, .inputs = try k.dupeIds(g, &.{ x, start }), .shape = out };
    }

    fn map(ctx: anytype, out: []const i64, in: []i64) bool {
        var o: usize = 0;
        for (0..ctx.rank) |a| {
            if (a == ctx.ax) {
                in[a] = ctx.start + if (ctx.single) 0 else out[o];
                if (!ctx.single) o += 1;
            } else {
                in[a] = out[o];
                o += 1;
            }
        }
        return true;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = attrs.axis.resolve(x.shape);
        const start: i64 = @intCast(ins[1].getI(0));
        return k.remap(e, x, node.shape, .{ .ax = ax, .start = start, .single = attrs.single, .rank = x.shape.rank() }, map, null);
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const s: Tensor.Slice = if (attrs.single) .dynSingle(ins[1]) else .dyn(ins[1], attrs.len);
        return switch (attrs.axis.pass()) {
            inline else => |a| ins[0].slice(a, s),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.slice(", .{ins[0]});
        try attrs.axis.print(w);
        if (attrs.single) return w.print(", .dynSingle({s}))", .{ins[1]});
        try w.print(", .dyn({s}, {d}))", .{ ins[1], attrs.len });
    }
};

// -----------------------------------------------------------------------------

pub const dynamic_update = struct {
    pub const Kind = enum {
        /// dynamicUpdateSlice1d(update, axis, offset)
        one_d,
        /// dynamicUpdateSlice(.{ .a = offset }, update) with update tagged as a subset.
        tagged,
        /// dynamicUpdateSlice(&.{ o0, o1, ... }, update) with one offset per axis.
        all_axes,
    };
    pub const Attrs = struct { kind: Kind, axis: u8 };
    pub const weight = 4;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        var kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .min_rank = 1, .max_rank = 4 });
        const sh = g.shapeOf(x);
        if (kind == .tagged and !k.gen.fullyTagged(sh)) kind = .one_d;
        const ax = g.rng.uintLessThan(usize, sh.rank());
        const d = sh.dim(ax);
        if (d == 0) return error.Reject;
        var ids: [10]u32 = undefined;
        ids[0] = x;
        var n: usize = 2;
        switch (kind) {
            .one_d => {
                const len = g.intIn(i64, 1, d);
                ids[1] = try g.operandLike(sh.setDim(ax, len), true);
                const dt = indexDtype(g, d);
                try g.avoid(.dus1d_offset_dtype, dt != .i32);
                ids[2] = try indexLiteral(g, .init(.{}, dt), d - len + 1);
                n = 3;
            },
            .tagged => {
                // Tagged update with the same axes, only the offset axis is smaller.
                const len = g.intIn(i64, 1, d);
                ids[1] = try g.operandLike(sh.setDim(ax, len), true);
                ids[2] = try indexLiteral(g, .init(.{}, indexDtype(g, d)), d - len + 1);
                n = 3;
            },
            .all_axes => {
                var ush = sh;
                for (0..sh.rank()) |a| ush = ush.setDim(a, g.intIn(i64, 1, @max(1, sh.dim(a))));
                if (sh.count() == 0) return error.Reject;
                ids[1] = try g.operandLike(ush, true);
                const dt = indexDtype(g, 8);
                for (0..sh.rank()) |a| ids[2 + a] = try indexLiteral(g, .init(.{}, dt), sh.dim(a) - ush.dim(a) + 1);
                n = 2 + sh.rank();
            },
        }
        return .{ .op = .{ .dynamic_update = .{ .kind = kind, .axis = @intCast(ax) } }, .inputs = try k.dupeIds(g, ids[0..n]), .shape = sh };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const u = ins[1];
        var v = try x.dupe(e.allocator);
        var starts: [8]i64 = @splat(0);
        switch (attrs.kind) {
            .one_d, .tagged => starts[attrs.axis] = @intCast(ins[2].getI(0)),
            .all_axes => for (0..x.shape.rank()) |a| {
                starts[a] = @intCast(ins[2 + a].getI(0));
            },
        }
        const xs = Value.strides(x.shape);
        var it: Value.Index = .init(u.shape);
        var i: usize = 0;
        var xi: [8]i64 = undefined;
        while (!it.done) : (i += 1) {
            for (0..x.shape.rank()) |a| xi[a] = it.idx[a] + starts[a];
            v.copyElem(Value.offset(xs, xi[0..x.shape.rank()]), u, i);
            _ = it.next();
        }
        _ = node;
        v.err = @max(x.err, u.err);
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        switch (attrs.kind) {
            .one_d => return x.dynamicUpdateSlice1d(ins[1], attrs.axis, ins[2]),
            .tagged => {
                const Body = struct {
                    x: Tensor,
                    u: Tensor,
                    pub const Result = Tensor;
                    pub fn call(b: @This(), s: anytype) Tensor {
                        return b.x.dynamicUpdateSlice(s, b.u);
                    }
                };
                return k.callWithTagStruct(TagId.of(x.shape().tag(attrs.axis)), ins[2], Body{ .x = x, .u = ins[1] });
            },
            .all_axes => return x.dynamicUpdateSlice(ins[2..], ins[1]),
        }
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (attrs.kind) {
            .one_d => try w.print("{s}.dynamicUpdateSlice1d({s}, {d}, {s})", .{ ins[0], ins[1], attrs.axis, ins[2] }),
            .tagged => try w.print("{s}.dynamicUpdateSlice(.{{ .{s} = {s} }}, {s})", .{ ins[0], TagId.of(node.shape.tag(attrs.axis)).name(), ins[2], ins[1] }),
            .all_axes => {
                try w.print("{s}.dynamicUpdateSlice(@as([]const zml.Tensor, &.{{ ", .{ins[0]});
                try k.printOperands(w, ins[2..]);
                try w.print(" }}), {s})", .{ins[1]});
            },
        }
    }
};

// -----------------------------------------------------------------------------

pub const sort = struct {
    pub const Kind = enum { values, indices, argsort, topk_values, topk_indices };
    pub const Attrs = struct {
        kind: Kind,
        axis: AxisRef,
        descending: bool,
        k: u32 = 0,
        /// For topK, `.{ .new = .old }` renames the sorted axis.
        rename: TagId = .none,
    };
    pub const weight = 4;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .class = .numeric, .min_rank = 1 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        const d = sh.dim(ax);
        if (d == 0) return error.Reject;
        var a: Attrs = .{ .kind = kind, .axis = .random(g, sh, ax), .descending = g.chance(0.5) };
        var out = sh;
        switch (kind) {
            .values => {},
            .indices, .argsort => out = sh.withDtype(.i32),
            .topk_values, .topk_indices => {
                a.k = g.intIn(u32, 1, @intCast(d));
                // topK spells the axis as an integer or as `.{ .new = .old }`.
                a.axis = .{ .index = if (g.chance(0.3)) @intCast(@as(i64, @intCast(ax)) - sh.rank()) else @intCast(ax) };
                if (sh.tag(ax) != Shape.TagUnknown and g.chance(0.5)) {
                    a.axis.tag = TagId.of(sh.tag(ax));
                    a.rename = TagId.of(g.freshTag(sh));
                }
                out = sh.setDim(ax, a.k);
                if (a.rename != .none) out._tags.set(ax, a.rename.tag());
                if (kind == .topk_indices) out = out.withDtype(.i32);
            },
        }
        return .{ .op = .{ .sort = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = if (attrs.axis.tag != .none and attrs.rename == .none) attrs.axis.resolve(x.shape) else blk: {
            const i = attrs.axis.index;
            break :blk if (i < 0) @as(usize, @intCast(@as(i64, x.shape.rank()) + i)) else @as(usize, @intCast(i));
        };
        const d: usize = @intCast(x.shape.dim(ax));
        var v = try e.alloc(node.shape);
        const xs = Value.strides(x.shape);
        const vs = Value.strides(node.shape);
        const float = Value.isFloat(x.dtype());
        const want_indices = switch (attrs.kind) {
            .indices, .argsort, .topk_indices => true,
            else => false,
        };
        const keep: usize = if (attrs.k > 0) attrs.k else d;
        const order = try e.allocator.alloc(usize, d);
        var it: Value.Index = .init(x.shape.setDim(ax, 1));
        while (!it.done) {
            const base = Value.offset(xs, it.idx[0..x.shape.rank()]);
            const obase = Value.offset(vs, it.idx[0..x.shape.rank()]);
            for (order, 0..) |*o, j| o.* = j;
            const Ctx = struct { x: Value, base: usize, stride: usize, desc: bool, float: bool };
            const ctx: Ctx = .{ .x = x, .base = base, .stride = xs[ax], .desc = attrs.descending, .float = float };
            // Stable sort: equal values keep their original order.
            std.sort.block(usize, order, ctx, struct {
                fn lt(c: Ctx, l: usize, r: usize) bool {
                    const lo = c.base + l * c.stride;
                    const ro = c.base + r * c.stride;
                    if (c.float) {
                        const a = c.x.getF(lo);
                        const b = c.x.getF(ro);
                        return if (c.desc) a > b else a < b;
                    }
                    const a = c.x.getI(lo);
                    const b = c.x.getI(ro);
                    return if (c.desc) a > b else a < b;
                }
            }.lt);
            if (want_indices and attrs.k > 0) {
                // topK lowers to an unstable TopK in XLA: the order of ties isn't specified.
                for (1..@min(d, keep + 1)) |j| {
                    const lo = base + order[j - 1] * xs[ax];
                    const hi = base + order[j] * xs[ax];
                    const tie = if (float) x.getF(lo) == x.getF(hi) else x.getI(lo) == x.getI(hi);
                    if (tie) try e.reject();
                }
            }
            if (float and x.err > 0 and want_indices) {
                // Indices are only well defined if the order can't flip.
                for (1..d) |j| {
                    const a = x.getF(base + order[j - 1] * xs[ax]);
                    const b = x.getF(base + order[j] * xs[ax]);
                    if (@abs(a - b) <= 4 * x.err) try e.reject();
                }
            }
            for (0..keep) |j| {
                const off = obase + j * vs[ax];
                if (want_indices) v.setI(off, @intCast(order[j])) else v.copyElem(off, x, base + order[j] * xs[ax]);
            }
            _ = it.next();
        }
        if (!want_indices) v.err = x.err;
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        switch (attrs.kind) {
            .values, .indices => {
                const r = switch (attrs.axis.pass()) {
                    inline else => |a| x.sort(a, .{ .descending = attrs.descending }),
                };
                return if (attrs.kind == .values) r.values else r.indices;
            },
            .argsort => return switch (attrs.axis.pass()) {
                inline else => |a| x.argsort(a, .{ .descending = attrs.descending }),
            },
            .topk_values, .topk_indices => {
                const r = if (attrs.rename != .none) blk: {
                    const Body = struct {
                        x: Tensor,
                        a: Attrs,
                        pub const Result = Tensor.SortRes;
                        pub fn call(b: @This(), s: anytype) Tensor.SortRes {
                            return b.x.topK(s, b.a.k, .{ .descending = b.a.descending });
                        }
                    };
                    break :blk k.callWithTagStruct(attrs.rename, attrs.axis.tag.tag(), Body{ .x = x, .a = attrs });
                } else x.topK(attrs.axis.index, attrs.k, .{ .descending = attrs.descending });
                return if (attrs.kind == .topk_values) r.values else r.indices;
            },
        }
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (attrs.kind) {
            .values, .indices => {
                try w.print("{s}.sort(", .{ins[0]});
                try attrs.axis.print(w);
                try w.print(", .{{ .descending = {} }}).{s}", .{ attrs.descending, if (attrs.kind == .values) "values" else "indices" });
            },
            .argsort => {
                try w.print("{s}.argsort(", .{ins[0]});
                try attrs.axis.print(w);
                try w.print(", .{{ .descending = {} }})", .{attrs.descending});
            },
            .topk_values, .topk_indices => {
                if (attrs.rename != .none) {
                    try w.print("{s}.topK(.{{ .{s} = .{s} }}", .{ ins[0], attrs.rename.name(), attrs.axis.tag.name() });
                } else {
                    try w.print("{s}.topK({d}", .{ ins[0], attrs.axis.index });
                }
                try w.print(", {d}, .{{ .descending = {} }}).{s}", .{ attrs.k, attrs.descending, if (attrs.kind == .topk_values) "values" else "indices" });
            },
        }
    }
};
