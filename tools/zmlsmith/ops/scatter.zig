//! scatterSlices and gatherSlices, interpreted from their doc comments.

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
const List = k.List;

const index_dtypes = [_]DataType{ .i32, .i32, .i64, .u32 };

fn shuffled(g: *Gen, sh: Shape) Shape {
    var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    g.rng.shuffle(u8, perm[0..sh.rank()]);
    var res: Shape = .init(.{}, sh.dtype());
    for (perm[0..sh.rank()]) |p| res = res.appendDim(sh.dim(p), sh.tag(p));
    return res;
}

// -----------------------------------------------------------------------------

pub const scatter = struct {
    pub const UpdateFn = enum { increment, override };
    pub const Attrs = struct {
        tagged: bool,
        /// For the tagged api, the self axis each index tensor points into.
        coords: List(TagId, 4),
        update_fn: UpdateFn,
        unique: bool,
        /// ScatterOpts.indices_are_sorted: the window starts are nondecreasing in index order.
        sorted: bool = false,
    };
    pub const weight = 4;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const tagged = g.chance(0.75);
        const update_fn = g.rng.enumValue(UpdateFn);
        const x = try g.operand(.{ .min_rank = 1, .max_rank = 4, .tagged = tagged, .class = if (update_fn == .increment) .numeric else .any });
        const sh = g.shapeOf(x);
        const idt = g.choose(DataType, &index_dtypes);
        // Promises of ScatterOpts; eval rejects the programs that break them.
        var a: Attrs = .{ .tagged = tagged, .coords = .{}, .update_fn = update_fn, .unique = g.chance(0.4) };

        var ish: Shape = .init(.{}, idt);
        var ush: Shape = .init(.{}, sh.dtype());
        // Window size along each self axis, 0 for batching axes.
        var window: [8]i64 = undefined;
        var is_coord: [8]bool = @splat(false);
        var in_update: [8]bool = @splat(false);
        const n_coords = g.intIn(usize, 1, @min(@as(usize, if (tagged) 2 else 3), sh.rank()));
        if (tagged) {
            var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
            g.rng.shuffle(u8, perm[0..sh.rank()]);
            for (perm[0..n_coords]) |ax| {
                is_coord[ax] = true;
                a.coords.append(TagId.of(sh.tag(ax)));
            }
            // Generate around the avoided bugs rather than reject.
            const scalar_indices = !g.cfg.avoid.contains(.scatter_scalar_indices) and g.chance(0.2);
            for (0..sh.rank()) |ax| {
                if (!is_coord[ax] and !scalar_indices and g.chance(0.3)) {
                    // Batching axis: in self, indices and updates, but not a coordinate.
                    ish = ish.appendDim(sh.dim(ax), sh.tag(ax));
                    window[ax] = 0;
                    continue;
                }
                const d = sh.dim(ax);
                if (is_coord[ax] and g.chance(0.4)) {
                    window[ax] = 1;
                } else {
                    window[ax] = g.intIn(i64, 1, d);
                    in_update[ax] = true;
                }
            }
            var n_extra = g.intIn(usize, 0, 2);
            if (ish.rank() == 0 and n_extra == 0 and g.cfg.avoid.contains(.scatter_scalar_indices)) n_extra = 1;
            if (!scalar_indices) for (0..n_extra) |_| {
                var t = g.freshTag(sh);
                while (ish.hasTag(t) != null) t = g.freshTag(sh);
                ish = ish.appendDim(g.intIn(i64, 1, 3), t);
            };
            ish = shuffled(g, ish);
            try g.avoid(.scatter_scalar_indices, ish.rank() == 0);
            ush = ish.withDtype(sh.dtype());
            for (0..sh.rank()) |ax| if (in_update[ax]) {
                ush = ush.appendDim(window[ax], sh.tag(ax));
            };
            if (!g.cfg.avoid.contains(.scatter_update_order)) ush = shuffled(g, ush);
        } else {
            // Untagged: coordinates index the leading axes, updates are indices ++ window.
            for (0..n_coords) |ax| is_coord[ax] = true;
            const n_ish = if (g.cfg.avoid.contains(.scatter_untagged_rank)) n_coords else g.intIn(usize, 0, 2);
            for (0..n_ish) |_| ish = ish.appendDim(g.intIn(i64, 1, 3), null);
            try g.avoid(.scatter_scalar_indices, ish.rank() == 0);
            try g.avoid(.scatter_untagged_rank, ish.rank() != n_coords);
            ush = ish.withDtype(sh.dtype());
            for (0..sh.rank()) |ax| {
                window[ax] = if (ax < n_coords) 1 else g.intIn(i64, 1, sh.dim(ax));
                if (ax >= n_coords) ush = ush.appendDim(window[ax], null);
            }
        }
        if (ush.rank() > Shape.MAX_RANK or ish.rank() + 1 > Shape.MAX_RANK) return error.Reject;
        for (0..sh.rank()) |ax| if (sh.dim(ax) == 0) return error.Reject;

        var ids: [6]u32 = undefined;
        ids[0] = x;
        a.sorted = n_coords == 1 and g.chance(0.4);
        var c: usize = 0;
        for (0..sh.rank()) |ax| {
            if (!is_coord[ax]) continue;
            const iv = try Value.alloc(g.allocator, ish);
            for (0..iv.count()) |i| iv.setI(i, g.intIn(i64, 0, sh.dim(ax) - window[ax]));
            if (a.sorted) {
                var vals: [512]i64 = undefined;
                if (iv.count() > vals.len) return error.Reject;
                for (0..iv.count()) |i| vals[i] = @intCast(iv.getI(i));
                std.mem.sort(i64, vals[0..iv.count()], {}, std.sort.asc(i64));
                for (0..iv.count()) |i| iv.setI(i, vals[i]);
            }
            ids[1 + c] = try g.literalInput(iv);
            c += 1;
        }
        // Tagged coordinates are passed in the order they were picked, not self order.
        if (tagged) {
            var tmp: [4]u32 = undefined;
            for (a.coords.slice(), 0..) |t, j| {
                var rank_in_self: usize = 0;
                for (0..sh.hasTag(t.tag()).?) |ax| if (is_coord[ax]) {
                    rank_in_self += 1;
                };
                tmp[j] = ids[1 + rank_in_self];
            }
            @memcpy(ids[1 .. 1 + n_coords], tmp[0..n_coords]);
        }
        ids[1 + n_coords] = try g.operandLike(ush, true);
        return .{ .op = .{ .scatter = a }, .inputs = try k.dupeIds(g, ids[0 .. 2 + n_coords]), .shape = sh };
    }

    /// Flat position in the index tensors of the index used by update element `u`.
    fn indexOffset(a: Attrs, ish: Shape, upd: Shape, u: []const i64) usize {
        var ii: [8]i64 = @splat(0);
        for (0..ish.rank()) |iax| ii[iax] = if (a.tagged) u[upd.hasTag(ish.tag(iax)).?] else u[iax];
        return Value.offset(Value.strides(ish), ii[0..ish.rank()]);
    }

    /// Position in self of every update element, per the documented pseudo code.
    fn target(a: Attrs, self: Shape, idx: []const Value, upd: Shape, u: []const i64, out: []i64) void {
        const ish = idx[0].shape;
        var ii: [8]i64 = @splat(0);
        if (a.tagged) {
            for (0..ish.rank()) |iax| ii[iax] = u[upd.hasTag(ish.tag(iax)).?];
        } else {
            for (0..ish.rank()) |iax| ii[iax] = u[iax];
        }
        const ioff = Value.offset(Value.strides(ish), ii[0..ish.rank()]);
        for (0..self.rank()) |ax| {
            const t = self.tag(ax);
            var start: i64 = 0;
            var coord_j: ?usize = null;
            if (a.tagged) {
                for (a.coords.slice(), 0..) |ct, j| if (ct.tag() == t) {
                    coord_j = j;
                };
            } else if (ax < idx.len) coord_j = ax;
            if (coord_j) |j| start = @intCast(idx[j].getI(ioff));
            // Batching and window axes are indexed by the update; batching ones have start 0.
            const offset: i64 = if (a.tagged)
                (if (upd.hasTag(t)) |uax| u[uax] else 0)
            else if (ax < idx.len) 0 else u[ish.rank() + ax - idx.len];
            out[ax] = start + offset;
        }
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const n_idx = ins.len - 2;
        const idx = ins[1 .. 1 + n_idx];
        const upd = ins[ins.len - 1];
        var v = try x.dupe(e.allocator);
        const xs = Value.strides(x.shape);
        const hits = try e.allocator.alloc(u32, x.count());
        @memset(hits, 0);
        // Window start of each index position: the smallest offset it updates.
        const starts = try e.allocator.alloc(usize, @max(idx[0].count(), 1));
        @memset(starts, std.math.maxInt(usize));
        var it: Value.Index = .init(upd.shape);
        var i: usize = 0;
        var pos: [8]i64 = undefined;
        const float = Value.isFloat(x.dtype());
        // Per element bounds: untouched elements keep the input's, overridden ones the
        // update's, incremented ones add up their operands' and their own rounding.
        const errs: []f64 = if (float) try v.trackErrs(e.allocator) else &.{};
        const mag = try e.allocator.alloc(f64, if (float) x.count() else 0);
        const inexact = try e.allocator.alloc(bool, if (float) x.count() else 0);
        if (float) for (errs, mag, inexact, 0..) |*er, *m, *ie, o| {
            er.* = x.inErr(o);
            m.* = @abs(x.getF(o));
            ie.* = false;
        };
        while (!it.done) : (i += 1) {
            target(a, x.shape, idx, upd.shape, it.idx[0..upd.shape.rank()], pos[0..x.shape.rank()]);
            const o = Value.offset(xs, pos[0..x.shape.rank()]);
            hits[o] += 1;
            const io = indexOffset(a, idx[0].shape, upd.shape, it.idx[0..upd.shape.rank()]);
            starts[io] = @min(starts[io], o);
            switch (a.update_fn) {
                .override => {
                    v.copyElem(o, upd, i);
                    if (float) errs[o] = upd.inErr(i);
                },
                .increment => if (float) {
                    const r = v.getF(o) + upd.getF(i);
                    errs[o] += upd.inErr(i);
                    mag[o] += @abs(upd.getF(i));
                    v.setF(o, r);
                    inexact[o] = inexact[o] or r != @round(r) or v.getF(o) != r;
                } else v.setI(o, v.getI(o) + upd.getI(i)),
            }
            _ = it.next();
        }
        var max_hits: u32 = 0;
        for (hits) |h| max_hits = @max(max_hits, h);
        // Overlapping overrides depend on the execution order.
        if (a.update_fn == .override and max_hits > 1) try e.reject();
        if (a.unique and max_hits > 1) try e.reject();
        if (a.sorted) for (starts[0 .. starts.len - 1], starts[1..]) |s0, s1| if (s0 > s1) try e.reject();
        if (float) {
            const dt = x.dtype();
            if (a.update_fn == .increment) for (errs, mag, inexact, hits, 0..) |*er, m, ie, h, o| {
                if (h == 0) continue;
                // Integral sums below 2^precision are exact in any order.
                if (er.* > 0 or ie or m >= std.math.ldexp(@as(f64, 1), Value.precisionBits(dt)))
                    er.* += @as(f64, @floatFromInt(h + 1)) * Value.eps(dt) * m;
                if (v.getF(o) == 0) er.* = @max(er.*, Value.any_zero_sign);
            };
            v.rounded = x.rounded or upd.rounded;
            v.finishErrs();
        }
        _ = node;
        return v;
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const idx = ins[1 .. ins.len - 1];
        const upd = ins[ins.len - 1];
        const opts: Tensor.ScatterOpts = .{
            .update_fn = if (a.update_fn == .increment) Tensor.ScatterOpts.increment else Tensor.ScatterOpts.override,
            .indices_are_unique = a.unique,
            .indices_are_sorted = a.sorted,
        };
        if (!a.tagged) return x.scatterSlices(idx, upd, opts);
        const Body = struct {
            x: Tensor,
            u: Tensor,
            o: Tensor.ScatterOpts,
            pub const Result = Tensor;
            pub fn call(b: @This(), s: anytype) Tensor {
                return b.x.scatterSlices(s, b.u, b.o);
            }
        };
        const c = a.coords.slice();
        return switch (c.len) {
            1 => k.callWithTagStruct(c[0], idx[0], Body{ .x = x, .u = upd, .o = opts }),
            else => k.callWithTagStruct2(c[0], idx[0], c[1], idx[1], Body{ .x = x, .u = upd, .o = opts }),
        };
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const n = ins.len - 2;
        try w.print("{s}.scatterSlices(", .{ins[0]});
        if (a.tagged) {
            try w.writeAll(".{ ");
            for (a.coords.slice(), 0..) |t, j| try w.print("{s}.{s} = {s}", .{ if (j > 0) ", " else "", t.name(), ins[1 + j] });
            try w.writeAll(" }");
        } else {
            try w.writeAll("@as([]const zml.Tensor, &.{ ");
            try k.printOperands(w, ins[1 .. 1 + n]);
            try w.writeAll(" })");
        }
        try w.print(", {s}, .{{ .update_fn = zml.Tensor.ScatterOpts.{s}, .indices_are_unique = {}, .indices_are_sorted = {} }})", .{ ins[ins.len - 1], @tagName(a.update_fn), a.unique, a.sorted });
    }
};

// -----------------------------------------------------------------------------

pub const gather_slices = struct {
    pub const Attrs = struct {
        tagged: bool,
        /// Slice size per self axis; 0 means the axis isn't sliced (tagged api).
        slice: List(i64, 8),
        /// The coordinate axis of the indices is tagged `.coord` instead of being last.
        coord_tag: bool,
    };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const tagged = g.chance(0.7);
        const x = try g.operand(.{ .min_rank = 1, .max_rank = 4, .tagged = tagged });
        const sh = g.shapeOf(x);
        for (0..sh.rank()) |ax| if (sh.dim(ax) == 0) return error.Reject;
        // `.coord` names the coordinate axis of the indices; an operand axis with that tag is
        // ambiguous.
        if (sh.hasTag(Shape.toTag(.coord)) != null) return error.Reject;
        var a: Attrs = .{ .tagged = tagged, .slice = .{}, .coord_tag = g.chance(0.5) };
        var ish: Shape = .init(.{}, g.choose(DataType, &index_dtypes));
        var batching: [8]bool = @splat(false);
        var n_slice: usize = 0;
        for (0..sh.rank()) |ax| {
            if (!tagged or g.chance(0.6)) {
                a.slice.append(g.intIn(i64, 1, sh.dim(ax)));
                n_slice += 1;
            } else {
                a.slice.append(0);
                if (g.chance(0.5)) {
                    batching[ax] = true;
                    ish = ish.appendDim(sh.dim(ax), sh.tag(ax));
                }
            }
        }
        if (n_slice == 0) return error.Reject;
        for (0..g.intIn(usize, 0, 2)) |_| {
            var t: ?Shape.Tag = null;
            if (tagged) {
                var tt = g.freshTag(sh);
                while (ish.hasTag(tt) != null) tt = g.freshTag(sh);
                t = tt;
            }
            ish = ish.appendDim(g.intIn(i64, 1, 3), t);
        }
        ish = shuffled(g, ish);
        // Coordinates go last, or anywhere when tagged .coord.
        const coord_pos = if (a.coord_tag) g.rng.uintLessThan(usize, ish.rank() + 1) else ish.rank();
        const full_ish = ish.insertTag(coord_pos, @intCast(n_slice), if (a.coord_tag) Shape.toTag(.coord) else Shape.TagUnknown);
        if (full_ish.rank() + sh.rank() > Shape.MAX_RANK + 1) return error.Reject;
        if (!a.coord_tag and !tagged and full_ish.hasTag(Shape.toTag(.coord)) != null) return error.Reject;

        // Result: indices axes without the coordinate axis, then self axes that aren't
        // batching, with the slice size on sliced axes.
        var out = ish.withDtype(sh.dtype());
        for (0..sh.rank()) |ax| {
            if (batching[ax]) continue;
            const s = a.slice.slice()[ax];
            out = out.appendDim(if (s > 0) s else sh.dim(ax), sh.tag(ax));
        }
        if (out.rank() > Shape.MAX_RANK) return error.Reject;

        const iv = try Value.alloc(g.allocator, full_ish);
        var it: Value.Index = .init(full_ish);
        var i: usize = 0;
        while (!it.done) : (i += 1) {
            // The j-th coordinate indexes the j-th sliced axis of self.
            const j: usize = @intCast(it.idx[coord_pos]);
            var seen: usize = 0;
            for (0..sh.rank()) |ax| {
                const s = a.slice.slice()[ax];
                if (s == 0) continue;
                if (seen == j) iv.setI(i, g.intIn(i64, 0, sh.dim(ax) - s));
                seen += 1;
            }
            _ = it.next();
        }
        const idx = try g.literalInput(iv);
        return .{ .op = .{ .gather_slices = a }, .inputs = try k.dupeIds(g, &.{ x, idx }), .shape = out };
    }

    fn sliceShape(a: Attrs, sh: Shape) Shape {
        var res: Shape = .init(.{}, .i32);
        for (a.slice.slice(), 0..) |s, ax| {
            if (a.tagged) {
                if (s > 0) res = res.appendDim(s, sh.tag(ax));
            } else res = res.appendDim(s, null);
        }
        return res;
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const iv = ins[1];
        const ish = iv.shape;
        const coord_ax: usize = if (a.coord_tag) ish.hasTag(Shape.toTag(.coord)).? else ish.rank() - 1;
        var v = try e.alloc(node.shape);
        const xs = Value.strides(x.shape);
        const is = Value.strides(ish);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        const n_idx_axes = ish.rank() - 1;
        while (!it.done) : (i += 1) {
            const o = it.idx[0..node.shape.rank()];
            var ii: [8]i64 = undefined;
            var p: usize = 0;
            for (0..ish.rank()) |iax| {
                if (iax == coord_ax) continue;
                ii[iax] = o[p];
                p += 1;
            }
            var xi: [8]i64 = undefined;
            var j: usize = 0;
            var q: usize = n_idx_axes;
            for (0..x.shape.rank()) |ax| {
                const s = a.slice.slice()[ax];
                const t = x.shape.tag(ax);
                const batch = a.tagged and s == 0 and t != Shape.TagUnknown and ish.hasTag(t) != null;
                if (batch) {
                    xi[ax] = ii[ish.hasTag(t).?];
                    continue;
                }
                if (s > 0) {
                    ii[coord_ax] = @intCast(j);
                    const start: i64 = @intCast(iv.getI(Value.offset(is, ii[0..ish.rank()])));
                    xi[ax] = start + o[q];
                    j += 1;
                } else xi[ax] = o[q];
                q += 1;
            }
            v.copyElem(i, x, Value.offset(xs, xi[0..x.shape.rank()]));
            _ = it.next();
        }
        v.err = x.err;
        return v;
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return ins[0].gatherSlices(sliceShape(a, ins[0].shape()), ins[1], .{});
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.gatherSlices(", .{ins[0]});
        try k.printShapeLiteral(w, sliceShape(a, k.emit_shapes[0]));
        try w.print(", {s}, .{{}})", .{ins[1]});
    }
};
