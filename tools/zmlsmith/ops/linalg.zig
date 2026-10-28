//! dot (tag based), dotGeneral (explicit axes) and outer.

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
const TagId = k.TagId;
const List = k.List;

/// Contracting and batching axis pairs, as `dotGeneral` takes them.
const Pairs = List([2]i8, 4);

/// Result shape documented for dotGeneral: batching axes, then lhs free axes, then rhs
/// free axes. A batching axis takes the lhs tag, or the rhs one if the lhs is untagged.
fn dotShape(lhs: Shape, rhs: Shape, contracting: []const [2]u8, batching: []const [2]u8) Shape {
    var res: Shape = .init(.{}, lhs.dtype());
    for (batching) |b| {
        var t = lhs.tag(b[0]);
        if (t == Shape.TagUnknown) t = rhs.tag(b[1]);
        res = res.appendDim(lhs.dim(b[0]), t);
    }
    for (0..lhs.rank()) |ax| {
        if (inPairs(contracting, 0, ax) or inPairs(batching, 0, ax)) continue;
        res = res.appendDim(lhs.dim(ax), lhs.tag(ax));
    }
    for (0..rhs.rank()) |ax| {
        if (inPairs(contracting, 1, ax) or inPairs(batching, 1, ax)) continue;
        res = res.appendDim(rhs.dim(ax), rhs.tag(ax));
    }
    return res;
}

fn inPairs(pairs: []const [2]u8, side: usize, ax: usize) bool {
    for (pairs) |p| if (p[side] == ax) return true;
    return false;
}

pub fn evalDot(e: *Eval, out: Shape, lhs: Value, rhs: Value, contracting: []const [2]u8, batching: []const [2]u8) Eval.Error!Value {
    var v = try e.alloc(out);
    const ls = Value.strides(lhs.shape);
    const rs = Value.strides(rhs.shape);
    // For each output axis, the lhs and rhs axes it indexes (or none).
    var out_l: [8]?u8 = @splat(null);
    var out_r: [8]?u8 = @splat(null);
    var o: usize = 0;
    for (batching) |b| {
        out_l[o] = b[0];
        out_r[o] = b[1];
        o += 1;
    }
    for (0..lhs.shape.rank()) |ax| {
        if (inPairs(contracting, 0, ax) or inPairs(batching, 0, ax)) continue;
        out_l[o] = @intCast(ax);
        o += 1;
    }
    for (0..rhs.shape.rank()) |ax| {
        if (inPairs(contracting, 1, ax) or inPairs(batching, 1, ax)) continue;
        out_r[o] = @intCast(ax);
        o += 1;
    }
    var csh: Shape = .init(.{}, lhs.dtype());
    for (contracting) |c| csh = csh.appendDim(lhs.shape.dim(c[0]), null);
    const n_contract = csh.count();
    const dt = out.dtype();
    const float = Value.isFloat(dt);
    // dotGeneral uses `.dot_precision = .fast`: f32 inputs are rounded to TF32 off CPU.
    const tf32 = dt == .f32 and k.ops.target != .cpu;
    const prod_eps: f64 = if (tf32) std.math.ldexp(@as(f64, 1), -10) else 0;
    var err: f64 = 0;
    // Bounds per output element, from the errors of the elements it reads.
    const errs = try v.trackErrs(e.allocator);
    var it: Value.Index = .init(out);
    var i: usize = 0;
    var li: [8]i64 = @splat(0);
    var ri: [8]i64 = @splat(0);
    while (!it.done) : (i += 1) {
        for (0..out.rank()) |ax| {
            if (out_l[ax]) |l| li[l] = it.idx[ax];
            if (out_r[ax]) |r| ri[r] = it.idx[ax];
        }
        var acc_i: i128 = 0;
        var acc_f: f64 = 0;
        var abs_sum: f64 = 0;
        var in_term: f64 = 0;
        var integral = true;
        var cit: Value.Index = .init(csh);
        while (!cit.done) {
            for (contracting, 0..) |c, j| {
                li[c[0]] = cit.idx[j];
                ri[c[1]] = cit.idx[j];
            }
            const lo = Value.offset(ls, li[0..lhs.shape.rank()]);
            const ro = Value.offset(rs, ri[0..rhs.shape.rank()]);
            if (float) {
                const lv = lhs.getF(lo);
                const rv = rhs.getF(ro);
                const p = lv * rv;
                in_term += @abs(lv) * rhs.inErr(ro) + @abs(rv) * lhs.inErr(lo) + lhs.inErr(lo) * rhs.inErr(ro);
                acc_f += p;
                abs_sum += @abs(p);
                integral = integral and p == @round(p);
            } else {
                acc_i = Value.wrapInt(dt, acc_i + lhs.getI(lo) *% rhs.getI(ro));
            }
            _ = cit.next();
        }
        if (float) {
            var lane_err = in_term;
            lane_err += 2 * prod_eps * abs_sum;
            const exact = !tf32 and in_term == 0 and integral and abs_sum < std.math.ldexp(@as(f64, 1), Value.precisionBits(dt));
            if (!exact) lane_err += @as(f64, @floatFromInt(n_contract + 1)) * Value.eps(dt) * abs_sum;
            if (!k.storeF(v, i, acc_f) and exact) lane_err += k.roundErr(dt, @abs(acc_f));
            // Zero results: the sign depends on the accumulation order.
            if (acc_f == 0) lane_err = @max(lane_err, Value.any_zero_sign);
            errs[i] = lane_err;
            err = @max(err, lane_err);
        } else v.setI(i, acc_i);
        _ = it.next();
    }
    v.err = err;
    v.rounded = float;
    return v;
}

pub const dot = struct {
    pub const Attrs = struct {
        /// `dot` with contracting tags, or `dotGeneral` with explicit pairs.
        general: bool,
        contracting: List(TagId, 4) = .{},
        c_pairs: Pairs = .{},
        b_pairs: Pairs = .{},
        list: k.Spelling = .slice,
    };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 7;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const general = g.chance(0.35);
        // The tag API also takes partially tagged operands: untagged axes stay free.
        const lhs = try g.operand(.{ .class = .numeric, .min_rank = 1, .max_rank = 4, .tagged = !general and g.chance(0.7) });
        const ls = g.shapeOf(lhs);
        // Assign each lhs axis a role.
        const Role = enum { contract, batch, free };
        var roles: [8]Role = undefined;
        var n_contract: usize = 0;
        var n_tagged: usize = 0;
        for (0..ls.rank()) |ax| {
            const r = g.rng.uintLessThan(u8, 10);
            roles[ax] = if (r < 4) .contract else if (r < 6) .batch else .free;
            const untagged = ls.tag(ax) == Shape.TagUnknown;
            if (!untagged) n_tagged += 1;
            if (!general and untagged) roles[ax] = .free;
            if (roles[ax] == .contract) n_contract += 1;
        }
        if (n_contract == 0) {
            if (!general and n_tagged == 0) return error.Reject;
            while (true) {
                const ax = g.rng.uintLessThan(usize, ls.rank());
                if (general or ls.tag(ax) != Shape.TagUnknown) {
                    roles[ax] = .contract;
                    break;
                }
            }
        }
        // rhs: shared axes plus up to two free axes, in random order.
        var dims: [8]i64 = undefined;
        var tags: [8]Shape.Tag = undefined;
        var src: [8]?u8 = undefined;
        var n: usize = 0;
        for (0..ls.rank()) |ax| if (roles[ax] != .free) {
            dims[n] = ls.dim(ax);
            tags[n] = ls.tag(ax);
            src[n] = @intCast(ax);
            n += 1;
        };
        const n_free = g.intIn(usize, 0, @min(2, 4 - @min(n, 4)));
        for (0..n_free) |_| {
            dims[n] = g.intIn(i64, 1, 5);
            // A free rhs tag must not appear in lhs, or it would become a batching axis.
            var t = g.freshTag(ls);
            for (tags[0..n]) |u| if (u == t) {
                t = Shape.TagUnknown;
            };
            tags[n] = if (g.chance(if (general) 0.5 else 0.3)) Shape.TagUnknown else t;
            src[n] = null;
            n += 1;
        }
        var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        g.rng.shuffle(u8, perm[0..n]);
        var rdims: [8]i64 = undefined;
        var rtags: [8]Shape.Tag = undefined;
        var rsrc: [8]?u8 = undefined;
        for (0..n) |i| {
            rdims[i] = dims[perm[i]];
            rtags[i] = tags[perm[i]];
            rsrc[i] = src[perm[i]];
        }
        const rs = k.ir.makeShape(ls.dtype(), rdims[0..n], rtags[0..n]);
        const rhs = try g.operandLike(rs, true);

        var a: Attrs = .{ .general = general, .list = g.rng.enumValue(k.Spelling) };
        var c_pairs: [8][2]u8 = undefined;
        var b_pairs: [8][2]u8 = undefined;
        var nc: usize = 0;
        var nb: usize = 0;
        for (0..ls.rank()) |ax| {
            if (roles[ax] == .free) continue;
            const r_ax: u8 = for (0..n) |i| {
                if (rsrc[i] != null and rsrc[i].? == ax) break @intCast(i);
            } else unreachable;
            if (roles[ax] == .contract) {
                c_pairs[nc] = .{ @intCast(ax), r_ax };
                nc += 1;
                a.contracting.append(TagId.of(ls.tag(ax)));
                a.c_pairs.append(.{ spell(g, ax, ls.rank()), spell(g, r_ax, n) });
            } else {
                b_pairs[nb] = .{ @intCast(ax), r_ax };
                nb += 1;
                a.b_pairs.append(.{ spell(g, ax, ls.rank()), spell(g, r_ax, n) });
            }
        }
        const out = dotShape(ls, rs, c_pairs[0..nc], b_pairs[0..nb]);
        if (out.rank() > Shape.MAX_RANK) return error.Reject;
        return .{ .op = .{ .dot = a }, .inputs = try k.dupeIds(g, &.{ lhs, rhs }), .shape = out };
    }

    /// Axis index, sometimes negative.
    fn spell(g: *Gen, ax: usize, rank: usize) i8 {
        if (g.chance(0.25)) return @intCast(@as(i64, @intCast(ax)) - @as(i64, @intCast(rank)));
        return @intCast(ax);
    }

    fn resolve(x: i8, rank: usize) u8 {
        return @intCast(if (x < 0) @as(i64, @intCast(rank)) + x else x);
    }

    fn pairs(attrs: Attrs, lhs: Shape, rhs: Shape, c: *[8][2]u8, b: *[8][2]u8) struct { usize, usize } {
        if (attrs.general) {
            for (attrs.c_pairs.slice(), 0..) |p, i| c[i] = .{ resolve(p[0], lhs.rank()), resolve(p[1], rhs.rank()) };
            for (attrs.b_pairs.slice(), 0..) |p, i| b[i] = .{ resolve(p[0], lhs.rank()), resolve(p[1], rhs.rank()) };
            return .{ attrs.c_pairs.len, attrs.b_pairs.len };
        }
        // Documented tag semantics: listed tags contract, other shared tags batch.
        var nc: usize = 0;
        for (attrs.contracting.slice()) |t| {
            c[nc] = .{ lhs.hasTag(t.tag()).?, rhs.hasTag(t.tag()).? };
            nc += 1;
        }
        var nb: usize = 0;
        for (0..lhs.rank()) |ax| {
            if (inPairs(c[0..nc], 0, ax)) continue;
            const t = lhs.tag(ax);
            if (t == Shape.TagUnknown) continue;
            if (rhs.hasTag(t)) |r| {
                b[nb] = .{ @intCast(ax), r };
                nb += 1;
            }
        }
        return .{ nc, nb };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var c: [8][2]u8 = undefined;
        var b: [8][2]u8 = undefined;
        const nc, const nb = pairs(attrs, ins[0].shape, ins[1].shape, &c, &b);
        return evalDot(e, node.shape, ins[0], ins[1], c[0..nc], b[0..nb]);
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        if (attrs.general) return ins[0].dotGeneral(ins[1], attrs.c_pairs.slice(), attrs.b_pairs.slice());
        const tags = attrs.contracting.slice();
        if (tags.len == 1) return ins[0].dot(ins[1], tags[0].tag());
        var t: [4]Shape.Tag = undefined;
        for (tags, 0..) |x, i| t[i] = x.tag();
        return k.callWithList(Shape.Tag, attrs.list, t[0..tags.len], struct {
            l: Tensor,
            r: Tensor,
            pub const Result = Tensor;
            pub fn call(c: @This(), list: anytype) Tensor {
                return c.l.dot(c.r, list);
            }
        }{ .l = ins[0], .r = ins[1] });
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        if (attrs.general) {
            try w.print("{s}.dotGeneral({s}, &.{{ ", .{ ins[0], ins[1] });
            for (attrs.c_pairs.slice(), 0..) |p, i| try w.print("{s}.{{ {d}, {d} }}", .{ if (i > 0) ", " else "", p[0], p[1] });
            try w.writeAll(" }, &.{ ");
            for (attrs.b_pairs.slice(), 0..) |p, i| try w.print("{s}.{{ {d}, {d} }}", .{ if (i > 0) ", " else "", p[0], p[1] });
            return w.writeAll(" })");
        }
        const tags = attrs.contracting.slice();
        if (tags.len == 1) return w.print("{s}.dot({s}, .{s})", .{ ins[0], ins[1], tags[0].name() });
        try w.print("{s}.dot({s}, ", .{ ins[0], ins[1] });
        var axes_: [4]u8 = undefined;
        for (tags, 0..) |t, i| axes_[i] = k.emit_shapes[0].hasTag(t.tag()).?;
        try k.printAxisList(w, attrs.list, true, k.emit_shapes[0], axes_[0..tags.len]);
        try w.writeAll(")");
    }
};

pub const outer = struct {
    pub const Attrs = struct {};
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const a = try g.operand(.{ .class = .numeric, .min_rank = 1, .max_rank = 2 });
        const sa = g.shapeOf(a);
        const sb = g.randShape(sa.dtype(), .{ .min_rank = 1, .max_rank = 2 });
        // Documented as the concatenation of both shapes; shared tags would be batching
        // axes, which is a separate restriction, so keep tags disjoint.
        var b_shape = sb;
        for (0..sb.rank()) |ax| {
            var t = g.freshTag(sa);
            while (b_shape.hasTag(t) != null) t = g.freshTag(sa);
            b_shape._tags.set(ax, if (sb.tag(ax) == Shape.TagUnknown) Shape.TagUnknown else t);
        }
        if (sa.rank() + b_shape.rank() > Shape.MAX_RANK) return error.Reject;
        try g.avoid(.outer_untagged, sa.rank() + b_shape.rank() > 1 and (!k.gen.fullyTagged(sa) or !k.gen.fullyTagged(b_shape)));
        const b = try g.freshInput(b_shape);
        var out = sa;
        for (0..b_shape.rank()) |ax| out = out.appendDim(b_shape.dim(ax), b_shape.tag(ax));
        return .{ .op = .{ .outer = .{} }, .inputs = try k.dupeIds(g, &.{ a, b }), .shape = out };
    }

    pub fn eval(e: *Eval, _: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        return evalDot(e, node.shape, ins[0], ins[1], &.{}, &.{});
    }

    pub fn lower(_: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return ins[0].outer(ins[1]);
    }

    pub fn emit(_: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.outer({s})", .{ ins[0], ins[1] });
    }
};
