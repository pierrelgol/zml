//! Data movement ops: transpose, reshape and axis manipulation, broadcasting, static
//! slicing, concatenation, reverse, pad and repeat.

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
const AxisRef = k.AxisRef;
const List = k.List;

fn permuted(sh: Shape, perm: []const u8) Shape {
    var res = sh;
    for (perm, 0..) |p, i| {
        res._dims.set(i, sh.dim(p));
        res._tags.set(i, sh.tag(p));
    }
    return res;
}

fn printTags(w: *Writer, sh: Shape, axes_: []const u8) Writer.Error!void {
    try w.writeAll(".{ ");
    for (axes_, 0..) |ax, i| {
        if (i > 0) try w.writeAll(", ");
        try k.printAxis(w, sh, ax);
    }
    try w.writeAll(" }");
}

fn printInts(w: *Writer, comptime T: type, xs: []const T) Writer.Error!void {
    try w.print("&[_]{s}{{ ", .{@typeName(T)});
    for (xs, 0..) |x, i| {
        if (i > 0) try w.writeAll(", ");
        try w.print("{d}", .{x});
    }
    try w.writeAll(" }");
}

// -----------------------------------------------------------------------------

pub const transpose = struct {
    pub const Spelling = enum { ints, tags, reverse_default, swap };
    pub const Attrs = struct { perm: List(u8, 8), spelling: Spelling, list: k.Spelling = .slice };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 7;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 1, .allow_storage = true });
        const sh = g.shapeOf(x);
        var spelling = g.rng.enumValue(Spelling);
        if (spelling == .tags and !k.gen.fullyTagged(sh)) spelling = .ints;
        var perm: List(u8, 8) = .{};
        for (0..sh.rank()) |i| perm.append(@intCast(i));
        switch (spelling) {
            .ints, .tags => g.rng.shuffle(u8, perm.mut()),
            .reverse_default => std.mem.reverse(u8, perm.mut()),
            .swap => {
                const a = g.rng.uintLessThan(usize, sh.rank());
                const b = g.rng.uintLessThan(usize, sh.rank());
                std.mem.swap(u8, &perm.mut()[a], &perm.mut()[b]);
            },
        }
        return .{ .op = .{ .transpose = .{ .perm = perm, .spelling = spelling, .list = g.rng.enumValue(k.Spelling) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = permuted(sh, perm.slice()) };
    }

    fn map(perm: []const u8, out: []const i64, in: []i64) bool {
        for (perm, 0..) |p, i| in[p] = out[i];
        return true;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        return k.remap(e, ins[0], node.shape, attrs.perm.slice(), map, null);
    }

    /// For `swap`, the two swapped axes.
    fn swapped(perm: []const u8) [2]u8 {
        var res: [2]u8 = .{ 0, 0 };
        var n: usize = 0;
        for (perm, 0..) |p, i| if (p != i) {
            if (n < 2) res[n] = @intCast(i);
            n += 1;
        };
        return res;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const perm = attrs.perm.slice();
        switch (attrs.spelling) {
            .ints => {
                var p: [8]i64 = undefined;
                for (perm, 0..) |a, i| p[i] = a;
                return k.callWithList(i64, attrs.list, p[0..perm.len], Call{ .x = x });
            },
            .tags => {
                var t: [8]Shape.Tag = undefined;
                for (perm, 0..) |a, i| t[i] = x.shape().tag(a);
                return k.callWithList(Shape.Tag, attrs.list, t[0..perm.len], Call{ .x = x });
            },
            .reverse_default => return x.transpose(.{}),
            .swap => {
                const ab = swapped(perm);
                return x.swapAxes(@as(i8, @intCast(ab[0])), @as(i8, @intCast(ab[1])));
            },
        }
    }

    const Call = struct {
        x: Tensor,
        pub const Result = Tensor;
        pub fn call(c: @This(), list: anytype) Tensor {
            return c.x.transpose(list);
        }
    };

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const perm = attrs.perm.slice();
        const in_shape = permutedInverse(node.shape, perm);
        switch (attrs.spelling) {
            .ints, .tags => {
                try w.print("{s}.transpose(", .{ins[0]});
                try k.printAxisList(w, attrs.list, attrs.spelling == .tags, in_shape, perm);
                try w.writeAll(")");
            },
            .reverse_default => try w.print("{s}.transpose(.{{}})", .{ins[0]}),
            .swap => {
                const ab = swapped(perm);
                try w.print("{s}.swapAxes({d}, {d})", .{ ins[0], ab[0], ab[1] });
            },
        }
    }

    fn permutedInverse(out: Shape, perm: []const u8) Shape {
        var res = out;
        for (perm, 0..) |p, i| {
            res._dims.set(p, out.dim(i));
            res._tags.set(p, out.tag(i));
        }
        return res;
    }
};

// -----------------------------------------------------------------------------

pub const reshape = struct {
    /// `auto` puts -1 in a list of dims, `auto_literal` the `.auto` enum literal in a tuple.
    pub const Spelling = enum { shape, dims, auto, flatten, as_scalar, auto_literal };
    pub const Attrs = struct { spelling: Spelling, auto_axis: u8 = 0, list: k.Spelling = .slice };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 6;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .allow_storage = true });
        const sh = g.shapeOf(x);
        const n = sh.count();
        var spelling = g.rng.enumValue(Spelling);
        if (spelling == .as_scalar and n != 1) spelling = .shape;
        switch (spelling) {
            .flatten => return .{ .op = .{ .reshape = .{ .spelling = .flatten } }, .inputs = try k.dupeIds(g, &.{x}), .shape = Shape.init(.{@as(i64, @intCast(n))}, sh.dtype()) },
            .as_scalar => return .{ .op = .{ .reshape = .{ .spelling = .as_scalar } }, .inputs = try k.dupeIds(g, &.{x}), .shape = Shape.init(.{}, sh.dtype()) },
            else => {},
        }
        // A random factorization of the element count.
        var dims: [Shape.MAX_RANK]i64 = undefined;
        var rank: usize = 0;
        var rest: i64 = @intCast(n);
        const target_rank = g.intIn(usize, 1, g.cfg.max_rank);
        while (rank + 1 < target_rank and rest > 1) {
            var divs: [16]i64 = undefined;
            var nd: usize = 0;
            var d: i64 = 1;
            while (d <= rest and nd < divs.len) : (d += 1) {
                if (@mod(rest, d) == 0) {
                    divs[nd] = d;
                    nd += 1;
                }
            }
            const f = divs[g.rng.uintLessThan(usize, nd)];
            dims[rank] = f;
            rank += 1;
            rest = @divExact(rest, f);
        }
        dims[rank] = rest;
        rank += 1;
        if (n == 0) return error.Reject;
        var tags: [Shape.MAX_RANK]Shape.Tag = @splat(Shape.TagUnknown);
        if (spelling == .shape and g.chance(0.8)) g.randTags(tags[0..rank]);
        const out = k.ir.makeShape(sh.dtype(), dims[0..rank], tags[0..rank]);
        const auto_axis: u8 = @intCast(g.rng.uintLessThan(usize, rank));
        return .{ .op = .{ .reshape = .{ .spelling = spelling, .auto_axis = auto_axis, .list = g.rng.enumValue(k.Spelling) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, _: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        @memcpy(v.bytes, ins[0].bytes);
        v.err = ins[0].err;
        return v;
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        switch (attrs.spelling) {
            .shape => return x.reshape(node.shape),
            .dims, .auto => {
                var d: [8]i64 = undefined;
                @memcpy(d[0..node.shape.rank()], node.shape.dims());
                if (attrs.spelling == .auto) d[attrs.auto_axis] = -1;
                return k.callWithList(i64, attrs.list, d[0..node.shape.rank()], struct {
                    x: Tensor,
                    pub const Result = Tensor;
                    pub fn call(c: @This(), list: anytype) Tensor {
                        return c.x.reshape(list);
                    }
                }{ .x = x });
            },
            .flatten => return x.flatten(),
            .as_scalar => return x.asScalar(),
            .auto_literal => {
                const d = node.shape.dims();
                inline for (1..Shape.MAX_RANK + 1) |r| {
                    if (r == d.len) {
                        inline for (0..r) |p| {
                            if (p == attrs.auto_axis) {
                                var before: @Tuple(&([_]type{i64} ** p)) = undefined;
                                inline for (0..p) |i| before[i] = d[i];
                                var after: @Tuple(&([_]type{i64} ** (r - p - 1))) = undefined;
                                inline for (0..r - p - 1) |i| after[i] = d[p + 1 + i];
                                return x.reshape(before ++ .{.auto} ++ after);
                            }
                        }
                    }
                }
                unreachable;
            },
        }
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (attrs.spelling) {
            .shape => {
                try w.print("{s}.reshape(", .{ins[0]});
                try k.printShapeLiteral(w, node.shape);
                try w.writeAll(")");
            },
            .dims, .auto => {
                try w.print("{s}.reshape(", .{ins[0]});
                try w.writeAll(if (attrs.list == .slice) "@as([]const i64, &.{ " else ".{ ");
                for (node.shape.dims(), 0..) |d, i| {
                    if (i > 0) try w.writeAll(", ");
                    const v = if (attrs.spelling == .auto and i == attrs.auto_axis) -1 else d;
                    if (attrs.list == .tuple) try w.print("@as(i64, {d})", .{v}) else try w.print("{d}", .{v});
                }
                try w.writeAll(if (attrs.list == .slice) " }))" else " })");
            },
            .flatten => try w.print("{s}.flatten()", .{ins[0]}),
            .as_scalar => try w.print("{s}.asScalar()", .{ins[0]}),
            .auto_literal => {
                try w.print("{s}.reshape(.{{ ", .{ins[0]});
                for (node.shape.dims(), 0..) |d, i| {
                    if (i > 0) try w.writeAll(", ");
                    if (i == attrs.auto_axis) try w.writeAll(".auto") else try w.print("@as(i64, {d})", .{d});
                }
                try w.writeAll(" })");
            },
        }
    }
};

// -----------------------------------------------------------------------------
// Axis manipulation: insertAxes, appendAxes, squeeze, splitAxis, unflatten, merge, rename.

pub const axes = struct {
    pub const Kind = enum { insert, append, squeeze, split, unflatten, merge, rename, rename_many, with_tags };
    pub const Attrs = struct {
        kind: Kind,
        axis: AxisRef = .{ .index = 0 },
        /// insert at `.last` instead of `axis`.
        last: bool = false,
        tags: List(TagId, 4) = .{},
        dims: List(i64, 4) = .{},
        /// Slice or tuple for merge and withTags; Shape or struct literal for splitAxis.
        list: k.Spelling = .slice,
    };
    pub const weight = 8;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        var a: Attrs = .{ .kind = kind, .list = g.rng.enumValue(k.Spelling) };
        var x: u32 = undefined;
        var out: Shape = undefined;
        switch (kind) {
            .insert, .append => {
                x = try g.operand(.{ .max_rank = Shape.MAX_RANK - 3 });
                const sh = g.shapeOf(x);
                const n = g.intIn(usize, 1, 2);
                var ax: usize = sh.rank();
                if (kind == .insert) {
                    a.last = g.chance(0.2);
                    if (!a.last) {
                        if (sh.rank() == 0) return error.Reject;
                        ax = g.rng.uintLessThan(usize, sh.rank());
                        a.axis = .random(g, sh, ax);
                    }
                }
                out = sh;
                for (0..n) |j| {
                    const t = g.freshTag(out);
                    a.tags.append(TagId.of(t));
                    out = out.insertTag(ax + j, 1, t);
                }
            },
            .squeeze => {
                x = g.pick(.{ .min_rank = 1 }) orelse return error.Reject;
                const sh = g.shapeOf(x);
                var ones: [8]usize = undefined;
                var n: usize = 0;
                for (0..sh.rank()) |ax| if (sh.dim(ax) == 1) {
                    ones[n] = ax;
                    n += 1;
                };
                if (n == 0) return error.Reject;
                const ax = ones[g.rng.uintLessThan(usize, n)];
                a.axis = .random(g, sh, ax);
                out = sh.remove(ax);
            },
            .split, .unflatten => {
                x = try g.operand(.{ .min_rank = 1, .max_rank = Shape.MAX_RANK - 2 });
                const sh = g.shapeOf(x);
                const ax = g.rng.uintLessThan(usize, sh.rank());
                const d = sh.dim(ax);
                if (d < 1) return error.Reject;
                var f: i64 = 1;
                var dd: i64 = 1;
                while (dd <= d) : (dd += 1) if (@mod(d, dd) == 0 and g.chance(0.5)) {
                    f = dd;
                };
                if (kind == .unflatten) {
                    // Documented as (d0, axis_m, d2) -> (d0, n, m/n, d2), torch style: the new
                    // axis is untagged and the split axis keeps its tag.
                    const negative = g.chance(0.3);
                    try g.avoid(.unflatten_negative_axis, negative);
                    a.axis = .{ .index = if (negative) @intCast(@as(i64, @intCast(ax)) - sh.rank()) else @intCast(ax) };
                    a.dims.append(f);
                    out = sh.setDim(ax, f).insertTag(ax + 1, @divExact(d, f), Shape.TagUnknown);
                } else {
                    a.axis = .random(g, sh, ax);
                    const t1 = g.freshTag(sh);
                    var t2 = g.freshTag(sh);
                    while (t2 == t1) t2 = g.freshTag(sh);
                    a.tags.append(TagId.of(t1));
                    a.tags.append(TagId.of(t2));
                    a.dims.append(f);
                    a.dims.append(@divExact(d, f));
                    out = sh.remove(ax).insertTag(ax, @divExact(d, f), t2).insertTag(ax, f, t1);
                }
            },
            .merge => {
                x = try g.operand(.{ .min_rank = 2, .tagged = true });
                const sh = g.shapeOf(x);
                const len = g.intIn(usize, 2, sh.rank());
                const start = g.rng.uintLessThan(usize, sh.rank() - len + 1);
                // The new tag may reuse one of the merged tags.
                const t = if (g.chance(0.3)) sh.tag(start) else g.freshTag(sh);
                a.tags.append(TagId.of(t));
                a.axis = .{ .index = @intCast(start) };
                a.dims.append(@intCast(len));
                var d: i64 = 1;
                for (start..start + len) |ax| d *= sh.dim(ax);
                out = sh;
                for (0..len) |_| out = out.remove(start);
                out = out.insertTag(start, d, t);
            },
            .rename => {
                x = try g.operand(.{ .min_rank = 1 });
                const sh = g.shapeOf(x);
                const ax = g.rng.uintLessThan(usize, sh.rank());
                if (sh.tag(ax) == Shape.TagUnknown) return error.Reject;
                const t = g.freshTag(sh);
                a.axis = .{ .index = @intCast(ax), .tag = TagId.of(sh.tag(ax)) };
                a.last = g.chance(0.5);
                a.tags.append(TagId.of(t));
                out = sh;
                out._tags.set(ax, t);
            },
            .rename_many => {
                // rename(.{ .old1 = new1, .old2 = new2 }).
                x = try g.operand(.{ .min_rank = 2, .tagged = true });
                const sh = g.shapeOf(x);
                var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
                g.rng.shuffle(u8, perm[0..sh.rank()]);
                out = sh;
                for (perm[0..2]) |ax| {
                    // rename checks new tags against the original shape: no swaps.
                    var t = g.freshTag(out);
                    while (out.hasTag(t) != null or sh.hasTag(t) != null) t = g.freshTag(out);
                    a.tags.append(TagId.of(sh.tag(ax)));
                    a.tags.append(TagId.of(t));
                    out._tags.set(ax, t);
                }
            },
            .with_tags => {
                x = try g.operand(.{ .min_rank = 1, .max_rank = 4 });
                const sh = g.shapeOf(x);
                var tags: [8]Shape.Tag = undefined;
                g.randTags(tags[0..sh.rank()]);
                out = sh;
                for (0..sh.rank()) |ax| {
                    a.tags.append(TagId.of(tags[ax]));
                    out._tags.set(ax, tags[ax]);
                }
            },
        }
        return .{ .op = .{ .axes = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, _: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        // All of them are reshapes of row-major data.
        var v = try e.alloc(node.shape);
        @memcpy(v.bytes, ins[0].bytes);
        v.err = ins[0].err;
        return v;
    }

    fn tagsOf(l: List(TagId, 4), out: *[4]Shape.Tag) []const Shape.Tag {
        for (l.slice(), 0..) |t, i| out[i] = t.tag();
        return out[0..l.len];
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        var buf: [4]Shape.Tag = undefined;
        switch (attrs.kind) {
            .insert, .append => {
                // insertAxes and appendAxes take a tuple of tag literals.
                const tids = attrs.tags.slice();
                const Body = struct {
                    x: Tensor,
                    attrs: Attrs,
                    pub const Result = Tensor;
                    pub fn call(b: @This(), comptime lit: anytype) Tensor {
                        if (b.attrs.kind == .append) return b.x.appendAxes(.{lit});
                        if (b.attrs.last) return b.x.insertAxes(.last, .{lit});
                        return switch (b.attrs.axis.pass()) {
                            inline else => |ax| b.x.insertAxes(ax, .{lit}),
                        };
                    }
                };
                const Body2 = struct {
                    x: Tensor,
                    attrs: Attrs,
                    second: TagId,
                    pub const Result = Tensor;
                    pub fn call(b: @This(), comptime lit: anytype) Tensor {
                        return k.callWithTagLiteral(b.second, struct {
                            x: Tensor,
                            attrs: Attrs,
                            pub const Result = Tensor;
                            pub fn call(c: @This(), comptime lit2: anytype) Tensor {
                                if (c.attrs.kind == .append) return c.x.appendAxes(.{ lit, lit2 });
                                if (c.attrs.last) return c.x.insertAxes(.last, .{ lit, lit2 });
                                return switch (c.attrs.axis.pass()) {
                                    inline else => |ax| c.x.insertAxes(ax, .{ lit, lit2 }),
                                };
                            }
                        }{ .x = b.x, .attrs = b.attrs });
                    }
                };
                if (tids.len == 1) return k.callWithTagLiteral(tids[0], Body{ .x = x, .attrs = attrs });
                return k.callWithTagLiteral(tids[0], Body2{ .x = x, .attrs = attrs, .second = tids[1] });
            },
            .squeeze => return switch (attrs.axis.pass()) {
                inline else => |ax| x.squeeze(ax),
            },
            .split => {
                const tags = tagsOf(attrs.tags, &buf);
                const d = attrs.dims.slice();
                if (attrs.list == .tuple) {
                    // splitAxis(ax, .{ .t1 = d1, .t2 = d2 })
                    return k.callWithTagStruct2(attrs.tags.slice()[0], d[0], attrs.tags.slice()[1], d[1], struct {
                        x: Tensor,
                        ax: AxisRef,
                        pub const Result = Tensor;
                        pub fn call(c: @This(), s: anytype) Tensor {
                            return switch (c.ax.pass()) {
                                inline else => |ax| c.x.splitAxis(ax, s),
                            };
                        }
                    }{ .x = x, .ax = attrs.axis });
                }
                const split = k.ir.makeShape(x.dtype(), d, tags);
                return switch (attrs.axis.pass()) {
                    inline else => |ax| x.splitAxis(ax, split),
                };
            },
            .unflatten => return x.unflatten(attrs.axis.index, attrs.dims.slice()[0]),
            .merge => {
                const start: usize = @intCast(attrs.axis.index);
                const len: usize = @intCast(attrs.dims.slice()[0]);
                var merged: [8]Shape.Tag = undefined;
                for (0..len) |i| merged[i] = x.shape().tag(start + i);
                const Body = struct {
                    x: Tensor,
                    pub const Result = Tensor;
                    pub fn call(b: @This(), s: anytype) Tensor {
                        return b.x.merge(s);
                    }
                };
                // .{ .new = <slice or tuple of the merged tags> }
                return k.callWithList(Shape.Tag, attrs.list, merged[0..len], struct {
                    x: Tensor,
                    new: TagId,
                    pub const Result = Tensor;
                    pub fn call(c: @This(), list: anytype) Tensor {
                        return k.callWithTagStruct(c.new, list, Body{ .x = c.x });
                    }
                }{ .x = x, .new = attrs.tags.slice()[0] });
            },
            .rename => {
                const new = attrs.tags.slice()[0];
                if (!attrs.last) return x.renameTag(attrs.axis.tag.tag(), new.tag());
                // renameAxis takes the new name as a literal.
                return k.callWithTagLiteral(new, struct {
                    x: Tensor,
                    ax: i8,
                    pub const Result = Tensor;
                    pub fn call(c: @This(), comptime lit: anytype) Tensor {
                        return c.x.renameAxis(c.ax, lit);
                    }
                }{ .x = x, .ax = attrs.axis.index });
            },
            .rename_many => {
                const t = attrs.tags.slice();
                return k.callWithTagStruct2(t[0], t[1].tag(), t[2], t[3].tag(), struct {
                    x: Tensor,
                    pub const Result = Tensor;
                    pub fn call(c: @This(), s: anytype) Tensor {
                        return c.x.rename(s);
                    }
                }{ .x = x });
            },
            .with_tags => {
                const tags = tagsOf(attrs.tags, &buf);
                _ = node;
                return k.callWithList(Shape.Tag, attrs.list, tags, struct {
                    x: Tensor,
                    pub const Result = Tensor;
                    pub fn call(c: @This(), list: anytype) Tensor {
                        return c.x.withTags(list);
                    }
                }{ .x = x });
            },
        }
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const tags = attrs.tags.slice();
        switch (attrs.kind) {
            .insert, .append => {
                if (attrs.kind == .append) {
                    try w.print("{s}.appendAxes(.{{ ", .{ins[0]});
                } else if (attrs.last) {
                    try w.print("{s}.insertAxes(.last, .{{ ", .{ins[0]});
                } else {
                    try w.print("{s}.insertAxes(", .{ins[0]});
                    try attrs.axis.print(w);
                    try w.writeAll(", .{ ");
                }
                for (tags, 0..) |t, i| try w.print("{s}.{s}", .{ if (i > 0) ", " else "", t.name() });
                try w.writeAll(" })");
            },
            .squeeze => {
                try w.print("{s}.squeeze(", .{ins[0]});
                try attrs.axis.print(w);
                try w.writeAll(")");
            },
            .split => {
                try w.print("{s}.splitAxis(", .{ins[0]});
                try attrs.axis.print(w);
                if (attrs.list == .tuple) {
                    try w.print(", .{{ .{s} = @as(i64, {d}), .{s} = @as(i64, {d}) }})", .{ tags[0].name(), attrs.dims.slice()[0], tags[1].name(), attrs.dims.slice()[1] });
                } else {
                    try w.writeAll(", ");
                    try k.printShapeLiteral(w, k.ir.makeShape(node.shape.dtype(), attrs.dims.slice(), &.{ tags[0].tag(), tags[1].tag() }));
                    try w.writeAll(")");
                }
            },
            .unflatten => try w.print("{s}.unflatten({d}, {d})", .{ ins[0], attrs.axis.index, attrs.dims.slice()[0] }),
            .merge => {
                const start: usize = @intCast(attrs.axis.index);
                const len: usize = @intCast(attrs.dims.slice()[0]);
                // The input tags are the ones replaced by the merged axis.
                try w.print("{s}.merge(.{{ .{s} = ", .{ ins[0], tags[0].name() });
                var axes_: [8]u8 = undefined;
                for (0..len) |i| axes_[i] = @intCast(start + i);
                try k.printAxisList(w, attrs.list, true, k.emit_shapes[0], axes_[0..len]);
                try w.writeAll(" })");
            },
            .rename => if (attrs.last)
                try w.print("{s}.renameAxis({d}, .{s})", .{ ins[0], attrs.axis.index, tags[0].name() })
            else
                try w.print("{s}.renameTag(.{s}, .{s})", .{ ins[0], attrs.axis.tag.name(), tags[0].name() }),
            .rename_many => try w.print("{s}.rename(.{{ .{s} = zml.Shape.toTag(.{s}), .{s} = zml.Shape.toTag(.{s}) }})", .{ ins[0], tags[0].name(), tags[1].name(), tags[2].name(), tags[3].name() }),
            .with_tags => {
                try w.print("{s}.withTags(", .{ins[0]});
                var axes_: [8]u8 = undefined;
                for (0..tags.len) |i| axes_[i] = @intCast(i);
                try k.printAxisList(w, attrs.list, true, node.shape, axes_[0..tags.len]);
                try w.writeAll(")");
            },
        }
    }
};

// -----------------------------------------------------------------------------

pub const broadcast = struct {
    pub const Kind = enum { broad, explicit, left, right };
    pub const Attrs = struct { kind: Kind, axes: List(u8, 8) };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 6;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .max_rank = 4 });
        const sh = g.shapeOf(x);
        // Build the output by expanding size-1 axes and inserting new axes.
        // Same dims with other tags: documented to return the target tags.
        if (kind != .explicit and sh.rank() > 0 and g.chance(0.15)) {
            try g.avoid(.broad_equal_dims_tags, true);
            var tags: [8]Shape.Tag = undefined;
            g.randTags(tags[0..sh.rank()]);
            var target = sh;
            for (0..sh.rank()) |ax| target._tags.set(ax, tags[ax]);
            var map: List(u8, 8) = .{};
            for (0..sh.rank()) |ax| map.append(@intCast(ax));
            return .{ .op = .{ .broadcast = .{ .kind = kind, .axes = map } }, .inputs = try k.dupeIds(g, &.{x}), .shape = target };
        }
        const no_new_axis = (kind == .broad or kind == .explicit) and sh.rank() > 0 and g.chance(0.3);
        const extra: usize = g.intIn(usize, if (no_new_axis) 0 else 1, @min(2, Shape.MAX_RANK - 1 - sh.rank()));
        var out = sh;
        for (0..sh.rank()) |ax| if (sh.dim(ax) == 1 and g.chance(0.6)) {
            out = out.setDim(ax, g.intIn(i64, 2, 4));
        };
        var map: List(u8, 8) = .{};
        for (0..sh.rank()) |ax| map.append(@intCast(ax));
        for (0..extra) |_| {
            const pos: usize = switch (kind) {
                .left => 0,
                .right => out.rank(),
                else => g.rng.uintLessThan(usize, out.rank() + 1),
            };
            out = out.insertTag(pos, g.intIn(i64, 1, 4), if (sh.rank() == 0 or k.gen.fullyTagged(sh)) g.freshTag(out) else Shape.TagUnknown);
            for (map.mut()) |*m| if (m.* >= pos) {
                m.* += 1;
            };
        }
        if (kind == .broad and sh.rank() != 0 and sh.rank() != out.rank() and !k.gen.fullyTagged(sh)) return error.Reject;
        if (kind == .explicit and g.chance(0.2) and sh.rank() > 1) {
            // Non monotonic mapping: broadcast_in_dim then also transposes.
            g.rng.shuffle(u8, map.mut());
            var fixed = out;
            for (0..sh.rank()) |ax| {
                fixed._dims.set(map.slice()[ax], if (sh.dim(ax) == 1) out.dim(map.slice()[ax]) else sh.dim(ax));
            }
            out = fixed;
        }
        if (kind == .explicit and std.mem.eql(i64, out.dims(), sh.dims())) {
            var identity = true;
            for (map.slice(), 0..) |m, ax| identity = identity and m == ax;
            try g.avoid(.broadcast_equal_dims_perm, !identity);
        }
        return .{ .op = .{ .broadcast = .{ .kind = kind, .axes = map } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    fn mapFn(ctx: anytype, out: []const i64, in: []i64) bool {
        for (ctx.axes, 0..) |a, i| in[i] = if (ctx.in.dim(i) == 1) 0 else out[a];
        return true;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        return k.remap(e, ins[0], node.shape, .{ .axes = attrs.axes.slice(), .in = ins[0].shape }, mapFn, null);
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        return switch (attrs.kind) {
            .broad => x.broad(node.shape),
            .explicit => blk: {
                var a: [8]i64 = undefined;
                for (attrs.axes.slice(), 0..) |m, i| a[i] = m;
                break :blk x.broadcast(node.shape, a[0..attrs.axes.len]);
            },
            .left => x.broadcastLeft(node.shape),
            .right => x.broadcastRight(node.shape),
        };
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const name = switch (attrs.kind) {
            .broad => "broad",
            .explicit => "broadcast",
            .left => "broadcastLeft",
            .right => "broadcastRight",
        };
        try w.print("{s}.{s}(", .{ ins[0], name });
        try k.printShapeLiteral(w, node.shape);
        if (attrs.kind == .explicit) {
            try w.writeAll(", ");
            var a: [8]i64 = undefined;
            for (attrs.axes.slice(), 0..) |m, i| a[i] = m;
            try printInts(w, i64, a[0..attrs.axes.len]);
        }
        try w.writeAll(")");
    }
};

// -----------------------------------------------------------------------------
// Static slices.

pub const SliceSpec = struct {
    axis: AxisRef,
    /// As written by the user, possibly negative.
    start: i64,
    /// Exclusive end as written, possibly negative; `to_end` means omitted.
    end: i64,
    /// When set, `.len` is used instead of `.end`.
    len: i64,
    step: i32,
    single: bool,

    pub const to_end = std.math.maxInt(i64);

    fn toZml(s: SliceSpec) Tensor.Slice {
        if (s.single) return .single(s.start);
        var res: Tensor.Slice = .{ .start = s.start, .step = s.step };
        if (s.len != to_end) res.len = s.len else res.end = s.end;
        return res;
    }

    /// Documented semantics: negative start and end count from the end of the axis.
    fn resolve(s: SliceSpec, d: i64) struct { start: i64, len: i64 } {
        const start = if (s.start < 0) d + s.start else s.start;
        const end = if (s.single) start + 1 else if (s.len != to_end) start + s.len else if (s.end == to_end) d else if (s.end < 0) d + s.end else s.end;
        const len = std.math.divCeil(i64, end - start, s.step) catch unreachable;
        return .{ .start = start, .len = len };
    }

    fn print(s: SliceSpec, w: *Writer) Writer.Error!void {
        if (s.single) return w.print(".single({d})", .{s.start});
        try w.print(".{{ .start = {d}", .{s.start});
        if (s.len != to_end) try w.print(", .len = {d}", .{s.len}) else if (s.end != to_end) try w.print(", .end = {d}", .{s.end});
        if (s.step != 1) try w.print(", .step = {d}", .{s.step});
        try w.writeAll(" }");
    }

    fn random(g: *Gen, sh: Shape, ax: usize) SliceSpec {
        const d = sh.dim(ax);
        const start = g.intIn(i64, 0, d - 1);
        const stop = g.intIn(i64, start + 1, d);
        var s: SliceSpec = .{ .axis = .random(g, sh, ax), .start = start, .end = stop, .len = to_end, .step = 1, .single = false };
        if (g.chance(0.3)) s.start = start - d;
        switch (g.rng.uintLessThan(u8, 6)) {
            0 => s.end = to_end,
            1 => if (!g.cfg.avoid.contains(.slice_len)) {
                s.end = to_end;
                s.len = stop - start;
            },
            2 => if (stop < d) {
                s.end = stop - d;
            },
            3 => s.single = true,
            else => {},
        }
        if (!s.single and g.chance(0.3)) s.step = g.intIn(i32, 2, 3);
        return s;
    }
};

pub const slice = struct {
    pub const Attrs = struct { specs: List(SliceSpec, 3), multi: bool, list: k.Spelling = .slice };
    pub const weight = 7;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 1, .allow_storage = true });
        const sh = g.shapeOf(x);
        const multi = g.chance(0.35) and sh.rank() > 1;
        var specs: List(SliceSpec, 3) = .{};
        var axes_: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        g.rng.shuffle(u8, axes_[0..sh.rank()]);
        const n: usize = if (multi) g.intIn(usize, 2, @min(3, sh.rank())) else 1;
        var out = sh;
        var removed: [8]bool = @splat(false);
        for (0..n) |i| {
            const ax = axes_[i];
            const s = SliceSpec.random(g, sh, ax);
            const r = s.resolve(sh.dim(ax));
            out = out.setDim(ax, r.len);
            if (s.single) removed[ax] = true;
            specs.append(s);
        }
        var j: usize = sh.rank();
        while (j > 0) {
            j -= 1;
            if (removed[j]) out = out.remove(j);
        }
        return .{ .op = .{ .slice = .{ .specs = specs, .multi = multi, .list = g.rng.enumValue(k.Spelling) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    const Ctx = struct { in: Shape, starts: [8]i64, steps: [8]i64, removed: [8]bool };

    fn map(c: *const Ctx, out: []const i64, in: []i64) bool {
        var o: usize = 0;
        for (0..c.in.rank()) |ax| {
            if (c.removed[ax]) {
                in[ax] = c.starts[ax];
            } else {
                in[ax] = c.starts[ax] + out[o] * c.steps[ax];
                o += 1;
            }
        }
        return true;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const sh = ins[0].shape;
        var c: Ctx = .{ .in = sh, .starts = @splat(0), .steps = @splat(1), .removed = @splat(false) };
        for (attrs.specs.slice()) |s| {
            const ax = s.axis.resolve(sh);
            c.starts[ax] = s.resolve(sh.dim(ax)).start;
            c.steps[ax] = s.step;
            c.removed[ax] = s.single;
        }
        return k.remap(e, ins[0], node.shape, &c, map, null);
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const specs = attrs.specs.slice();
        if (!attrs.multi) {
            return switch (specs[0].axis.pass()) {
                inline else => |ax| x.slice(ax, specs[0].toZml()),
            };
        }
        var zs: [3]Tensor.Slice = undefined;
        var ax: [3]i64 = undefined;
        for (specs, 0..) |s, i| {
            zs[i] = s.toZml();
            ax[i] = @intCast(s.axis.resolve(x.shape()));
        }
        return k.callWithList(i64, attrs.list, ax[0..specs.len], struct {
            x: Tensor,
            zs: []const Tensor.Slice,
            pub const Result = Tensor;
            pub fn call(c: @This(), list: anytype) Tensor {
                return c.x.slices(list, c.zs);
            }
        }{ .x = x, .zs = zs[0..specs.len] });
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const specs = attrs.specs.slice();
        if (!attrs.multi) {
            try w.print("{s}.slice(", .{ins[0]});
            try specs[0].axis.print(w);
            try w.writeAll(", ");
            try specs[0].print(w);
            return w.writeAll(")");
        }
        try w.print("{s}.slices(", .{ins[0]});
        var axes_: [3]u8 = undefined;
        for (specs, 0..) |s, i| axes_[i] = @intCast(s.axis.resolve(k.emit_shapes[0]));
        try k.printAxisList(w, attrs.list, false, k.emit_shapes[0], axes_[0..specs.len]);
        try w.writeAll(", &.{ ");
        for (specs, 0..) |s, i| {
            if (i > 0) try w.writeAll(", ");
            try s.print(w);
        }
        try w.writeAll(" })");
    }
};

// -----------------------------------------------------------------------------

pub const concat = struct {
    pub const Attrs = struct { axis: AxisRef, stack: bool, last: bool = false, tag: TagId = .none };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 5;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const stack = g.chance(0.4);
        const x = try g.operand(.{ .min_rank = if (stack) 0 else 1, .max_rank = Shape.MAX_RANK - 2, .allow_storage = true });
        const sh = g.shapeOf(x);
        const n = g.intIn(usize, 1, 4);
        var ids: [4]u32 = undefined;
        ids[0] = x;
        if (stack) {
            // stack wants exactly the same shapes, tags included.
            for (1..n) |i| ids[i] = try g.operandLike(sh, true);
            const last = sh.rank() == 0 or g.chance(0.3);
            const ax = if (last) sh.rank() else g.rng.uintLessThan(usize, sh.rank());
            const t = g.freshTag(sh);
            const out = sh.insertTag(ax, @intCast(n), t);
            const axis: AxisRef = if (last) .{ .index = 0 } else .random(g, sh, ax);
            return .{ .op = .{ .concat = .{ .axis = axis, .stack = true, .last = last, .tag = TagId.of(t) } }, .inputs = try k.dupeIds(g, ids[0..n]), .shape = out };
        }
        const ax = g.rng.uintLessThan(usize, sh.rank());
        var total = sh.dim(ax);
        for (1..n) |i| {
            const other = sh.setDim(ax, g.intIn(i64, 1, 4));
            ids[i] = try g.operandLike(other, true);
            total += other.dim(ax);
        }
        return .{ .op = .{ .concat = .{ .axis = .random(g, sh, ax), .stack = false } }, .inputs = try k.dupeIds(g, ids[0..n]), .shape = sh.setDim(ax, total) };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const out = node.shape;
        var v = try e.alloc(out);
        const ax = if (attrs.stack) (if (attrs.last) ins[0].shape.rank() else attrs.axis.resolve(ins[0].shape)) else attrs.axis.resolve(ins[0].shape);
        var it: Value.Index = .init(out);
        var i: usize = 0;
        var in_idx: [8]i64 = undefined;
        while (!it.done) : (i += 1) {
            const idx = it.idx[0..out.rank()];
            var pos = idx[ax];
            var which: usize = 0;
            if (attrs.stack) {
                which = @intCast(pos);
                var o: usize = 0;
                for (0..out.rank()) |a| if (a != ax) {
                    in_idx[o] = idx[a];
                    o += 1;
                };
            } else {
                while (pos >= ins[which].shape.dim(ax)) {
                    pos -= ins[which].shape.dim(ax);
                    which += 1;
                }
                @memcpy(in_idx[0..out.rank()], idx);
                in_idx[ax] = pos;
            }
            const in = ins[which];
            v.copyElem(i, in, Value.offset(Value.strides(in.shape), in_idx[0..in.shape.rank()]));
            _ = it.next();
        }
        v.err = k.maxErr(ins);
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        if (!attrs.stack) {
            return switch (attrs.axis.pass()) {
                inline else => |ax| Tensor.concatenate(ins, ax),
            };
        }
        if (attrs.last) return Tensor.stack(ins, .last, attrs.tag.tag());
        return switch (attrs.axis.pass()) {
            inline else => |ax| Tensor.stack(ins, ax, attrs.tag.tag()),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("zml.Tensor.{s}(&.{{ ", .{if (attrs.stack) "stack" else "concatenate"});
        try k.printOperands(w, ins);
        try w.writeAll(" }, ");
        if (attrs.stack and attrs.last) try w.writeAll(".last") else try attrs.axis.print(w);
        if (attrs.stack) try w.print(", .{s}", .{attrs.tag.name()});
        try w.writeAll(")");
    }
};

// -----------------------------------------------------------------------------

pub const reverse = struct {
    pub const Attrs = struct { axes: List(AxisRef, 4), list: k.Spelling = .slice };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 1, .allow_storage = true });
        const sh = g.shapeOf(x);
        var a: List(AxisRef, 4) = .{};
        var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        g.rng.shuffle(u8, perm[0..sh.rank()]);
        const n = g.intIn(usize, 1, @min(4, sh.rank()));
        // Spell all axes the same way, as a slice of ints or of tags.
        const by_tag = k.gen.fullyTagged(sh) and g.chance(0.5);
        for (perm[0..n]) |ax| a.append(if (by_tag) .{ .index = @intCast(ax), .tag = TagId.of(sh.tag(ax)) } else .{ .index = @intCast(ax) });
        return .{ .op = .{ .reverse = .{ .axes = a, .list = g.rng.enumValue(k.Spelling) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = sh };
    }

    fn map(ctx: anytype, out: []const i64, in: []i64) bool {
        for (out, 0..) |o, ax| in[ax] = if (ctx.flip[ax]) ctx.sh.dim(ax) - 1 - o else o;
        return true;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var flip: [8]bool = @splat(false);
        for (attrs.axes.slice()) |a| flip[a.resolve(node.shape)] = true;
        return k.remap(e, ins[0], node.shape, .{ .sh = node.shape, .flip = flip }, map, null);
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const refs = attrs.axes.slice();
        if (refs[0].tag != .none) {
            var t: [4]Shape.Tag = undefined;
            for (refs, 0..) |r, i| t[i] = r.tag.tag();
            return k.callWithList(Shape.Tag, attrs.list, t[0..refs.len], Rev{ .x = ins[0] });
        }
        var a: [4]i64 = undefined;
        for (refs, 0..) |r, i| a[i] = r.index;
        return k.callWithList(i64, attrs.list, a[0..refs.len], Rev{ .x = ins[0] });
    }

    const Rev = struct {
        x: Tensor,
        pub const Result = Tensor;
        pub fn call(c: @This(), list: anytype) Tensor {
            return c.x.reverse(list);
        }
    };

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.reverse(", .{ins[0]});
        var axes_: [4]u8 = undefined;
        const refs = attrs.axes.slice();
        for (refs, 0..) |r, i| axes_[i] = @intCast(r.index);
        try k.printAxisList(w, attrs.list, refs[0].tag != .none, k.emit_shapes[0], axes_[0..refs.len]);
        try w.writeAll(")");
    }
};

// -----------------------------------------------------------------------------

pub const pad = struct {
    pub const Attrs = struct { axis: AxisRef, low: i64, high: i64, interior: i64, value: i64 };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        // pad takes `.{ .tag = Pad{...} }`, so the padded axis must be tagged.
        const x = try g.operand(.{ .min_rank = 1, .tagged = true });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        const d = sh.dim(ax);
        const interior = if (g.chance(0.3)) g.intIn(i64, 1, 2) else 0;
        const full = d + @max(d - 1, 0) * interior;
        // stablehlo allows negative padding, which crops.
        const low = g.intIn(i64, -@divFloor(full - 1, 2), 3);
        const high = g.intIn(i64, -@divFloor(full - 1, 2), 3);
        try g.avoid(.xla_spmd_interior_pad, g.prog.sharded and interior > 0 and (low < 0 or high < 0));
        const out_d = full + low + high;
        if (out_d <= 0) return error.Reject;
        const dt = sh.dtype();
        var value = g.intIn(i64, -3, 3);
        if (dt == .bool or (Value.isInt(dt) and !dt.isSignedInt())) value = @intCast(@abs(value) % 2);
        return .{
            .op = .{ .pad = .{ .axis = .{ .index = @intCast(ax), .tag = TagId.of(sh.tag(ax)) }, .low = low, .high = high, .interior = interior, .value = value } },
            .inputs = try k.dupeIds(g, &.{x}),
            .shape = sh.setDim(ax, out_d),
        };
    }

    fn map(ctx: anytype, out: []const i64, in: []i64) bool {
        @memcpy(in, out);
        const p = out[ctx.ax] - ctx.a.low;
        if (p < 0) return false;
        const stride = ctx.a.interior + 1;
        if (@mod(p, stride) != 0) return false;
        const src = @divExact(p, stride);
        if (src >= ctx.d) return false;
        in[ctx.ax] = src;
        return true;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        var fill = try e.alloc(.init(.{}, x.dtype()));
        if (Value.isFloat(x.dtype())) fill.setF(0, @floatFromInt(attrs.value)) else fill.setI(0, attrs.value);
        _ = try fill.trackErrs(e.allocator);
        const ax = attrs.axis.resolve(x.shape);
        return k.remap(e, x, node.shape, .{ .ax = ax, .a = attrs, .d = x.shape.dim(ax) }, map, fill);
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const p: Tensor.Pad = .{ .low = attrs.low, .high = attrs.high, .interior = attrs.interior };
        const Body = struct {
            x: Tensor,
            v: i64,
            bool_: bool,
            pub const Result = Tensor;
            pub fn call(b: @This(), s: anytype) Tensor {
                if (b.bool_) return b.x.pad(b.v != 0, s);
                return b.x.pad(b.v, s);
            }
        };
        return k.callWithTagStruct(attrs.axis.tag, p, Body{ .x = ins[0], .v = attrs.value, .bool_ = node.shape.dtype() == .bool });
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        if (node.shape.dtype() == .bool) {
            try w.print("{s}.pad({}, ", .{ ins[0], attrs.value != 0 });
        } else {
            try w.print("{s}.pad({d}, ", .{ ins[0], attrs.value });
        }
        try w.print(".{{ .{s} = zml.Tensor.Pad{{ .low = {d}, .high = {d}, .interior = {d} }} }})", .{ attrs.axis.tag.name(), attrs.low, attrs.high, attrs.interior });
    }
};

// -----------------------------------------------------------------------------

pub const repeat = struct {
    pub const Kind = enum { repeat1d, repeat, stutter1d, stutter };
    pub const Attrs = struct { kind: Kind, axis: AxisRef, reps: List(u8, 8) };
    pub const weight = 4;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .min_rank = 1, .max_rank = Shape.MAX_RANK - 1 });
        const sh = g.shapeOf(x);
        var reps: List(u8, 8) = .{};
        for (0..sh.rank()) |_| reps.append(1);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        var axis: AxisRef = .random(g, sh, ax);
        // stutter1d takes an i8 axis.
        if (kind == .stutter1d) axis = .{ .index = axis.index };
        switch (kind) {
            .repeat1d, .stutter1d => reps.mut()[ax] = g.intIn(u8, 1, 3),
            .repeat, .stutter => for (reps.mut()) |*r| {
                if (g.chance(0.4)) r.* = g.intIn(u8, 1, 3);
            },
        }
        var out = sh;
        for (reps.slice(), 0..) |r, a| out = out.setDim(a, sh.dim(a) * r);
        if (out.count() > g.cfg.max_elems * 2) return error.Reject;
        return .{ .op = .{ .repeat = .{ .kind = kind, .axis = axis, .reps = reps } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    fn map(ctx: anytype, out: []const i64, in: []i64) bool {
        for (out, 0..) |o, ax| {
            const d = ctx.in.dim(ax);
            // repeat tiles the axis, stutter repeats each element in place.
            in[ax] = if (ctx.stutter) @divFloor(o, ctx.reps[ax]) else @mod(o, d);
        }
        return true;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const stutter = attrs.kind == .stutter or attrs.kind == .stutter1d;
        return k.remap(e, ins[0], node.shape, .{ .in = ins[0].shape, .reps = attrs.reps.slice(), .stutter = stutter }, map, null);
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const ax = attrs.axis.resolve(node.shape);
        const n = attrs.reps.slice()[ax];
        switch (attrs.kind) {
            .repeat1d => return switch (attrs.axis.pass()) {
                inline else => |a| x.repeat1d(a, n),
            },
            .stutter1d => return x.stutter1d(attrs.axis.index, n),
            .repeat => {
                var r: [8]u63 = undefined;
                for (attrs.reps.slice(), 0..) |v, i| r[i] = v;
                return x.repeat(r[0..attrs.reps.len]);
            },
            .stutter => {
                var r: [8]u32 = undefined;
                for (attrs.reps.slice(), 0..) |v, i| r[i] = v;
                return x.stutter(r[0..attrs.reps.len]);
            },
        }
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const ax = attrs.axis.resolve(node.shape);
        switch (attrs.kind) {
            .repeat1d, .stutter1d => {
                try w.print("{s}.{s}(", .{ ins[0], @tagName(attrs.kind) });
                try attrs.axis.print(w);
                try w.print(", {d})", .{attrs.reps.slice()[ax]});
            },
            .repeat => {
                try w.print("{s}.repeat(", .{ins[0]});
                var r: [8]u63 = undefined;
                for (attrs.reps.slice(), 0..) |v, i| r[i] = v;
                try printInts(w, u63, r[0..attrs.reps.len]);
                try w.writeAll(")");
            },
            .stutter => {
                try w.print("{s}.stutter(", .{ins[0]});
                var r: [8]u32 = undefined;
                for (attrs.reps.slice(), 0..) |v, i| r[i] = v;
                try printInts(w, u32, r[0..attrs.reps.len]);
                try w.writeAll(")");
            },
        }
    }
};
