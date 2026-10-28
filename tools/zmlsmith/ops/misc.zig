//! The rest of the Tensor API: chunking and splitting, roll, triangular and diagonal
//! matrices, cartesian products, axis grouping, width changing bitcasts, barriers,
//! constant helpers, mask, fmod, reducePrecision, softmaxBiased and swiglu.

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

fn axisOf(ref: AxisRef, sh: Shape) usize {
    return ref.resolve(sh);
}

/// Copies the slice [start, start+len) of `x` along `ax` into a new value.
fn sliceAxis(e: *Eval, x: Value, ax: usize, start: i64, out_shape: Shape) Eval.Error!Value {
    return k.remap(e, x, out_shape, .{ .ax = ax, .start = start }, struct {
        fn map(c: anytype, out: []const i64, in: []i64) bool {
            @memcpy(in, out);
            in[c.ax] += c.start;
            return true;
        }
    }.map, null);
}

// -----------------------------------------------------------------------------

pub const chunk = struct {
    pub const Kind = enum { exact, allow_trailing, split };
    pub const Attrs = struct { kind: Kind, axis: AxisRef, n: u8, sizes: List(i64, 6), pick: u8 };
    pub const weight = 4;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .min_rank = 1 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        const d = sh.dim(ax);
        if (d == 0) return error.Reject;
        var a: Attrs = .{ .kind = kind, .axis = .random(g, sh, ax), .n = 1, .sizes = .{}, .pick = 0 };
        // Documented pieces, in order.
        var sizes: List(i64, 8) = .{};
        switch (kind) {
            .exact => {
                var cands: [3]u8 = undefined;
                var nc: usize = 0;
                for ([_]u8{ 2, 3, 4 }) |n| if (@mod(d, n) == 0) {
                    cands[nc] = n;
                    nc += 1;
                };
                if (nc == 0) return error.Reject;
                a.n = cands[g.rng.uintLessThan(usize, nc)];
                for (0..a.n) |_| sizes.append(@divExact(d, a.n));
            },
            .allow_trailing => {
                // chunkAllowTrailing takes an i64 axis.
                a.axis = .{ .index = a.axis.index };
                a.n = g.intIn(u8, 1, @intCast(@min(d, 5)));
                const size = @divFloor(d, a.n);
                for (0..a.n) |_| sizes.append(size);
                if (d - size * a.n > 0) sizes.append(d - size * a.n);
                try g.avoid(.chunk_trailing, @rem(d, size) != d - size * a.n);
            },
            .split => {
                var rest = d;
                while (rest > 0 and a.sizes.len < 5) {
                    const s = g.intIn(i64, 1, rest);
                    a.sizes.append(s);
                    sizes.append(s);
                    rest -= s;
                }
                if (rest > 0) {
                    a.sizes.append(rest);
                    sizes.append(rest);
                }
            },
        }
        a.pick = @intCast(g.rng.uintLessThan(usize, sizes.len));
        return .{ .op = .{ .chunk = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = sh.setDim(ax, sizes.slice()[a.pick]) };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = axisOf(attrs.axis, x.shape);
        const d = x.shape.dim(ax);
        var start: i64 = 0;
        switch (attrs.kind) {
            .exact, .allow_trailing => start = @as(i64, attrs.pick) * @divFloor(d, attrs.n),
            .split => for (attrs.sizes.slice()[0..attrs.pick]) |s| {
                start += s;
            },
        }
        return sliceAxis(e, x, ax, start, node.shape);
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        switch (attrs.kind) {
            .exact => {
                inline for (.{ 2, 3, 4 }) |n| if (attrs.n == n) {
                    const chunks = switch (attrs.axis.pass()) {
                        inline else => |a| x.chunkExact(a, n),
                    };
                    return chunks[attrs.pick];
                };
                unreachable;
            },
            .allow_trailing => {
                const chunks = x.chunkAllowTrailing(attrs.axis.index, attrs.n);
                // A missing piece is reported by the shape oracle as a mismatch on this node.
                return chunks[@min(attrs.pick, chunks.len - 1)];
            },
            .split => {
                const pieces = switch (attrs.axis.pass()) {
                    inline else => |a| x.split(a, attrs.sizes.slice()),
                };
                return pieces[attrs.pick];
            },
        }
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (attrs.kind) {
            .exact => {
                try w.print("{s}.chunkExact(", .{ins[0]});
                try attrs.axis.print(w);
                try w.print(", {d})[{d}]", .{ attrs.n, attrs.pick });
            },
            .allow_trailing => try w.print("{s}.chunkAllowTrailing({d}, {d})[{d}]", .{ ins[0], attrs.axis.index, attrs.n, attrs.pick }),
            .split => {
                try w.print("{s}.split(", .{ins[0]});
                try attrs.axis.print(w);
                try w.writeAll(", &.{ ");
                for (attrs.sizes.slice(), 0..) |s, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", s });
                try w.print(" }})[{d}]", .{attrs.pick});
            },
        }
    }
};

// -----------------------------------------------------------------------------

pub const roll = struct {
    pub const Attrs = struct { axis: AxisRef, shift: i64 };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 1 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        if (sh.dim(ax) == 0) return error.Reject;
        return .{ .op = .{ .roll = .{ .axis = .random(g, sh, ax), .shift = g.intIn(i64, -9, 9) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = sh };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const ax = axisOf(attrs.axis, node.shape);
        return k.remap(e, ins[0], node.shape, .{ .ax = ax, .shift = attrs.shift, .n = node.shape.dim(ax) }, struct {
            fn map(c: anytype, out: []const i64, in: []i64) bool {
                @memcpy(in, out);
                in[c.ax] = @mod(out[c.ax] - c.shift, c.n);
                return true;
            }
        }.map, null);
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return switch (attrs.axis.pass()) {
            inline else => |a| ins[0].rollRight1d(a, attrs.shift),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.rollRight1d(", .{ins[0]});
        try attrs.axis.print(w);
        try w.print(", {d})", .{attrs.shift});
    }
};

// -----------------------------------------------------------------------------

pub const triangular = struct {
    pub const Attrs = struct { a: AxisRef, b: AxisRef, diagonals: i32 };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 2 });
        const sh = g.shapeOf(x);
        var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        g.rng.shuffle(u8, perm[0..sh.rank()]);
        return .{
            .op = .{ .triangular = .{ .a = .random(g, sh, perm[0]), .b = .random(g, sh, perm[1]), .diagonals = g.intIn(i32, -2, 2) } },
            .inputs = try k.dupeIds(g, &.{x}),
            .shape = sh,
        };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        var zero = try e.alloc(.init(.{}, x.dtype()));
        _ = try zero.trackErrs(e.allocator);
        // Documented: keeps element (i, j) when i + num_diagonals >= j.
        return k.remap(e, x, node.shape, .{ .a = axisOf(attrs.a, x.shape), .b = axisOf(attrs.b, x.shape), .n = attrs.diagonals }, struct {
            fn map(c: anytype, out: []const i64, in: []i64) bool {
                @memcpy(in, out);
                return out[c.a] + c.n >= out[c.b];
            }
        }.map, zero);
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const a: i8 = @intCast(axisOf(attrs.a, ins[0].shape()));
        const b: i8 = @intCast(axisOf(attrs.b, ins[0].shape()));
        if (attrs.a.tag != .none and attrs.b.tag != .none) return ins[0].triangular(.{ attrs.a.tag.tag(), attrs.b.tag.tag() }, attrs.diagonals);
        return ins[0].triangular(.{ a, b }, attrs.diagonals);
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.triangular(.{{ ", .{ins[0]});
        try attrs.a.print(w);
        try w.writeAll(", ");
        try attrs.b.print(w);
        try w.print(" }}, {d})", .{attrs.diagonals});
    }
};

// -----------------------------------------------------------------------------

pub const to_diagonal = struct {
    pub const Attrs = struct { axis: AxisRef, t0: TagId, t1: TagId };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 1, .max_rank = Shape.MAX_RANK - 3 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        const t0 = g.freshTag(sh);
        var t1 = g.freshTag(sh);
        while (t1 == t0) t1 = g.freshTag(sh);
        const d = sh.dim(ax);
        const out = sh.remove(ax).insertTag(ax, d, t1).insertTag(ax, d, t0);
        if (out.count() > 4 * g.cfg.max_elems) return error.Reject;
        return .{ .op = .{ .to_diagonal = .{ .axis = .random(g, sh, ax), .t0 = TagId.of(t0), .t1 = TagId.of(t1) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        var zero = try e.alloc(.init(.{}, x.dtype()));
        _ = try zero.trackErrs(e.allocator);
        return k.remap(e, x, node.shape, .{ .ax = axisOf(attrs.axis, x.shape), .rank = x.shape.rank() }, struct {
            fn map(c: anytype, out: []const i64, in: []i64) bool {
                for (0..c.rank) |a| in[a] = if (a <= c.ax) out[a] else out[a + 1];
                return out[c.ax] == out[c.ax + 1];
            }
        }.map, zero);
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        return k.callWithTagLiteral(attrs.t0, struct {
            x: Tensor,
            a: Attrs,
            pub const Result = Tensor;
            pub fn call(b: @This(), comptime l0: anytype) Tensor {
                return k.callWithTagLiteral(b.a.t1, struct {
                    x: Tensor,
                    a: Attrs,
                    pub const Result = Tensor;
                    pub fn call(c: @This(), comptime l1: anytype) Tensor {
                        return switch (c.a.axis.pass()) {
                            inline else => |ax| c.x.toDiagonal(ax, .{ l0, l1 }),
                        };
                    }
                }{ .x = b.x, .a = b.a });
            }
        }{ .x = x, .a = attrs });
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.toDiagonal(", .{ins[0]});
        try attrs.axis.print(w);
        try w.print(", .{{ .{s}, .{s} }})", .{ attrs.t0.name(), attrs.t1.name() });
    }
};

// -----------------------------------------------------------------------------

pub const cartesian = struct {
    pub const Attrs = struct { stacked: bool, pick: u8 };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const n = g.intIn(usize, 1, 3);
        const dt = g.randDtype(.any) orelse return null;
        var ids: [4]u32 = undefined;
        var out: Shape = .init(.{}, dt);
        var used: Shape = .init(.{}, dt);
        for (0..n) |i| {
            const rank: u4 = if (g.chance(0.2) and !g.cfg.avoid.contains(.cartesian_scalar)) 0 else 1;
            var sh: Shape = .init(.{}, dt);
            if (rank == 1) {
                const t = if (g.chance(0.7)) g.freshTag(used) else Shape.TagUnknown;
                sh = sh.appendDim(g.intIn(i64, 1, 5), t);
                if (t != Shape.TagUnknown) used = used.appendDim(1, t);
            }
            ids[i] = try g.operandLike(sh, true);
            const s = g.shapeOf(ids[i]);
            out = if (s.rank() == 0) out.appendDim(1, null) else out.appendDim(s.dim(0), s.tag(0));
        }
        const stacked = g.chance(0.4);
        const pick: u8 = @intCast(g.rng.uintLessThan(usize, n));
        if (stacked) out = out.appendDim(@intCast(n), Shape.toTag(.coord));
        return .{ .op = .{ .cartesian = .{ .stacked = stacked, .pick = pick } }, .inputs = try k.dupeIds(g, ids[0..n]), .shape = out };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        while (!it.done) : (i += 1) {
            const which: usize = if (attrs.stacked) @intCast(it.idx[ins.len]) else attrs.pick;
            const src = ins[which];
            v.copyElem(i, src, if (src.shape.rank() == 0) 0 else @intCast(it.idx[which]));
            _ = it.next();
        }
        v.err = k.maxErr(ins);
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        if (attrs.stacked) return Tensor.cartesianProductStacked(ins);
        inline for (1..4) |n| if (ins.len == n) {
            var arr: [n]Tensor = undefined;
            @memcpy(&arr, ins);
            return Tensor.cartesianProduct(n, arr)[attrs.pick];
        };
        unreachable;
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        if (attrs.stacked) {
            try w.writeAll("zml.Tensor.cartesianProductStacked(&.{ ");
            try k.printOperands(w, ins);
            return w.writeAll(" })");
        }
        try w.print("zml.Tensor.cartesianProduct({d}, .{{ ", .{ins.len});
        try k.printOperands(w, ins);
        try w.print(" }})[{d}]", .{attrs.pick});
    }
};

// -----------------------------------------------------------------------------

/// contiguous(axes) and mergeTranspose(axes, merged).
pub const group = struct {
    pub const Attrs = struct { axes: List(TagId, 4), merge: bool, merged: TagId = .none, list: k.Spelling = .slice };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .min_rank = 2, .tagged = true });
        const sh = g.shapeOf(x);
        var perm: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        g.rng.shuffle(u8, perm[0..sh.rank()]);
        const merge = g.chance(0.5);
        const n = g.intIn(usize, if (merge) 2 else 1, @min(4, sh.rank()));
        var a: Attrs = .{ .axes = .{}, .merge = merge, .list = g.rng.enumValue(k.Spelling) };
        for (perm[0..n]) |ax| a.axes.append(TagId.of(sh.tag(ax)));
        // Documented: the other axes keep their order, then the listed ones in the given order.
        var out: Shape = .init(.{}, sh.dtype());
        for (0..sh.rank()) |ax| if (std.mem.indexOfScalar(u8, perm[0..n], @intCast(ax)) == null) {
            out = out.appendDim(sh.dim(ax), sh.tag(ax));
        };
        if (merge) {
            var d: i64 = 1;
            for (perm[0..n]) |ax| d *= sh.dim(ax);
            var t = g.freshTag(sh);
            if (g.chance(0.3)) t = sh.tag(perm[0]);
            a.merged = TagId.of(t);
            out = out.appendDim(d, t);
        } else {
            for (perm[0..n]) |ax| out = out.appendDim(sh.dim(ax), sh.tag(ax));
        }
        return .{ .op = .{ .group = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        var perm: [8]u8 = undefined;
        var n: usize = 0;
        for (0..x.shape.rank()) |ax| {
            var listed = false;
            for (attrs.axes.slice()) |t| if (x.shape.tag(ax) == t.tag()) {
                listed = true;
            };
            if (!listed) {
                perm[n] = @intCast(ax);
                n += 1;
            }
        }
        for (attrs.axes.slice()) |t| {
            perm[n] = x.shape.hasTag(t.tag()).?;
            n += 1;
        }
        // Transposed data in row-major order is also the merged data.
        var tsh = x.shape;
        for (perm[0..n], 0..) |p, i| {
            tsh._dims.set(i, x.shape.dim(p));
            tsh._tags.set(i, x.shape.tag(p));
        }
        var v = try k.remap(e, x, tsh, perm[0..n], struct {
            fn map(p: []const u8, out: []const i64, in: []i64) bool {
                for (p, 0..) |pp, i| in[pp] = out[i];
                return true;
            }
        }.map, null);
        v.shape = node.shape;
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        var t: [4]Shape.Tag = undefined;
        for (attrs.axes.slice(), 0..) |x, i| t[i] = x.tag();
        const tags: []const Shape.Tag = t[0..attrs.axes.len];
        return k.callWithList(Shape.Tag, attrs.list, tags, struct {
            x: Tensor,
            a: Attrs,
            pub const Result = Tensor;
            pub fn call(c: @This(), list: anytype) Tensor {
                if (!c.a.merge) return c.x.contiguous(list);
                return k.callWithTagLiteral(c.a.merged, struct {
                    x: Tensor,
                    list: @TypeOf(list),
                    pub const Result = Tensor;
                    pub fn call(b: @This(), comptime lit: anytype) Tensor {
                        return b.x.mergeTranspose(b.list, lit);
                    }
                }{ .x = c.x, .list = list });
            }
        }{ .x = ins[0], .a = attrs });
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.{s}(", .{ ins[0], if (attrs.merge) "mergeTranspose" else "contiguous" });
        var axes_: [4]u8 = undefined;
        for (attrs.axes.slice(), 0..) |t, i| axes_[i] = k.emit_shapes[0].hasTag(t.tag()).?;
        try k.printAxisList(w, attrs.list, true, k.emit_shapes[0], axes_[0..attrs.axes.len]);
        if (attrs.merge) try w.print(", .{s}", .{attrs.merged.name()});
        try w.writeAll(")");
    }
};

// -----------------------------------------------------------------------------

/// bitCast to a different width, and bytes().
pub const reinterpret = struct {
    pub const Attrs = struct { bytes: bool };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const bytes = g.chance(0.4);
        const x = try g.operand(.{ .class = .numeric, .exact = true, .max_rank = Shape.MAX_RANK - 2 });
        const sh = g.shapeOf(x);
        const src = sh.dtype();
        // Chained narrowing bitcasts hit an XLA verifier bug.
        const prev = g.prog.nodes.items[x].op;
        try g.avoid(.xla_bitcast_chain, prev == .reinterpret or (prev == .convert and prev.convert.bitcast));
        if (bytes) {
            // Documented: the tensor as one contiguous buffer of bytes, tagged .bytes.
            const n: i64 = @intCast(sh.count() * src.sizeOf());
            return .{ .op = .{ .reinterpret = .{ .bytes = true } }, .inputs = try k.dupeIds(g, &.{x}), .shape = Shape.init(.{n}, .u8).withTags(.{.bytes}) };
        }
        var cands: [16]DataType = undefined;
        var nc: usize = 0;
        for (g.cfg.dtypes) |dt| {
            if (dt == .bool or dt.bitSizeOf() == src.bitSizeOf()) continue;
            const wider = dt.bitSizeOf() > src.bitSizeOf();
            if (wider and (sh.rank() == 0 or sh.dim(-1) * src.bitSizeOf() != dt.bitSizeOf())) continue;
            if (nc < cands.len) {
                cands[nc] = dt;
                nc += 1;
            }
        }
        if (nc == 0) return error.Reject;
        const dst = cands[g.rng.uintLessThan(usize, nc)];
        const out = if (dst.bitSizeOf() < src.bitSizeOf())
            sh.appendDim(@divExact(src.bitSizeOf(), dst.bitSizeOf()), Shape.toTag(.bitcast)).withDtype(dst)
        else
            sh.remove(-1).withDtype(dst);
        return .{ .op = .{ .reinterpret = .{ .bytes = false } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, _: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        if (ins[0].err > 0) try e.reject();
        // The bits expose the sign of zeros, only known when the producer tracks per
        // element bounds (a dot of zeros may give -0).
        if (Value.isFloat(ins[0].dtype()) and ins[0].errs == null) {
            for (0..ins[0].count()) |i| if (ins[0].getF(i) == 0) try e.reject();
        }
        var v = try e.alloc(node.shape);
        @memcpy(v.bytes, ins[0].bytes);
        if (Value.isFloat(node.shape.dtype()) and !v.allFinite()) try e.reject();
        return v;
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        if (attrs.bytes) return ins[0].bytes();
        return ins[0].bitCast(node.shape.dtype());
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        if (attrs.bytes) return w.print("{s}.bytes()", .{ins[0]});
        try w.print("{s}.bitCast(.{s})", .{ ins[0], @tagName(node.shape.dtype()) });
    }
};

// -----------------------------------------------------------------------------

/// Small helpers: optimizationBarrier, zeroes, constant, mask, withPartialTags.
pub const helpers = struct {
    pub const Kind = enum { barrier, zeroes, constant, mask, partial_tags };
    pub const Attrs = struct { kind: Kind, value: i64 = 0, tags: List(TagId, 4) = .{}, list: k.Spelling = .slice };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 4;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        var a: Attrs = .{ .kind = kind, .list = g.rng.enumValue(k.Spelling) };
        switch (kind) {
            .barrier => {
                const x = try g.operand(.{});
                return .{ .op = .{ .helpers = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = g.shapeOf(x) };
            },
            .zeroes => {
                const dt = g.randDtype(.any) orelse return null;
                return .{ .op = .{ .helpers = a }, .shape = g.randShape(dt, .{}) };
            },
            .constant => {
                const dt = g.randDtype(.any) orelse return null;
                a.value = if (dt == .bool) g.intIn(i64, 0, 1) else if (Value.isInt(dt) and !dt.isSignedInt()) g.intIn(i64, 0, 7) else g.intIn(i64, -7, 7);
                return .{ .op = .{ .helpers = a }, .shape = .init(.{}, dt) };
            },
            .mask => {
                const x = try g.operand(.{ .min_rank = 1 });
                const sh = g.shapeOf(x);
                // The predicate broadcasts to x, here with the same dims.
                const p = try g.operandLike(sh.withDtype(.bool), true);
                const dt = sh.dtype();
                a.value = if (dt == .bool) g.intIn(i64, 0, 1) else if (Value.isInt(dt) and !dt.isSignedInt()) g.intIn(i64, 0, 7) else g.intIn(i64, -7, 7);
                return .{ .op = .{ .helpers = a }, .inputs = try k.dupeIds(g, &.{ x, p }), .shape = sh };
            },
            .partial_tags => {
                const x = try g.operand(.{ .min_rank = 1, .max_rank = 4 });
                const sh = g.shapeOf(x);
                const n = g.intIn(usize, 1, sh.rank());
                var tags: [8]Shape.Tag = undefined;
                g.randTags(tags[0..sh.rank()]);
                var out = sh;
                // Documented: tags the n last axes. Keep the result tags distinct.
                for (0..sh.rank() - n) |ax| for (tags[0..n]) |t| if (out.tag(ax) == t) {
                    out._tags.set(ax, Shape.TagUnknown);
                };
                if (!std.mem.eql(Shape.Tag, out.tags(), sh.tags())) return error.Reject;
                for (0..n) |i| {
                    a.tags.append(TagId.of(tags[i]));
                    out._tags.set(sh.rank() - n + i, tags[i]);
                }
                return .{ .op = .{ .helpers = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
            },
        }
    }

    fn fillScalar(v: Value, i: usize, value: i64) void {
        if (Value.isFloat(v.dtype())) v.setF(i, @floatFromInt(value)) else v.setI(i, value);
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        switch (attrs.kind) {
            .barrier, .partial_tags => {
                var v = try ins[0].dupe(e.allocator);
                v.shape = node.shape;
                return v;
            },
            .zeroes => return e.alloc(node.shape),
            .constant => {
                const v = try e.alloc(node.shape);
                fillScalar(v, 0, attrs.value);
                return v;
            },
            .mask => {
                var v = try ins[0].dupe(e.allocator);
                for (0..v.count()) |i| if (ins[1].bytes[i] == 0) fillScalar(v, i, attrs.value);
                return v;
            },
        }
    }

    pub fn lower(attrs: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const dt = node.shape.dtype();
        switch (attrs.kind) {
            .barrier => return ins[0].optimizationBarrier(),
            .zeroes => return Tensor.zeroes(node.shape),
            .constant => return if (Value.isFloat(dt))
                Tensor.constant(dt.constant(@as(f64, @floatFromInt(attrs.value))))
            else if (dt == .bool) Tensor.constant(dt.constant(attrs.value != 0)) else Tensor.constant(dt.constant(attrs.value)),
            .mask => return if (dt == .bool) ins[0].mask(ins[1], attrs.value != 0) else ins[0].mask(ins[1], attrs.value),
            .partial_tags => {
                var t: [4]Shape.Tag = undefined;
                for (attrs.tags.slice(), 0..) |x, i| t[i] = x.tag();
                return k.callWithList(Shape.Tag, attrs.list, t[0..attrs.tags.len], struct {
                    x: Tensor,
                    pub const Result = Tensor;
                    pub fn call(c: @This(), list: anytype) Tensor {
                        return c.x.withPartialTags(list);
                    }
                }{ .x = ins[0] });
            },
        }
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const dt = node.shape.dtype();
        switch (attrs.kind) {
            .barrier => try w.print("{s}.optimizationBarrier()", .{ins[0]}),
            .zeroes => {
                try w.writeAll("zml.Tensor.zeroes(");
                try k.printShapeLiteral(w, node.shape);
                try w.writeAll(")");
            },
            .constant => if (dt == .bool)
                try w.print("zml.Tensor.constant(zml.DataType.constant(.bool, {}))", .{attrs.value != 0})
            else if (k.Value.isFloat(dt))
                try w.print("zml.Tensor.constant(zml.DataType.constant(.{s}, @as(f64, {d})))", .{ @tagName(dt), attrs.value })
            else
                try w.print("zml.Tensor.constant(zml.DataType.constant(.{s}, @as(i64, {d})))", .{ @tagName(dt), attrs.value }),
            .mask => if (dt == .bool)
                try w.print("{s}.mask({s}, {})", .{ ins[0], ins[1], attrs.value != 0 })
            else
                try w.print("{s}.mask({s}, {d})", .{ ins[0], ins[1], attrs.value }),
            .partial_tags => {
                try w.print("{s}.withPartialTags(", .{ins[0]});
                const n = attrs.tags.len;
                var axes_: [4]u8 = undefined;
                for (0..n) |i| axes_[i] = @intCast(node.shape.rank() - n + i);
                try k.printAxisList(w, attrs.list, true, node.shape, axes_[0..n]);
                try w.writeAll(")");
            },
        }
    }
};

// -----------------------------------------------------------------------------

/// fmod, round and reducePrecision.
pub const float_ops = struct {
    pub const Kind = enum { fmod, round, reduce_precision };
    pub const Attrs = struct { kind: Kind, divisor: f64 = 1, exponent_bits: i32 = 8, mantissa_bits: i32 = 7 };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .class = .float });
        var a: Attrs = .{ .kind = kind };
        switch (kind) {
            .fmod => {
                a.divisor = g.choose(f64, &.{ 2, -1.5, 0.75, 3 });
                try g.avoid(.fmod_non_f32, g.shapeOf(x).dtype() != .f32);
            },
            .round => {},
            .reduce_precision => {
                // Narrow exponents overflow to inf and flush small values to zero.
                a.exponent_bits = if (g.chance(0.5)) g.intIn(i32, 1, 7) else g.intIn(i32, 8, 11);
                a.mantissa_bits = g.intIn(i32, 1, 23);
            },
        }
        return .{ .op = .{ .float_ops = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = g.shapeOf(x) };
    }

    /// Bits of the exponent field of `dt`.
    fn exponentBits(dt: DataType) i32 {
        return switch (dt) {
            .f64 => 11,
            .f16 => 5,
            else => 8,
        };
    }

    /// XLA's reduce-precision: round the mantissa to nearest even, then, when the exponent
    /// is narrower than the input's, overflow to inf and flush values below the smallest
    /// normal of the reduced format to zero, both keeping the sign (EmitReducePrecisionIR).
    /// The StableHLO spec describes a conversion to the reduced type, which keeps its
    /// subnormals.
    fn reducePrecision(x: f64, dt: DataType, exponent_bits: i32, mantissa_bits: i32) f64 {
        if (std.math.isNan(x)) return x;
        const r = roundMantissa(x, mantissa_bits);
        if (exponent_bits >= exponentBits(dt) or r == 0 or std.math.isInf(r)) return r;
        const bias = (@as(i64, 1) << @intCast(exponent_bits - 1)) - 1;
        const exp = @as(i64, std.math.frexp(r).exponent) - 1;
        if (exp > bias) return std.math.copysign(std.math.inf(f64), r);
        if (exp <= -bias) return std.math.copysign(@as(f64, 0), r);
        return r;
    }

    fn roundMantissa(x: f64, bits: i32) f64 {
        if (x == 0 or std.math.isInf(x)) return x;
        const e = std.math.frexp(x);
        // x = m * 2^exp with 0.5 <= |m| < 1; keep bits+1 significant bits.
        const scale = std.math.ldexp(@as(f64, 1), bits + 1);
        const m = e.significand * scale;
        var r = @round(m);
        if (@abs(m - @trunc(m)) == 0.5) r = 2 * @round(m / 2);
        return std.math.ldexp(r / scale, e.exponent);
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const dt = x.dtype();
        switch (attrs.kind) {
            .fmod => {
                // Documented as torch.fmod with an f32 divisor.
                var c = try e.alloc(.init(.{}, dt));
                c.setF(0, attrs.divisor);
                return @import("elementwise.zig").binary.eval(e, .{ .kind = .remainder, .bcast = .rhs_scalar }, node, &.{ x, c });
            },
            .round => return @import("elementwise.zig").unary.eval(e, .{ .kind = .round_even }, node, &.{x}),
            .reduce_precision => {
                const v = try e.alloc(node.shape);
                // Rounding is a discontinuity: imprecise inputs make it ambiguous.
                if (x.err > 0) try e.reject();
                for (0..x.count()) |i| _ = k.storeF(v, i, reducePrecision(x.getF(i), dt, attrs.exponent_bits, attrs.mantissa_bits));
                return v;
            },
        }
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return switch (attrs.kind) {
            .fmod => ins[0].fmod(@floatCast(attrs.divisor)),
            .round => ins[0].round(),
            .reduce_precision => ins[0].reducePrecision(attrs.exponent_bits, attrs.mantissa_bits),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (attrs.kind) {
            .fmod => try w.print("{s}.fmod({d})", .{ ins[0], attrs.divisor }),
            .round => try w.print("{s}.round()", .{ins[0]}),
            .reduce_precision => try w.print("{s}.reducePrecision({d}, {d})", .{ ins[0], attrs.exponent_bits, attrs.mantissa_bits }),
        }
    }
};

// -----------------------------------------------------------------------------

pub const softmax_biased = struct {
    pub const Attrs = struct { axis: AxisRef, has_bias: bool };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .class = .float, .min_rank = 1 });
        const sh = g.shapeOf(x);
        const ax = g.rng.uintLessThan(usize, sh.rank());
        if (sh.dim(ax) == 0) return error.Reject;
        const has_bias = g.chance(0.8);
        var ids: [2]u32 = .{ x, 0 };
        if (has_bias) ids[1] = try g.operandLike(if (g.chance(0.5)) Shape.init(.{}, sh.dtype()) else sh.setDim(ax, 1), true);
        return .{ .op = .{ .softmax_biased = .{ .axis = .random(g, sh, ax), .has_bias = has_bias } }, .inputs = try k.dupeIds(g, ids[0..if (has_bias) 2 else 1]), .shape = sh };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = axisOf(attrs.axis, x.shape);
        const n: usize = @intCast(x.shape.dim(ax));
        var v = try e.alloc(node.shape);
        const st = Value.strides(x.shape);
        var it: Value.Index = .init(x.shape.setDim(ax, 1));
        var lane: usize = 0;
        while (!it.done) : (lane += 1) {
            const base = Value.offset(st, it.idx[0..x.shape.rank()]);
            // Documented: y[i] = exp(x[i]) / (sum_k exp(x[k]) + exp(bias)).
            var m: f64 = -std.math.inf(f64);
            for (0..n) |j| m = @max(m, x.getF(base + j * st[ax]));
            var b: ?f64 = null;
            if (attrs.has_bias) b = ins[1].getF(if (ins[1].shape.rank() == 0) 0 else lane);
            if (b) |bb| m = @max(m, bb);
            var s: f64 = 0;
            for (0..n) |j| s += @exp(x.getF(base + j * st[ax]) - m);
            if (b) |bb| s += @exp(bb - m);
            for (0..n) |j| {
                const off = base + j * st[ax];
                _ = k.storeF(v, off, @exp(x.getF(off) - m) / s);
            }
            _ = it.next();
        }
        const bias_err = if (attrs.has_bias) ins[1].err else 0;
        v.err = 2 * (x.err + bias_err) + k.approxErr(x.dtype(), 1, 1) + 4 * Value.eps(.f32) * @as(f64, @floatFromInt(n));
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const bias: ?Tensor = if (attrs.has_bias) ins[1] else null;
        return switch (attrs.axis.pass()) {
            inline else => |a| ins[0].softmaxBiased(a, bias),
        };
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.softmaxBiased(", .{ins[0]});
        try attrs.axis.print(w);
        try w.print(", {s})", .{if (attrs.has_bias) ins[1] else "null"});
    }
};

// -----------------------------------------------------------------------------

pub const swiglu = struct {
    pub const Attrs = struct { beta: f64 };
    pub const weight = 1;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        // swiglu(x[d], w[c, d], v[e, d], b[c], c[e], .d) -> [c, e], as documented by its test.
        const dt = g.randDtype(.float) orelse return null;
        var tags: [3]Shape.Tag = undefined;
        g.randTags(&tags);
        const d = g.intIn(i64, 1, 5);
        const c = g.intIn(i64, 1, 4);
        const e_ = g.intIn(i64, 1, 4);
        const x = try g.operandLike(k.ir.makeShape(dt, &.{d}, tags[0..1]), true);
        const w = try g.freshInput(k.ir.makeShape(dt, &.{ c, d }, &.{ tags[1], tags[0] }));
        const v = try g.freshInput(k.ir.makeShape(dt, &.{ e_, d }, &.{ tags[2], tags[0] }));
        const b = try g.freshInput(k.ir.makeShape(dt, &.{c}, tags[1..2]));
        const cc = try g.freshInput(k.ir.makeShape(dt, &.{e_}, tags[2..3]));
        return .{
            .op = .{ .swiglu = .{ .beta = g.choose(f64, &.{ 1, 0.5, 2 }) } },
            .inputs = try k.dupeIds(g, &.{ x, w, v, b, cc }),
            .shape = k.ir.makeShape(dt, &.{ c, e_ }, &.{ tags[1], tags[2] }),
        };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const w = ins[1];
        const vv = ins[2];
        const d: usize = @intCast(x.shape.dim(0));
        const c: usize = @intCast(w.shape.dim(0));
        const n_e: usize = @intCast(vv.shape.dim(0));
        var out = try e.alloc(node.shape);
        var mag: f64 = 0;
        for (0..c) |i| for (0..n_e) |j| {
            var value: f64 = ins[3].getF(i);
            var gate: f64 = ins[4].getF(j);
            for (0..d) |t| {
                value += w.getF(i * d + t) * x.getF(t);
                gate += vv.getF(j * d + t) * x.getF(t);
            }
            const r = value * (1 / (1 + @exp(-attrs.beta * value))) * gate;
            mag = @max(mag, @abs(value) + @abs(gate) + @abs(r));
            _ = k.storeF(out, i * n_e + j, r);
        };
        var err_in: f64 = 0;
        for (ins) |v| err_in = @max(err_in, v.err);
        out.err = (err_in + k.approxErr(x.dtype(), mag, 1)) * mag * @as(f64, @floatFromInt(4 * d + 4));
        return out;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return ins[0].swiglu(@floatCast(attrs.beta), ins[1], ins[2], ins[3], ins[4], ins[0].shape().tag(0));
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.swiglu({d}, {s}, {s}, {s}, {s}, zml.Shape.toTag(.{s}))", .{ ins[0], attrs.beta, ins[1], ins[2], ins[3], ins[4], k.tagLiteral(k.emit_shapes[0].tag(0)) });
    }
};

// -----------------------------------------------------------------------------

/// Buffer donation with reuseBuffer, and memory placement with toMemory / onMemory (no-ops
/// on CPU, so only their shapes are checked there).
pub const memory = struct {
    /// The `_all` kinds go through Tensor.toMemoryAll / onMemoryAll on a struct.
    pub const Kind = enum { reuse_buffer, to_memory, on_memory, to_memory_all, on_memory_all };
    pub const Attrs = struct { kind: Kind, mem: zml.Memory.Kind = .device };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        switch (kind) {
            .reuse_buffer => {
                // A computed node reusing the buffer of an argument with the same shape. Only
                // program arguments can be donated: top level only.
                if (g.scope != 0) return error.Reject;
                const x = try g.operand(.{});
                // A real computation: placement ops are identities on CPU and would alias an
                // argument, the "returned twice" case of KnownBug.reuse_duplicate_output.
                const xop = g.prog.nodes.items[x].op;
                if (k.ops.isArgument(xop) or xop == .memory) return error.Reject;
                for (g.prog.nodes.items) |m| if (m.op == .memory and m.op.memory.kind == .reuse_buffer and m.inputs.slice()[0] == x) return error.Reject;
                const sh = g.shapeOf(x);
                // An argument can be donated only once.
                var arg: ?u32 = null;
                for (g.prog.nodes.items, 0..) |n, i| {
                    if (!k.ops.isArgument(n.op) or !n.shape.eql(sh)) continue;
                    var donated = false;
                    for (g.prog.nodes.items) |m| if (m.op == .memory and m.op.memory.kind == .reuse_buffer and m.inputs.slice()[1] == i) {
                        donated = true;
                    };
                    if (!donated) arg = @intCast(i);
                }
                const origin = arg orelse try g.freshInput(sh);
                if (!k.ops.isArgument(g.prog.nodes.items[origin].op)) return error.Reject;
                return .{ .op = .{ .memory = .{ .kind = .reuse_buffer } }, .inputs = try k.dupeIds(g, &.{ x, origin }), .shape = sh };
            },
            .to_memory, .on_memory, .to_memory_all, .on_memory_all => {
                const x = try g.operand(.{ .allow_storage = true });
                return .{ .op = .{ .memory = .{ .kind = kind, .mem = g.rng.enumValue(zml.Memory.Kind) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = g.shapeOf(x) };
            },
        }
    }

    pub fn eval(e: *Eval, _: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var v = try ins[0].dupe(e.allocator);
        v.shape = node.shape;
        return v;
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return switch (a.kind) {
            .reuse_buffer => ins[0].reuseBuffer(ins[1]),
            .to_memory => ins[0].toMemory(a.mem),
            .on_memory => ins[0].onMemory(a.mem),
            .to_memory_all => Tensor.toMemoryAll(.{ .x = ins[0] }, a.mem).x,
            .on_memory_all => {
                Tensor.onMemoryAll(.{ .x = ins[0] }, a.mem);
                return ins[0];
            },
        };
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (a.kind) {
            .reuse_buffer => try w.print("{s}.reuseBuffer({s})", .{ ins[0], ins[1] }),
            .to_memory => try w.print("{s}.toMemory(.{s})", .{ ins[0], @tagName(a.mem) }),
            .on_memory => try w.print("{s}.onMemory(.{s})", .{ ins[0], @tagName(a.mem) }),
            .to_memory_all => try w.print("zml.Tensor.toMemoryAll(.{{ .x = {s} }}, .{s}).x", .{ ins[0], @tagName(a.mem) }),
            .on_memory_all => try w.print("blk: {{ zml.Tensor.onMemoryAll(.{{ .x = {s} }}, .{s}); break :blk {s}; }}", .{ ins[0], @tagName(a.mem), ins[0] }),
        }
    }
};
