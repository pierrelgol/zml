//! Shared helpers for op families.

const std = @import("std");

const zml = @import("zml");
pub const Shape = zml.Shape;
pub const DataType = zml.DataType;
pub const Tensor = zml.Tensor;

pub const ir = @import("ir.zig");
pub const gen = @import("gen.zig");
pub const Gen = gen.Gen;
pub const Proposal = gen.Proposal;
pub const Value = @import("Value.zig");
pub const ops = @import("ops.zig");
pub const Eval = ops.Eval;
pub const Node = ir.Node;
pub const TagId = ir.TagId;
pub const List = ir.List;
pub const Writer = std.Io.Writer;

pub const EvalError = Eval.Error;
pub const GenError = gen.Error;

/// Stores `x` into element `i` and reports whether it was stored without rounding.
pub fn storeF(v: Value, i: usize, x: f64) bool {
    v.setF(i, x);
    return v.getF(i) == x;
}

/// Error added by rounding a result whose largest magnitude is `mag`.
pub fn roundErr(dt: DataType, mag: f64) f64 {
    return Value.eps(dt) * mag;
}

/// Error of a transcendental function result; device implementations differ by a few ulps
/// from libm, and `floor` covers results near zero.
pub fn approxErr(dt: DataType, mag: f64, floor_: f64) f64 {
    // f32 kernels in XLA are accurate to a few ulps; reduced precision types are computed
    // in f32 and rounded once.
    const ulps: f64 = if (dt == .f32 or dt == .f64) 16 else 2;
    return ulps * Value.eps(dt) * @max(mag, floor_);
}

pub fn maxErr(ins: []const Value) f64 {
    var m: f64 = 0;
    for (ins) |v| m = @max(m, v.err);
    return m;
}

/// Shape with the same dims and tags but another dtype.
pub fn withDtype(sh: Shape, dt: DataType) Shape {
    return sh.withDtype(dt);
}

pub fn tagLiteral(t: Shape.Tag) []const u8 {
    return TagId.of(t).name();
}

/// Prints a Zig literal for an axis: `.a` when tagged, the index otherwise.
pub fn printAxis(w: *Writer, sh: Shape, ax: usize) Writer.Error!void {
    const t = sh.tag(ax);
    if (t != Shape.TagUnknown) return w.print(".{s}", .{tagLiteral(t)});
    return w.print("{d}", .{ax});
}

/// Prints a `zml.Shape.init(.{ .a = 2, ... }, .f32)` expression.
pub fn printShapeLiteral(w: *Writer, sh: Shape) Writer.Error!void {
    try w.writeAll("zml.Shape.init(.{ ");
    var all_tagged = true;
    for (sh.tags()) |t| if (t == Shape.TagUnknown) {
        all_tagged = false;
    };
    for (0..sh.rank()) |ax| {
        if (ax > 0) try w.writeAll(", ");
        if (all_tagged) try w.print(".{s} = ", .{tagLiteral(sh.tag(ax))});
        try w.print("{d}", .{sh.dim(ax)});
    }
    try w.print(" }}, .{s})", .{@tagName(sh.dtype())});
    if (!all_tagged) {
        var any = false;
        for (sh.tags()) |t| if (t != Shape.TagUnknown) {
            any = true;
        };
        if (any) {
            try w.writeAll(".withTags(.{ ");
            for (0..sh.rank()) |ax| {
                if (ax > 0) try w.writeAll(", ");
                try w.print(".{s}", .{tagLiteral(sh.tag(ax))});
            }
            try w.writeAll(" })");
        }
    }
}

/// Axis selection as the user would write it: by tag when the axis is tagged, else by index.
/// Stored in attributes so both spellings get exercised.
pub const AxisRef = struct {
    index: i8,
    tag: TagId = .none,
    /// Spelled `._N`, the tag syntax for axis N (index is then non negative).
    literal: bool = false,

    const literal_tags = [_]Shape.Tag{ Shape.toTag(._0), Shape.toTag(._1), Shape.toTag(._2), Shape.toTag(._3), Shape.toTag(._4), Shape.toTag(._5), Shape.toTag(._6), Shape.toTag(._7) };

    pub fn resolve(self: AxisRef, sh: Shape) usize {
        if (self.tag != .none) return sh.hasTag(self.tag.tag()).?;
        return if (self.index < 0) @intCast(@as(i64, sh.rank()) + self.index) else @intCast(self.index);
    }

    /// Calls `f` with the axis argument in the spelling this ref uses.
    pub fn pass(self: AxisRef) union(enum) { tag: Shape.Tag, index: i8 } {
        if (self.tag != .none) return .{ .tag = self.tag.tag() };
        if (self.literal) return .{ .tag = literal_tags[@intCast(self.index)] };
        return .{ .index = self.index };
    }

    pub fn print(self: AxisRef, w: *Writer) Writer.Error!void {
        if (self.tag != .none) return w.print(".{s}", .{self.tag.name()});
        if (self.literal) return w.print("._{d}", .{self.index});
        return w.print("{d}", .{self.index});
    }

    /// Picks a random spelling for axis `ax` of `sh`: tag, positive or negative index, or `._N`.
    pub fn random(g: *Gen, sh: Shape, ax: usize) AxisRef {
        const t = sh.tag(ax);
        const r = g.rng.uintLessThan(u8, 10);
        if (t != Shape.TagUnknown and r < 6) return .{ .index = @intCast(ax), .tag = TagId.of(t) };
        if (r < 7) return .{ .index = @intCast(ax), .literal = true };
        if (r < 8) return .{ .index = @intCast(ax) };
        return .{ .index = @intCast(@as(i64, @intCast(ax)) - sh.rank()) };
    }
};

/// A node operand used by the regression emitter.
pub fn printOperands(w: *Writer, ins: []const []const u8) Writer.Error!void {
    for (ins, 0..) |in, i| {
        if (i > 0) try w.writeAll(", ");
        try w.writeAll(in);
    }
}

/// Enum literals for the tag pool, indexable at comptime: `tag_literals[i]` is the literal
/// whose name is `ir.tag_names[i]`.
pub const tag_literals = .{ .a, .b, .c, .d, .e, .f, .g, .h, .k, .m, .n, .p, .coord, .bitcast, .bytes, .q, .s, .hd, .voc, .sc, .v };

/// A struct type with a single field named after tag `i`, as ZML APIs taking
/// `.{ .a = ... }` expect.
pub fn TagStruct(comptime i: usize, comptime T: type) type {
    return @Struct(.auto, null, &.{ir.tag_names[i]}, &.{T}, &.{.{}});
}

/// Builds `.{ .<tag> = value }` for a runtime tag and passes it to `body.call`.
pub fn callWithTagStruct(tid: TagId, value: anytype, body: anytype) @TypeOf(body).Result {
    inline for (0..ir.tag_names.len) |i| {
        if (@intFromEnum(tid) == i) {
            var s: TagStruct(i, @TypeOf(value)) = undefined;
            @field(s, ir.tag_names[i]) = value;
            return body.call(s);
        }
    }
    unreachable;
}

/// Passes the enum literal of a runtime tag to `body.call`.
pub fn callWithTagLiteral(tid: TagId, body: anytype) @TypeOf(body).Result {
    inline for (0..ir.tag_names.len) |i| {
        if (@intFromEnum(tid) == i) return body.call(tag_literals[i]);
    }
    unreachable;
}

/// Generic index remapping used by layout ops. For each output index, `map` fills the
/// input index and returns false when the element comes from `fill` instead.
pub fn remap(e: *Eval, in: Value, out_shape: Shape, ctx_: anytype, comptime map: anytype, fill: ?Value) EvalError!Value {
    const ctx = ctx_;
    var v = try e.alloc(out_shape);
    const st = Value.strides(in.shape);
    var it: Value.Index = .init(out_shape);
    var i: usize = 0;
    var in_idx: [Shape.MAX_RANK]i64 = @splat(0);
    // A pure index map keeps per element bounds when all its sources track them.
    const errs: ?[]f64 = if (in.errs != null and (fill == null or fill.?.errs != null)) try v.trackErrs(e.allocator) else null;
    while (!it.done) : (i += 1) {
        if (map(ctx, it.idx[0..out_shape.rank()], in_idx[0..in.shape.rank()])) {
            const j = Value.offset(st, in_idx[0..in.shape.rank()]);
            v.copyElem(i, in, j);
            if (errs) |es| es[i] = in.errs.?[j];
        } else {
            v.copyElem(i, fill.?, 0);
            if (errs) |es| es[i] = fill.?.errs.?[0];
        }
        _ = it.next();
    }
    v.rounded = in.rounded or (fill != null and fill.?.rounded);
    v.contractible = in.contractible;
    if (errs != null) {
        v.finishErrs();
    } else {
        v.err = in.err;
        if (fill) |f| v.err = @max(v.err, f.err);
    }
    return v;
}

pub fn dupeIds(g: *Gen, ids: []const u32) GenError![]const u32 {
    return g.allocator.dupe(u32, ids);
}

/// Like `callWithTagStruct` with two fields, `.{ .<t0> = v0, .<t1> = v1 }`.
pub fn callWithTagStruct2(t0: TagId, v0: anytype, t1: TagId, v1: anytype, body: anytype) @TypeOf(body).Result {
    inline for (0..ir.tag_names.len) |i| {
        if (@intFromEnum(t0) == i) {
            inline for (0..ir.tag_names.len) |j| {
                if (i != j and @intFromEnum(t1) == j) {
                    const S = @Struct(.auto, null, &.{ ir.tag_names[i], ir.tag_names[j] }, &.{ @TypeOf(v0), @TypeOf(v1) }, &.{ .{}, .{} });
                    var s: S = undefined;
                    @field(s, ir.tag_names[i]) = v0;
                    @field(s, ir.tag_names[j]) = v1;
                    return body.call(s);
                }
            }
        }
    }
    unreachable;
}

/// Operand shapes of the node being emitted, for emitters that print operand tags.
pub threadlocal var emit_shapes: []const Shape = &.{};

/// How a list argument is passed: as a slice, or as a tuple of the same runtime values.
/// ZML parses the two through different branches.
pub const Spelling = enum { slice, tuple };

/// Builds a tuple holding `items` and passes it to `body.call`.
pub fn callWithTuple(comptime T: type, items: []const T, body: anytype) @TypeOf(body).Result {
    inline for (0..9) |n| {
        if (items.len == n) {
            var t: @Tuple(&([_]type{T} ** n)) = undefined;
            inline for (0..n) |i| t[i] = items[i];
            return body.call(t);
        }
    }
    unreachable;
}

/// Passes `items` to `body.call` with the given spelling.
pub fn callWithList(comptime T: type, sp: Spelling, items: []const T, body: anytype) @TypeOf(body).Result {
    return switch (sp) {
        .slice => body.call(items),
        .tuple => callWithTuple(T, items, body),
    };
}

/// Prints a list of axes the way `callWithList` passes them: a slice or a tuple of tags
/// or indices. `tags` selects tag literals over integers.
pub fn printAxisList(w: *Writer, sp: Spelling, tags: bool, sh: Shape, axes_: []const u8) Writer.Error!void {
    if (sp == .slice) try w.print("@as([]const {s}, &.{{ ", .{if (tags) "zml.Shape.Tag" else "i64"}) else try w.writeAll(".{ ");
    for (axes_, 0..) |ax, i| {
        if (i > 0) try w.writeAll(", ");
        if (tags) {
            if (sp == .slice) try w.print("zml.Shape.toTag(.{s})", .{tagLiteral(sh.tag(ax))}) else try w.print(".{s}", .{tagLiteral(sh.tag(ax))});
        } else {
            if (sp == .slice) try w.print("{d}", .{ax}) else try w.print("@as(i64, {d})", .{ax});
        }
    }
    try w.writeAll(if (sp == .slice) " })" else " }");
}
