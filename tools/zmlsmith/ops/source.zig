//! Nodes without tensor operands: program arguments and the tensor creation API.

const std = @import("std");

const k = @import("../kit.zig");
const Shape = k.Shape;
const Tensor = k.Tensor;
const Value = k.Value;
const Gen = k.Gen;
const Eval = k.Eval;
const Node = k.Node;
const Writer = k.Writer;

pub const input = struct {
    pub const Dist = enum { int_range, bits, uniform, dyadic, signed_zero };
    pub const Attrs = struct {
        seed: u64,
        dist: Dist,
        lo: i64 = -4,
        hi: i64 = 4,
    };
    pub const weight = 0;

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, _: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        k.gen.fillInput(v, attrs);
        // Exact data, sign of zero included.
        if (Value.isFloat(node.shape.dtype())) _ = try v.trackErrs(e.allocator);
        return v;
    }

    pub fn lower(_: Attrs, _: *const Node, _: []const Tensor) Tensor {
        unreachable;
    }

    pub fn emit(_: Attrs, _: *const Node, _: *Writer, _: []const []const u8) Writer.Error!void {
        unreachable;
    }
};

/// A program argument with explicit data. Used for indices, and by the reducer to
/// replace a computation by its value.
pub const literal = struct {
    pub const Attrs = struct { data: []const u8 };
    pub const weight = 0;

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, _: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        @memcpy(v.bytes, attrs.data);
        if (Value.isFloat(node.shape.dtype())) {
            _ = try v.trackErrs(e.allocator);
            // Special values on purpose (ieee family, reduced programs).
            v.allow_inf = !v.allFinite();
        }
        return v;
    }

    pub fn lower(_: Attrs, _: *const Node, _: []const Tensor) Tensor {
        unreachable;
    }

    pub fn emit(_: Attrs, _: *const Node, _: *Writer, _: []const []const u8) Writer.Error!void {
        unreachable;
    }
};

/// `Tensor.constantTensor`: data embedded in the program.
pub const constant = struct {
    pub const Attrs = struct { data: []const u8 };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.any) orelse return null;
        const sh = g.randShape(dt, .{ .max_rank = 3 });
        const v = try Value.alloc(g.allocator, sh);
        k.gen.fillInput(v, .{ .seed = g.rng.int(u64), .dist = if (dt == .bool) .int_range else .dyadic, .lo = if (dt == .bool) 0 else -4, .hi = if (dt == .bool) 1 else 4 });
        return .{ .op = .{ .constant = .{ .data = v.bytes } }, .shape = sh };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, _: []const Value) Eval.Error!Value {
        const v = try e.alloc(node.shape);
        @memcpy(v.bytes, attrs.data);
        return v;
    }

    pub fn lower(attrs: Attrs, node: *const Node, _: []const Tensor) Tensor {
        return Tensor.constantTensor(node.shape, attrs.data);
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, _: []const []const u8) Writer.Error!void {
        try w.writeAll("zml.Tensor.constantTensor(");
        try k.printShapeLiteral(w, node.shape);
        try w.writeAll(", &[_]u8{ ");
        for (attrs.data, 0..) |b, i| {
            if (i > 0) try w.writeAll(", ");
            try w.print("{d}", .{b});
        }
        try w.writeAll(" })");
    }
};

/// `Tensor.scalar(value, dtype)`.
pub const scalar = struct {
    pub const Attrs = struct { value: i64 };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.any) orelse return null;
        var val = g.intIn(i64, -5, 5);
        if (dt == .bool) val = g.intIn(i64, 0, 1);
        if (k.Value.isInt(dt) and !dt.isSignedInt()) val = @intCast(@abs(val));
        return .{ .op = .{ .scalar = .{ .value = val } }, .shape = .init(.{}, dt) };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, _: []const Value) Eval.Error!Value {
        const v = try e.alloc(node.shape);
        if (Value.isFloat(node.shape.dtype())) v.setF(0, @floatFromInt(attrs.value)) else v.setI(0, attrs.value);
        return v;
    }

    pub fn lower(attrs: Attrs, node: *const Node, _: []const Tensor) Tensor {
        if (node.shape.dtype() == .bool) return Tensor.scalar(attrs.value != 0, .bool);
        return Tensor.scalar(attrs.value, node.shape.dtype());
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, _: []const []const u8) Writer.Error!void {
        if (node.shape.dtype() == .bool) return w.print("zml.Tensor.scalar({}, .bool)", .{attrs.value != 0});
        try w.print("zml.Tensor.scalar({d}, .{s})", .{ attrs.value, @tagName(node.shape.dtype()) });
    }
};

/// `Tensor.iota(shape, axis)`: documented to produce i32 for small axes.
pub const iota = struct {
    pub const Attrs = struct { axis: k.AxisRef, shape_dtype: k.DataType };
    /// Handles size-0 tensors.
    pub const empty_ok = true;
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.any) orelse return null;
        const sh = g.randShape(dt, .{ .min_rank = 1, .max_rank = 4 });
        if (sh.count() == 0) return error.Reject;
        const ax = g.rng.uintLessThan(usize, sh.rank());
        return .{
            .op = .{ .iota = .{ .axis = .random(g, sh, ax), .shape_dtype = dt } },
            .shape = sh.withDtype(.i32),
        };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, _: []const Value) Eval.Error!Value {
        const v = try e.alloc(node.shape);
        const ax = attrs.axis.resolve(node.shape);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        while (!it.done) : (i += 1) {
            v.setI(i, it.idx[ax]);
            _ = it.next();
        }
        return v;
    }

    pub fn lower(attrs: Attrs, node: *const Node, _: []const Tensor) Tensor {
        const sh = node.shape.withDtype(attrs.shape_dtype);
        return switch (attrs.axis.pass()) {
            inline else => |a| Tensor.iota(sh, a),
        };
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, _: []const []const u8) Writer.Error!void {
        try w.writeAll("zml.Tensor.iota(");
        try k.printShapeLiteral(w, node.shape.withDtype(attrs.shape_dtype));
        try w.writeAll(", ");
        try attrs.axis.print(w);
        try w.writeAll(")");
    }
};

/// `Tensor.arange(.{ .start, .end, .step }, dtype)` and `Tensor.linspace`.
pub const arange = struct {
    pub const Attrs = struct {
        start: i64,
        end: i64,
        step: i64,
        /// Use `linspace(.{ .start, .end, .steps = step }, dtype)` instead.
        linspace: bool,
    };
    pub const weight = 2;

    pub fn label(attrs: Attrs) []const u8 {
        return if (attrs.linspace) "linspace" else "arange";
    }

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const linspace = g.chance(0.3);
        try g.avoid(.linspace, linspace);
        const dt = g.randDtype(if (linspace) .float else .numeric) orelse return null;
        const unsigned = Value.isInt(dt) and !dt.isSignedInt();
        const start = if (unsigned) g.intIn(i64, 0, 5) else g.intIn(i64, -5, 5);
        if (linspace) {
            const end = start + g.intIn(i64, 1, 8);
            const steps = g.intIn(i64, 1, 9);
            return .{ .op = .{ .arange = .{ .start = start, .end = end, .step = steps, .linspace = true } }, .shape = .init(.{steps}, dt) };
        }
        const end = start + g.intIn(i64, 1, 12);
        const step = g.intIn(i64, 1, 4);
        const n = std.math.divCeil(i64, end - start, step) catch unreachable;
        return .{ .op = .{ .arange = .{ .start = start, .end = end, .step = step, .linspace = false } }, .shape = .init(.{n}, dt) };
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, _: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        const n: usize = @intCast(node.shape.dim(0));
        const dt = node.shape.dtype();
        if (attrs.linspace) {
            // Documented as `steps` values evenly spaced from start to end, inclusive.
            const start: f64 = @floatFromInt(attrs.start);
            const end: f64 = @floatFromInt(attrs.end);
            const denom: f64 = if (n > 1) @floatFromInt(n - 1) else 1;
            var exact = true;
            for (0..n) |i| exact = k.storeF(v, i, start + (end - start) * @as(f64, @floatFromInt(i)) / denom) and exact;
            if (!exact) v.err = 2 * k.roundErr(dt, @max(@abs(start), @abs(end)));
            return v;
        }
        for (0..n) |i| {
            const x = attrs.start + @as(i64, @intCast(i)) * attrs.step;
            if (Value.isFloat(dt)) {
                if (!k.storeF(v, i, @floatFromInt(x))) try e.reject();
            } else {
                if (x < Value.minInt(dt) or x > Value.maxInt(dt)) try e.reject();
                v.setI(i, x);
            }
        }
        return v;
    }

    pub fn lower(attrs: Attrs, node: *const Node, _: []const Tensor) Tensor {
        if (attrs.linspace) return Tensor.linspace(.{
            .start = @floatFromInt(attrs.start),
            .end = @floatFromInt(attrs.end),
            .steps = attrs.step,
        }, node.shape.dtype());
        return Tensor.arange(.{ .start = attrs.start, .end = attrs.end, .step = attrs.step }, node.shape.dtype());
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, _: []const []const u8) Writer.Error!void {
        if (attrs.linspace) return w.print("zml.Tensor.linspace(.{{ .start = {d}, .end = {d}, .steps = {d} }}, .{s})", .{ attrs.start, attrs.end, attrs.step, @tagName(node.shape.dtype()) });
        try w.print("zml.Tensor.arange(.{{ .start = {d}, .end = {d}, .step = {d} }}, .{s})", .{ attrs.start, attrs.end, attrs.step, @tagName(node.shape.dtype()) });
    }
};
