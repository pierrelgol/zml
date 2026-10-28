//! Lowers a program by calling the ZML API inside a traced function, checking after every
//! op that ZML inferred the shape, tags and dtype the generator predicted.

const std = @import("std");

const zml = @import("zml");
const Shape = zml.Shape;
const Tensor = zml.Tensor;

const ir = @import("ir.zig");
const ops = @import("ops.zig");

pub const ShapeMismatch = struct {
    node: u32,
    expected: Shape,
    actual: Shape,
    /// Dims or dtype differ, not only tags.
    severe: bool,
};

pub const Ctx = struct {
    allocator: std.mem.Allocator,
    prog: *const ir.Program,
    /// Return every node instead of the program outputs, to locate the first divergence.
    all_nodes: bool = false,
    mismatches: std.ArrayList(ShapeMismatch) = .empty,
    /// Set when a severe mismatch stopped the lowering; `returned` then lists what the
    /// executable actually returns.
    aborted: bool = false,
    returned: std.ArrayList(u32) = .empty,
    /// The tensor of every node, filled as nodes are lowered. Region nodes are overwritten
    /// each time their region is traced.
    tensors: []Tensor = &.{},
    /// Tensors bound to the params of the regions being lowered, innermost last.
    bindings: std.ArrayList([]const Tensor) = .empty,
};

/// The node being lowered, for the panic handler.
pub threadlocal var current_node: ?u32 = null;
pub threadlocal var current_prog: ?*const ir.Program = null;

/// The lowering context, passed out of band: zml.meta would otherwise walk the whole
/// program type looking for tensors.
pub threadlocal var current_ctx: ?*Ctx = null;

pub fn forward(args: []const Tensor) []Tensor {
    return forwardImpl(current_ctx.?, args) catch @panic("OOM");
}

/// The tensor bound to param `index` of the innermost region.
pub fn binding(index: u8) Tensor {
    const ctx = current_ctx.?;
    return ctx.bindings.items[ctx.bindings.items.len - 1][index];
}

/// Lowers the nodes of region `scope` in [start, end) with its params bound.
pub fn lowerRegion(ctx: *Ctx, start: u32, end: u32, scope: u32, bindings: []const Tensor) void {
    ctx.bindings.append(ctx.allocator, bindings) catch @panic("OOM");
    defer _ = ctx.bindings.pop();
    lowerScope(ctx, start, end, scope, &.{}) catch @panic("OOM");
}

fn lowerScope(ctx: *Ctx, start: u32, end: u32, scope: u32, args: []const Tensor) !void {
    const nodes = ctx.prog.nodes.items;
    var next_arg: usize = 0;
    for (start..end) |i| {
        const n = &nodes[i];
        if (ops.isArgument(n.op)) {
            tensors(ctx)[i] = args[next_arg];
            next_arg += 1;
            continue;
        }
        if (n.scope != scope) continue;
        if (ctx.aborted) return;
        const saved = current_node;
        current_node = @intCast(i);
        defer current_node = saved;
        var ins: [ir.max_operands]Tensor = undefined;
        for (n.inputs.slice(), 0..) |in, j| ins[j] = tensors(ctx)[in];
        var t = switch (n.op) {
            inline else => |_, tag| blk: {
                const I = @field(ops.impls, @tagName(tag));
                if (comptime @hasDecl(I, "lowerCtl")) break :blk I.lowerCtl(ctx, ctx.tensors, @intCast(i), ins[0..n.inputs.len]);
                break :blk ops.lower(n, ins[0..n.inputs.len]);
            },
        };
        const actual = t.shape();
        const same_dims = actual.eql(n.shape);
        const same_tags = same_dims and std.mem.eql(Shape.Tag, actual.tags(), n.shape.tags());
        if (!same_tags) {
            try ctx.mismatches.append(ctx.allocator, .{ .node = @intCast(i), .expected = n.shape, .actual = actual, .severe = !same_dims });
            if (!same_dims) {
                ctx.aborted = true;
                try ctx.returned.append(ctx.allocator, @intCast(i));
                tensors(ctx)[i] = t;
                return;
            }
            // Keep going with the predicted tags so later ops are checked independently.
            t._shape._tags = n.shape._tags;
        }
        tensors(ctx)[i] = t;
    }
}

fn tensors(ctx: *Ctx) []Tensor {
    return ctx.tensors;
}

fn forwardImpl(ctx: *Ctx, args: []const Tensor) ![]Tensor {
    const nodes = ctx.prog.nodes.items;
    ctx.tensors = try ctx.allocator.alloc(Tensor, nodes.len);
    current_prog = ctx.prog;
    try lowerScope(ctx, 0, @intCast(nodes.len), 0, args);
    if (ctx.aborted) {
        const out = try ctx.allocator.alloc(Tensor, 1);
        out[0] = ctx.tensors[ctx.returned.items[0]];
        return out;
    }
    if (ctx.all_nodes) {
        // reuseBuffer returns its input tensor: returning both would donate twice.
        outer: for (nodes, 0..) |n, i| if (n.scope == 0) {
            for (nodes) |m| if (m.op == .memory and m.op.memory.kind == .reuse_buffer and m.inputs.slice()[0] == i) continue :outer;
            try ctx.returned.append(ctx.allocator, @intCast(i));
        };
    } else {
        try ctx.returned.appendSlice(ctx.allocator, ctx.prog.outputs.items);
    }
    const out = try ctx.allocator.alloc(Tensor, ctx.returned.items.len);
    for (ctx.returned.items, out) |id, *o| o.* = ctx.tensors[id];
    return out;
}

/// A structured entry point, so that zml.meta argument flattening (nested structs, non
/// tensor fields, optionals, arrays, slices) and Bufferized results get exercised.
/// The program arguments are, in order: x, inner.y, opt, arr[0], arr[1], then rest.
pub const Nested = struct {
    x: Tensor,
    inner: struct { y: Tensor, scale: u32 = 3 },
    opt: ?Tensor,
    arr: [2]Tensor,
};
pub const nested_slots = 5;

pub const NestedOut = struct { first: Tensor, more: []Tensor };

pub fn forwardNested(a: Nested, rest: []const Tensor) NestedOut {
    const ctx = current_ctx.?;
    const all = ctx.allocator.alloc(Tensor, nested_slots + rest.len) catch @panic("OOM");
    all[0] = a.x;
    all[1] = a.inner.y;
    all[2] = a.opt.?;
    all[3] = a.arr[0];
    all[4] = a.arr[1];
    @memcpy(all[nested_slots..], rest);
    std.debug.assert(a.inner.scale == 3);
    const outs = forward(all);
    return .{ .first = outs[0], .more = outs[1..] };
}
