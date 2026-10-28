//! Sharding: Tensor.withPartitioning constraints in programs compiled on the `data` x
//! `model` mesh. Partitioning never changes values, so the reference is the identity, and
//! any difference is a partitioner or ZML sharding bug.

const std = @import("std");

const zml = @import("zml");

const k = @import("../kit.zig");
const Shape = k.Shape;
const Tensor = k.Tensor;
const Value = k.Value;
const Gen = k.Gen;
const Eval = k.Eval;
const Node = k.Node;
const Writer = k.Writer;
const TagId = k.TagId;

/// Partitioned axes must divide evenly over any split of the 4 CPU devices.
pub const shard_multiple = 4;
/// Devices of the CPU mesh, the number of partitions of manual computations.
pub const n_devices = 4;

pub const partition = struct {
    pub const Spec = enum { data, model, replicated, open };
    pub const Attrs = struct { axis: TagId, spec: Spec, axis2: TagId = .none, spec2: Spec = .replicated };
    pub const weight = 4;

    pub fn pickAxis(g: *Gen, sh: Shape, spec: Spec, not: ?usize) ?usize {
        var cands: [8]usize = undefined;
        var n: usize = 0;
        for (0..sh.rank()) |ax| {
            if (not != null and not.? == ax) continue;
            // The lowering dispatches over the random tag pool only.
            if (@intFromEnum(TagId.of(sh.tag(ax))) >= k.ir.random_tag_count) continue;
            const needs_div = spec == .data or spec == .model;
            if (needs_div and @mod(sh.dim(ax), shard_multiple) != 0) continue;
            cands[n] = ax;
            n += 1;
        }
        if (n == 0) return null;
        return cands[g.rng.uintLessThan(usize, n)];
    }

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        // withPartitioning panics without a registered sharding.
        if (!g.prog.sharded) return error.Reject;
        var x = try g.operand(.{ .min_rank = 1, .tagged = true });
        var sh = g.shapeOf(x);
        var spec = g.rng.enumValue(Spec);
        // Sharded intermediates hit the reduced axis bug as soon as an op shrinks them.
        if (g.cfg.avoid.contains(.sharded_reduced_axis) and (spec == .data or spec == .model)) spec = if (g.chance(0.5)) .replicated else .open;
        const ax = pickAxis(g, sh, spec, null) orelse blk: {
            if (@intFromEnum(TagId.of(sh.tag(0))) >= k.ir.random_tag_count) return error.Reject;
            // Make an axis shardable.
            var bigger = sh.setDim(0, shard_multiple * g.intIn(i64, 1, 2));
            if (bigger.count() > g.cfg.max_elems) bigger = sh.setDim(0, shard_multiple);
            x = try g.freshInput(bigger);
            sh = bigger;
            break :blk 0;
        };
        var a: Attrs = .{ .axis = TagId.of(sh.tag(ax)), .spec = spec };
        if (spec == .data and g.chance(0.4)) {
            // A second axis on the other mesh axis.
            if (pickAxis(g, sh, .model, ax)) |ax2| {
                a.axis2 = TagId.of(sh.tag(ax2));
                a.spec2 = .model;
            }
        }
        return .{ .op = .{ .partition = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = sh };
    }

    pub fn eval(e: *Eval, _: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var v = try ins[0].dupe(e.allocator);
        v.shape = node.shape;
        return v;
    }

    /// `.{ .<tag> = .<spec> }` with comptime fields, as withPartitioning needs.
    fn Spec1(comptime t: usize, comptime spec: Spec) type {
        const lit = specLiteral(spec);
        return @Struct(.auto, null, &.{k.ir.tag_names[t]}, &.{@EnumLiteral()}, &.{.{ .@"comptime" = true, .default_value_ptr = &lit }});
    }

    fn Spec2(comptime t: usize, comptime spec: Spec, comptime t2: usize, comptime spec2: Spec) type {
        const lit = specLiteral(spec);
        const lit2 = specLiteral(spec2);
        return @Struct(.auto, null, &.{ k.ir.tag_names[t], k.ir.tag_names[t2] }, &.{ @EnumLiteral(), @EnumLiteral() }, &.{
            .{ .@"comptime" = true, .default_value_ptr = &lit },
            .{ .@"comptime" = true, .default_value_ptr = &lit2 },
        });
    }

    fn specLiteral(comptime spec: Spec) @EnumLiteral() {
        return switch (spec) {
            .data => .data,
            .model => .model,
            .replicated => .replicated,
            .open => .open,
        };
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        @setEvalBranchQuota(100_000);
        const x = ins[0];
        inline for (0..k.ir.random_tag_count) |t| {
            if (@intFromEnum(a.axis) == t) {
                inline for (std.meta.fields(Spec)) |sf| {
                    const spec: Spec = @enumFromInt(sf.value);
                    if (a.spec == spec) {
                        if (a.axis2 == .none) return x.withPartitioning(Spec1(t, spec){});
                        inline for (0..k.ir.random_tag_count) |t2| {
                            if (t2 != t and @intFromEnum(a.axis2) == t2) return x.withPartitioning(Spec2(t, spec, t2, .model){});
                        }
                    }
                }
            }
        }
        unreachable;
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        try w.print("{s}.withPartitioning(.{{ .{s} = .{s}", .{ ins[0], a.axis.name(), @tagName(a.spec) });
        if (a.axis2 != .none) try w.print(", .{s} = .{s}", .{ a.axis2.name(), @tagName(a.spec2) });
        try w.writeAll(" })");
    }
};

/// The partitioning applied to arguments of sharded programs: the first tagged axis whose
/// dim is a multiple of 4 goes on `data`.
pub fn argumentShape(sh: Shape) Shape {
    var res = sh;
    // KnownBug.sharded_subbyte: always avoided for arguments, the runner can't see the config.
    if (sh.dtype().bitSizeOf() < 8) return res;
    // KnownBug.sharded_empty, likewise.
    if (sh.count() == 0) return res;
    for (0..sh.rank()) |ax| {
        if (sh.tag(ax) != Shape.TagUnknown and sh.dim(ax) > 0 and @mod(sh.dim(ax), shard_multiple) == 0) {
            res = res.withDefaultPartitioning();
            res._partitioning.set(ax, .{ .axis = Shape.toTag(.data) });
            return res;
        }
    }
    return res;
}

/// zml.ops.manualComputation with an elementwise body on a tensor partitioned along
/// `data`: running the body on each shard must equal running it on the whole tensor.
/// The collective bodies return predicates that hold on every element:
/// `all_reduce_bounds` checks allReduce(max) >= x >= allReduce(min) on each shard,
/// `partition_ids` that the partition ids of all devices are 0..n-1 (their allReduce sum
/// is n(n-1)/2 and their max n-1).
pub const manual = struct {
    pub const Body = enum { negate, abs, relu, double, square, all_reduce_bounds, partition_ids };

    fn isPredicate(body: Body) bool {
        return body == .all_reduce_bounds or body == .partition_ids;
    }
    pub const Attrs = struct {
        body: Body,
        axis: TagId,
        /// Constrain the result back to replicated, which steers around
        /// KnownBug.sharded_reduced_axis.
        replicate_result: bool = false,
    };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        if (!g.prog.sharded) return error.Reject;
        const body = g.rng.enumValue(Body);
        try g.avoid(.gspmd_partition_id, body == .partition_ids and @import("../runner.zig").gspmd);
        try g.avoid(.gspmd_mesh2d_manual, @import("../runner.zig").gspmd and @import("../runner.zig").mesh2d);
        var x = try g.operand(.{ .min_rank = 1, .tagged = true, .class = if (body == .abs) .signed else .numeric });
        const xop = g.prog.nodes.items[x].op;
        try g.avoid(.gspmd_pool_manual, @import("../runner.zig").gspmd and (xop == .pool or xop == .reduce_window));
        var sh = g.shapeOf(x);
        const ax = partition.pickAxis(g, sh, .data, null) orelse blk: {
            if (@intFromEnum(TagId.of(sh.tag(0))) >= k.ir.random_tag_count) return error.Reject;
            sh = sh.setDim(0, shard_multiple);
            if (sh.count() > g.cfg.max_elems) return error.Reject;
            x = try g.freshInput(sh);
            break :blk 0;
        };
        const out = if (isPredicate(body)) sh.withDtype(.bool) else sh;
        return .{ .op = .{ .manual = .{ .body = body, .axis = TagId.of(sh.tag(ax)), .replicate_result = g.cfg.avoid.contains(.sharded_reduced_axis) } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const U = @import("elementwise.zig").unary;
        const B = @import("elementwise.zig").binary;
        return switch (a.body) {
            .all_reduce_bounds, .partition_ids => {
                const v = try e.alloc(node.shape);
                @memset(v.bytes, 1);
                return v;
            },
            .negate => U.eval(e, .{ .kind = .negate }, node, ins),
            .abs => U.eval(e, .{ .kind = .abs }, node, ins),
            .relu => U.eval(e, .{ .kind = .relu }, node, ins),
            .double => B.eval(e, .{ .kind = .add, .bcast = .same }, node, &.{ ins[0], ins[0] }),
            .square => B.eval(e, .{ .kind = .mul, .bcast = .same }, node, &.{ ins[0], ins[0] }),
        };
    }

    pub fn lower(a: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const x = partition.lower(.{ .axis = a.axis, .spec = .data }, node, ins);
        const Ctx = struct {
            x: Tensor,
            body: Body,
            fn run(c: @This(), out: Shape) Tensor {
                const r = switch (c.body) {
                    .all_reduce_bounds => blk: {
                        const hi = zml.ops.allReduce(c.x, Tensor.maximum);
                        const lo = zml.ops.allReduce(c.x, Tensor.minimum);
                        break :blk hi.cmp(.GE, c.x).logical(.AND, c.x.cmp(.GE, lo));
                    },
                    .partition_ids => blk: {
                        const id = zml.ops.partitionId().convert(.i32);
                        const sum = zml.ops.allReduce(id, Tensor.add);
                        const max = zml.ops.allReduce(id, Tensor.maximum);
                        const ok = sum.cmp(.EQ, Tensor.scalar(n_devices * (n_devices - 1) / 2, .i32)).logical(.AND, max.cmp(.EQ, Tensor.scalar(n_devices - 1, .i32)));
                        break :blk ok.broad(out);
                    },
                    .negate => c.x.negate(),
                    .abs => c.x.abs(),
                    .relu => c.x.relu(),
                    .double => c.x.add(c.x),
                    .square => c.x.mul(c.x),
                };
                return r.reshape(out);
            }
        };
        const out = if (isPredicate(a.body)) x.shape().withDtype(.bool) else x.shape();
        const r = zml.ops.manualComputation(Ctx.run, .{ .x = x, .body = a.body }, out);
        if (a.replicate_result) return partition.lower(.{ .axis = a.axis, .spec = .replicated }, node, &.{r});
        return r;
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const op = switch (a.body) {
            .negate => "c.x.negate()",
            .abs => "c.x.abs()",
            .relu => "c.x.relu()",
            .double => "c.x.add(c.x)",
            .square => "c.x.mul(c.x)",
            .all_reduce_bounds => "zml.ops.allReduce(c.x, zml.Tensor.maximum).cmp(.GE, c.x).logical(.AND, c.x.cmp(.GE, zml.ops.allReduce(c.x, zml.Tensor.minimum)))",
            .partition_ids => std.fmt.comptimePrint("zml.ops.allReduce(zml.ops.partitionId().convert(.i32), zml.Tensor.add).cmp(.EQ, zml.Tensor.scalar({d}, .i32)).logical(.AND, zml.ops.allReduce(zml.ops.partitionId().convert(.i32), zml.Tensor.maximum).cmp(.EQ, zml.Tensor.scalar({d}, .i32))).broad(out)", .{ n_devices * (n_devices - 1) / 2, n_devices - 1 }),
        };
        try w.print("blk: {{\n                const p = {s}.withPartitioning(.{{ .{s} = .data }});\n", .{ ins[0], a.axis.name() });
        const out = if (isPredicate(a.body)) "p.shape().withDtype(.bool)" else "p.shape()";
        try w.print("                const r = zml.ops.manualComputation(struct {{\n                    x: zml.Tensor,\n                    fn run(c: @This(), out: zml.Shape) zml.Tensor {{\n                        _ = &c;\n                        return {s}.reshape(out);\n                    }}\n                }}.run, .{{ .x = p }}, {s});\n", .{ op, out });
        if (a.replicate_result) {
            try w.print("                break :blk r.withPartitioning(.{{ .{s} = .replicated }});\n            }}", .{a.axis.name()});
        } else {
            try w.writeAll("                break :blk r;\n            }");
        }
    }
};
