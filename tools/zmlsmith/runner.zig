//! Compiles and runs a program on a platform, and compares against the interpreter.

const std = @import("std");

const zml = @import("zml");
const Shape = zml.Shape;
const Tensor = zml.Tensor;

const ir = @import("ir.zig");
const lower = @import("lower.zig");
const ops = @import("ops.zig");
const Value = @import("Value.zig");

pub const Kind = enum {
    ok,
    /// ZML returned the predicted dims and dtype but different tags.
    tags,
    /// ZML returned different dims or dtype.
    shape,
    /// PJRT rejected the program.
    compile_error,
    /// The executable output shapes differ from the traced ones.
    output_shape,
    /// The device computed different values.
    value,
    /// Running the same executable twice on the same inputs gave different results.
    nondeterministic,
};

pub const Finding = struct {
    kind: Kind,
    node: u32 = 0,
    /// Stable description used to deduplicate findings: kind, op and the main attribute.
    signature: []const u8 = "",
    detail: []const u8 = "",
};

pub const Report = struct {
    findings: std.ArrayList(Finding) = .empty,

    pub fn worst(self: Report) Kind {
        var w: Kind = .ok;
        for (self.findings.items) |f| {
            if (@intFromEnum(f.kind) > @intFromEnum(w) or w == .ok) w = f.kind;
        }
        return w;
    }
};

/// A short label for a node: the family, and the `kind` attribute when there is one.
pub fn opLabel(allocator: std.mem.Allocator, n: *const ir.Node) ![]const u8 {
    switch (n.op) {
        inline else => |attrs, tag| {
            const A = @TypeOf(attrs);
            const I = @field(ops.impls, @tagName(tag));
            if (@hasDecl(I, "label")) return std.fmt.allocPrint(allocator, "{s}.{s}", .{ @tagName(tag), I.label(attrs) });
            if (A != void and @hasField(A, "kind")) {
                return std.fmt.allocPrint(allocator, "{s}.{s}", .{ @tagName(tag), @tagName(attrs.kind) });
            }
            return std.fmt.allocPrint(allocator, "{s}", .{@tagName(tag)});
        },
    }
}

/// Evaluates every node with the reference interpreter, without rejecting anything.
pub fn reference(allocator: std.mem.Allocator, prog: *const ir.Program) ![]Value {
    const values = try allocator.alloc(Value, prog.nodes.items.len);
    var e: ops.Eval = .{ .allocator = allocator, .strict = false, .prog = prog };
    for (prog.nodes.items, 0..) |*n, i| {
        // Region nodes are evaluated by the control flow op owning them.
        if (n.scope != 0) {
            values[i] = try Value.alloc(allocator, n.shape);
            continue;
        }
        var ins: [ir.max_operands]Value = undefined;
        for (n.inputs.slice(), 0..) |in, j| ins[j] = values[in];
        values[i] = ops.eval(&e, n, ins[0..n.inputs.len]) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Reject => unreachable,
        };
    }
    return values;
}

/// The `data` x `model` mesh used by sharded programs, registered once per platform.
pub var mesh: ?zml.Sharding = null;
/// `--mesh2d`: a two level physical mesh, so `data` and `model` map to distinct physical
/// axes. The default CPU mesh is one bus: `model` collapses onto `data`.
pub var mesh2d = false;
/// `--gspmd`: partition with GSPMD instead of the default Shardy, to compare the two.
pub var gspmd = false;

var compile_msg_buf: [96]u8 = undefined;
var compile_msg_len: usize = 0;

/// Called by the log function with the message of a failed PJRT compile.
pub fn setCompileMessage(msg: []const u8) void {
    // Keep words, drop values, as crash signatures do.
    var n: usize = 0;
    var i: usize = 0;
    while (i < msg.len and n < compile_msg_buf.len) {
        const c = msg[i];
        if (c == '\n' or c == '%' or c == '{') break;
        if (std.ascii.isDigit(c)) {
            while (i < msg.len and (std.ascii.isAlphanumeric(msg[i]) or msg[i] == '.')) i += 1;
            compile_msg_buf[n] = '#';
            n += 1;
            continue;
        }
        compile_msg_buf[n] = c;
        n += 1;
        i += 1;
    }
    compile_msg_len = n;
}

/// Two halves of the devices on a bus, each half a ring.
pub fn twoLevelMesh(allocator: std.mem.Allocator, target: zml.Target, devices: []const zml.platform.Device) anyerror!zml.Sharding.PhysicalMesh {
    if (devices.len < 4 or devices.len % 2 != 0) return zml.Sharding.PhysicalMesh.auto(allocator, target, devices);
    const half = devices.len / 2;
    const groups = try allocator.alloc(zml.Sharding.PhysicalNode, 2);
    for (groups, 0..) |*grp, gi| {
        const leaves = try allocator.alloc(zml.Sharding.PhysicalNode, half);
        for (leaves, 0..) |*l, j| l.* = .device(devices[gi * half + j]);
        grp.* = .axis(.link, .{ .ring = .linear }, leaves);
    }
    return zml.Sharding.PhysicalMesh.fromTree(allocator, target, .axis(.bus, .tree, groups));
}

const sharding_ops = @import("ops/sharding.zig");

pub const Options = struct {
    all_nodes: bool = false,
    /// Skip execution, only trace and compile.
    compile_only: bool = false,
};

/// Runs the program; on a value mismatch, runs it again returning every node and reports
/// the earliest diverging node first, so findings are attributed to the op at fault.
pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    prog: *const ir.Program,
    opts: Options,
) !Report {
    var report = try runOnce(allocator, io, platform, prog, opts);
    if (opts.all_nodes or report.worst() != .value) return report;
    var all = try runOnce(allocator, io, platform, prog, .{ .all_nodes = true });
    var earliest: ?Finding = null;
    for (all.findings.items) |f| {
        if (f.kind != .value) continue;
        if (earliest == null or f.node < earliest.?.node) earliest = f;
    }
    if (earliest) |f| {
        var e = f;
        e.detail = try std.fmt.allocPrint(allocator, "earliest divergence when returning every node: {s}", .{f.detail});
        // Put it first: the supervisor and reducer key on the most severe, earliest finding.
        var res: Report = .{};
        try res.findings.append(allocator, e);
        for (report.findings.items) |g| if (g.kind != .value) try res.findings.append(allocator, g);
        for (report.findings.items) |g| if (g.kind == .value) try res.findings.append(allocator, .{ .kind = .value, .node = g.node, .signature = try std.fmt.allocPrint(allocator, "downstream:{s}", .{g.signature}), .detail = g.detail });
        _ = &all;
        report = res;
    }
    return report;
}

fn runOnce(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    prog: *const ir.Program,
    opts: Options,
) !Report {
    var report: Report = .{};
    ops.target = platform.target;
    const values = try reference(allocator, prog);

    const arg_ids = try prog.argumentNodes(allocator);
    const sharded = prog.sharded and mesh != null;
    const arg_shapes = try allocator.alloc(Shape, arg_ids.len);
    for (arg_ids, arg_shapes) |id, *sh| {
        sh.* = prog.nodes.items[id].shape;
        if (sharded and prog.shard_args) sh.* = sharding_ops.argumentShape(sh.*);
    }
    const arg_tensors = try allocator.alloc(Tensor, arg_ids.len);
    for (arg_shapes, arg_tensors) |sh, *t| t.* = .fromShape(sh);

    var ctx: lower.Ctx = .{ .allocator = allocator, .prog = prog, .all_nodes = opts.all_nodes };
    lower.current_ctx = &ctx;
    defer lower.current_ctx = null;
    const shardings: []const zml.Sharding = if (sharded) &.{mesh.?} else &.{};
    const compile_opts: zml.Compiler.Options = .{ .program_name = "zmlsmith", .shardings = shardings, .partitioner = if (gspmd) .gspmd else null };
    // One program in three goes through the structured entry point.
    const nested = prog.seed % 3 == 1;
    // Padding arguments for the structured entry point when the program has fewer.
    var padded = arg_tensors;
    if (nested and arg_tensors.len < lower.nested_slots) {
        padded = try allocator.alloc(Tensor, lower.nested_slots);
        @memcpy(padded[0..arg_tensors.len], arg_tensors);
        for (padded[arg_tensors.len..]) |*t| t.* = .init(.{}, .i32);
    }
    const nested_args: lower.Nested = if (nested) .{
        .x = padded[0],
        .inner = .{ .y = padded[1] },
        .opt = padded[2],
        .arr = .{ padded[3], padded[4] },
    } else undefined;
    const exe_or_err = if (nested)
        platform.compileFn(allocator, io, lower.forwardNested, .{ nested_args, padded[lower.nested_slots..] }, compile_opts)
    else
        platform.compileFn(allocator, io, lower.forward, .{arg_tensors}, compile_opts);
    var exe = exe_or_err catch |err| {
        try addShapeFindings(allocator, &report, prog, &ctx);
        try report.findings.append(allocator, .{
            .kind = .compile_error,
            .signature = blk: {
                defer compile_msg_len = 0;
                if (compile_msg_len == 0) break :blk try std.fmt.allocPrint(allocator, "compile_error:{t}", .{err});
                break :blk try std.fmt.allocPrint(allocator, "compile_error:{t}:{s}", .{ err, std.mem.trim(u8, compile_msg_buf[0..compile_msg_len], " ") });
            },
            .detail = try std.fmt.allocPrint(allocator, "{t}", .{err}),
        });
        return report;
    };
    defer exe.deinit();

    try addShapeFindings(allocator, &report, prog, &ctx);
    if (ctx.aborted) return report;

    for (ctx.returned.items, exe.output_shapes, 0..) |id, sh, i| {
        const expected = prog.nodes.items[id].shape;
        if (!sh.eql(expected)) {
            try report.findings.append(allocator, .{
                .kind = .output_shape,
                .node = id,
                .signature = try std.fmt.allocPrint(allocator, "output_shape:{s}", .{try opLabel(allocator, &prog.nodes.items[id])}),
                .detail = try std.fmt.allocPrint(allocator, "output {d} (%{d}): executable returns {f}, traced {f}", .{ i, id, sh, expected }),
            });
        }
    }
    if (report.findings.items.len > 0 and report.worst() == .output_shape) return report;
    if (opts.compile_only) return report;

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    const n_bufs = if (nested) padded.len else arg_ids.len;
    const bufs = try allocator.alloc(zml.Buffer, n_bufs);
    for (arg_ids, arg_shapes, bufs[0..arg_ids.len]) |id, sh, *b| b.* = try .fromBytes(io, platform, sh, if (sharded) mesh.? else .replicated, values[id].bytes);
    for (bufs[arg_ids.len..]) |*b| b.* = try .fromBytes(io, platform, .init(.{}, .i32), if (sharded) mesh.? else .replicated, &.{ 0, 0, 0, 0 });
    defer for (bufs) |*b| b.deinit();
    var nested_bufs: zml.Bufferized(lower.Nested) = undefined;
    if (nested) {
        nested_bufs.x = bufs[0];
        nested_bufs.inner.y = bufs[1];
        nested_bufs.opt = bufs[2];
        nested_bufs.arr = .{ bufs[3], bufs[4] };
        args.set(.{ nested_bufs, bufs[lower.nested_slots..] });
    } else args.set(.{bufs});
    exe.callOpts(io, args, &results, .{ .wait = true });

    const outs = try allocator.alloc(zml.Buffer, ctx.returned.items.len);
    if (nested) {
        var out: zml.Bufferized(lower.NestedOut) = .{ .first = undefined, .more = outs[1..] };
        results.fill(.{&out});
        outs[0] = out.first;
    } else results.fill(.{outs});
    defer for (outs) |*o| o.deinit();

    // A second run of the same executable must reproduce the first one bit for bit. Donated
    // arguments were consumed by the first run, so upload them again.
    const bufs2 = try allocator.alloc(zml.Buffer, n_bufs);
    for (arg_ids, arg_shapes, bufs2[0..arg_ids.len]) |id, sh, *b| b.* = try .fromBytes(io, platform, sh, if (sharded) mesh.? else .replicated, values[id].bytes);
    for (bufs2[arg_ids.len..]) |*b| b.* = try .fromBytes(io, platform, .init(.{}, .i32), if (sharded) mesh.? else .replicated, &.{ 0, 0, 0, 0 });
    defer for (bufs2) |*b| b.deinit();
    var args2 = try exe.args(allocator);
    defer args2.deinit(allocator);
    if (nested) {
        var nb: zml.Bufferized(lower.Nested) = undefined;
        nb.x = bufs2[0];
        nb.inner.y = bufs2[1];
        nb.opt = bufs2[2];
        nb.arr = .{ bufs2[3], bufs2[4] };
        args2.set(.{ nb, bufs2[lower.nested_slots..] });
    } else args2.set(.{bufs2});
    var results2 = try exe.results(allocator);
    defer results2.deinit(allocator);
    exe.callOpts(io, args2, &results2, .{ .wait = true });
    const outs2 = try allocator.alloc(zml.Buffer, ctx.returned.items.len);
    if (nested) {
        var out2: zml.Bufferized(lower.NestedOut) = .{ .first = undefined, .more = outs2[1..] };
        results2.fill(.{&out2});
        outs2[0] = out2.first;
    } else results2.fill(.{outs2});
    defer for (outs2) |*o| o.deinit();
    for (ctx.returned.items, outs, outs2, 0..) |id, *o1, *o2, i| {
        const s1 = try o1.toSliceAlloc(allocator, io);
        const s2 = try o2.toSliceAlloc(allocator, io);
        if (!std.mem.eql(u8, s1.constData(), s2.constData())) {
            try report.findings.append(allocator, .{
                .kind = .nondeterministic,
                .node = id,
                .signature = try std.fmt.allocPrint(allocator, "nondeterministic:{s}", .{try opLabel(allocator, &prog.nodes.items[id])}),
                .detail = try std.fmt.allocPrint(allocator, "output {d} (%{d}) differs between two runs", .{ i, id }),
            });
        }
    }

    for (ctx.returned.items, outs, 0..) |id, *o, i| {
        const got = try o.toSliceAlloc(allocator, io);
        const expected = values[id];
        if (try compare(allocator, expected, got.constData())) |msg| {
            try report.findings.append(allocator, .{
                .kind = .value,
                .node = id,
                .signature = try std.fmt.allocPrint(allocator, "value:{s}", .{try opLabel(allocator, &prog.nodes.items[id])}),
                .detail = try std.fmt.allocPrint(allocator, "output {d} (%{d}, {s}): {s}", .{ i, id, @tagName(expected.dtype()), msg }),
            });
        }
    }
    return report;
}

fn addShapeFindings(allocator: std.mem.Allocator, report: *Report, prog: *const ir.Program, ctx: *const lower.Ctx) !void {
    for (ctx.mismatches.items) |m| {
        const label = try opLabel(allocator, &prog.nodes.items[m.node]);
        try report.findings.append(allocator, .{
            .kind = if (m.severe) .shape else .tags,
            .node = m.node,
            .signature = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ if (m.severe) "shape" else "tags", label }),
            .detail = try std.fmt.allocPrint(allocator, "%{d} {s}: ZML returned {f}, expected {f}", .{ m.node, label, m.actual, m.expected }),
        });
    }
}

/// Returns a description of the first difference, or null when the values match.
pub fn compare(allocator: std.mem.Allocator, expected: Value, got_bytes: []const u8) !?[]const u8 {
    const got = try Value.alloc(allocator, expected.shape);
    @memcpy(got.bytes, got_bytes);
    const dt = expected.dtype();
    const n = expected.count();
    var bad: usize = 0;
    var first: ?usize = null;
    var worst: f64 = 0;
    for (0..n) |i| {
        const ok = if (Value.isFloat(dt)) blk: {
            const r = expected.getF(i);
            const a = got.getF(i);
            if (std.math.isNan(r)) break :blk std.math.isNan(a);
            if (!std.math.isFinite(r)) break :blk a == r;
            if (!std.math.isFinite(a)) break :blk false;
            const diff = @abs(a - r);
            worst = @max(worst, diff);
            if (expected.errs) |errs| {
                // Per element bounds: zero means bit exact, sign of zero included.
                if (errs[i] == 0) break :blk diff == 0 and std.math.signbit(a) == std.math.signbit(r);
                break :blk diff <= errs[i];
            }
            if (expected.err == 0) break :blk diff == 0;
            break :blk diff <= 2 * expected.err + 2 * Value.eps(dt) * @abs(r);
        } else if (dt == .bool)
            (expected.bytes[i] != 0) == (got.bytes[i] != 0)
        else
            expected.getI(i) == got.getI(i);
        if (!ok) {
            bad += 1;
            if (first == null) first = i;
        }
    }
    const i = first orelse return null;
    var w: std.Io.Writer.Allocating = .init(allocator);
    try w.writer.print("{d}/{d} elements differ, first at flat index {d}: expected ", .{ bad, n, i });
    try printElem(&w.writer, expected, i);
    try w.writer.writeAll(", got ");
    try printElem(&w.writer, got, i);
    if (Value.isFloat(dt)) try w.writer.print(" (error bound {e}, max abs diff {e})", .{ expected.errAt(i), worst });
    return w.written();
}

fn printElem(w: *std.Io.Writer, v: Value, i: usize) !void {
    if (Value.isFloat(v.dtype())) return w.print("{d}", .{v.getF(i)});
    try w.print("{d}", .{v.getI(i)});
}
