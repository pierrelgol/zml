//! Automatic test case reduction.
//!
//! A candidate is interesting when running it in a child process yields the same signature
//! as the original failure (crash in the same op with the same message, or the same kind
//! of mismatch on the same op). Passes, repeated until nothing shrinks:
//!
//! 1. drop program outputs,
//! 2. replace a computed node by a literal holding its reference value, which cuts its
//!    whole input cone,
//! 3. reroute an operand to an earlier node with the same shape,
//! 4. simplify argument data to zeros, ones or small integers,
//!
//! each followed by dead code elimination.

const std = @import("std");

const zml = @import("zml");

const child = @import("child.zig");
const ir = @import("ir.zig");
const ops = @import("ops.zig");
const runner = @import("runner.zig");
const Value = @import("Value.zig");

pub const Options = struct {
    timeout_s: i64 = 600,
    /// Directory of the candidate file, the working directory by default.
    tmp_dir: ?[]const u8 = null,
    verbose: bool = false,
    /// Signature to preserve; computed from the input program when null.
    signature: ?[]const u8 = null,
    max_candidates: usize = 400,
    /// Mutation the failure was found under, passed on to each candidate.
    mutate: ?[]const u8 = null,
};

pub const Result = struct {
    program: ir.Program,
    signature: []const u8,
    candidates: usize,
};

const Oracle = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    exe: []const u8,
    tmp_path: []const u8,
    opts: Options,
    signature: []const u8 = "",
    candidates: usize = 0,

    fn argv(o: *const Oracle) []const []const u8 {
        // Repeating --no-trace is harmless: it fills the slots of the flags not set.
        const mesh: []const u8 = if (runner.mesh2d) "--mesh2d" else "--no-trace";
        const part: []const u8 = if (runner.gspmd) "--gspmd" else "--no-trace";
        if (o.opts.mutate) |m| return o.allocator.dupe([]const u8, &.{ o.exe, "repro", o.tmp_path, "--no-trace", mesh, part, "--mutate", m }) catch @panic("OOM");
        return o.allocator.dupe([]const u8, &.{ o.exe, "repro", o.tmp_path, "--no-trace", mesh, part }) catch @panic("OOM");
    }

    fn outcomeOf(o: *Oracle, prog: *const ir.Program) !child.Outcome {
        var w: std.Io.Writer.Allocating = .init(o.allocator);
        try prog.format(&w.writer);
        try std.Io.Dir.cwd().writeFile(o.io, .{ .sub_path = o.tmp_path, .data = w.written() });
        o.candidates += 1;
        return child.run(o.allocator, o.io, o.argv(), o.opts.timeout_s);
    }

    fn signatureOf(o: *Oracle, prog: *const ir.Program) ![]const u8 {
        var w: std.Io.Writer.Allocating = .init(o.allocator);
        try prog.format(&w.writer);
        try std.Io.Dir.cwd().writeFile(o.io, .{ .sub_path = o.tmp_path, .data = w.written() });
        o.candidates += 1;
        const out = try child.run(o.allocator, o.io, o.argv(), o.opts.timeout_s);
        return out.signature;
    }

    fn interesting(o: *Oracle, prog: *const ir.Program) !bool {
        if (o.candidates >= o.opts.max_candidates) return false;
        const sig = try o.signatureOf(prog);
        const ok = std.mem.eql(u8, sig, o.signature);
        if (o.opts.verbose) std.debug.print("reduce: {d} nodes -> {s}{s}\n", .{ prog.nodes.items.len, sig, if (ok) " (kept)" else "" });
        return ok;
    }
};

/// Removes nodes not reachable from the outputs and renumbers the rest.
pub fn deadCodeElim(allocator: std.mem.Allocator, prog: *const ir.Program) !ir.Program {
    const n = prog.nodes.items.len;
    const live = try allocator.alloc(bool, n);
    @memset(live, false);
    for (prog.outputs.items) |o| live[o] = true;
    var i = n;
    while (i > 0) {
        i -= 1;
        if (!live[i]) continue;
        for (prog.nodes.items[i].inputs.slice()) |in| live[in] = true;
        // A live control flow op keeps the params of its regions (they are positional)
        // and the region nodes its results depend on.
        switch (prog.nodes.items[i].op) {
            inline else => |attrs, tag| {
                const I = @field(ops.impls, @tagName(tag));
                if (comptime @hasDecl(I, "regions")) {
                    for (I.regions(attrs)) |r| for (r[0]..r[1]) |j| {
                        const nj = prog.nodes.items[j];
                        if (nj.op == .param and nj.scope == r[0] + 1) live[j] = true;
                    };
                    for (I.roots(attrs)) |root| live[root] = true;
                }
            },
        }
    }
    const remap = try allocator.alloc(u32, n);
    var res: ir.Program = .{ .seed = prog.seed, .sharded = prog.sharded, .shard_args = prog.shard_args };
    for (prog.nodes.items, 0..) |node, j| {
        if (!live[j]) continue;
        // Assigned first: a region's first node maps its own scope id through it.
        remap[j] = @intCast(res.nodes.items.len);
        var nn = node;
        for (nn.inputs.mut()) |*in| in.* = remap[in.*];
        // Region ids are the index of the region's first node plus one.
        if (nn.scope != 0) nn.scope = remap[nn.scope - 1] + 1;
        switch (nn.op) {
            inline else => |*attrs, tag| {
                const I = @field(ops.impls, @tagName(tag));
                if (comptime @hasDecl(I, "remap")) I.remap(attrs, remap);
            },
        }
        _ = try res.add(allocator, nn);
    }
    for (prog.outputs.items) |o| try res.outputs.append(allocator, remap[o]);
    return res;
}

pub fn reduce(allocator: std.mem.Allocator, io: std.Io, prog_in: *const ir.Program, opts: Options) !Result {
    var oracle: Oracle = .{
        .allocator = allocator,
        .io = io,
        .exe = try child.selfExe(allocator, io),
        .tmp_path = if (opts.tmp_dir) |d| try std.fmt.allocPrint(allocator, "{s}/.reduce-{d}.zs", .{ d, prog_in.seed }) else try std.fmt.allocPrint(allocator, ".zmlsmith-reduce-{d}.zs", .{prog_in.seed}),
        .opts = opts,
    };
    defer std.Io.Dir.cwd().deleteFile(io, oracle.tmp_path) catch {};

    const first = try oracle.outcomeOf(prog_in);
    oracle.signature = opts.signature orelse first.signature;
    if (std.mem.eql(u8, oracle.signature, "ok")) return error.NotFailing;

    var best = try deadCodeElim(allocator, prog_in);
    // Dead nodes are still lowered (a dead region node can break the compile): if the
    // failure needs them, start from the program as is.
    if (best.nodes.items.len != prog_in.nodes.items.len and !try oracle.interesting(&best)) best = try prog_in.clone(allocator);
    // 0. Observe only the failing node: everything after it goes away.
    if (first.node) |node| if (node < best.nodes.items.len and std.mem.eql(u8, first.signature, oracle.signature)) {
        var cand = try prog_in.clone(allocator);
        cand.outputs.clearRetainingCapacity();
        try cand.outputs.append(allocator, node);
        cand = try deadCodeElim(allocator, &cand);
        if (try oracle.interesting(&cand)) best = cand;
    };
    var changed = true;
    while (changed and oracle.candidates < opts.max_candidates) {
        changed = false;

        // 1. Drop outputs, one at a time.
        var o: usize = 0;
        while (o < best.outputs.items.len and best.outputs.items.len > 1) {
            var cand = try best.clone(allocator);
            _ = cand.outputs.orderedRemove(o);
            cand = try deadCodeElim(allocator, &cand);
            if (try oracle.interesting(&cand)) {
                best = cand;
                changed = true;
            } else o += 1;
        }

        // 2. Replace computed nodes by literals, from the outputs backwards.
        const values = try runner.reference(allocator, &best);
        var i = best.nodes.items.len;
        while (i > 0) {
            i -= 1;
            if (i >= best.nodes.items.len) continue;
            const node = best.nodes.items[i];
            if (ops.isArgument(node.op) or node.inputs.len == 0) continue;
            // The failing node itself has to stay computed, try its operands instead.
            if (node.scope != 0) continue;
            var cand = try best.clone(allocator);
            cand.nodes.items[i] = .{ .op = .{ .literal = .{ .data = values[i].bytes } }, .shape = node.shape };
            cand = try deadCodeElim(allocator, &cand);
            if (try oracle.interesting(&cand)) {
                best = cand;
                changed = true;
                break;
            }
        }
        if (changed) continue;

        // 3. Reroute operands to earlier nodes of the same shape, so more nodes die.
        reroute: for (best.nodes.items, 0..) |node, j| {
            for (node.inputs.slice(), 0..) |in, slot| {
                for (0..in) |cand_id| {
                    const other = best.nodes.items[cand_id];
                    if (other.scope != node.scope) continue;
                    if (!other.shape.eqlWithTags(best.nodes.items[in].shape)) continue;
                    var cand = try best.clone(allocator);
                    cand.nodes.items[j].inputs.mut()[slot] = @intCast(cand_id);
                    cand = try deadCodeElim(allocator, &cand);
                    if (cand.nodes.items.len >= best.nodes.items.len) continue;
                    if (try oracle.interesting(&cand)) {
                        best = cand;
                        changed = true;
                        break :reroute;
                    }
                }
            }
        }
        if (changed) continue;

        // 3b. Bypass a node: every use of it, region roots and outputs included, reads one of
        // its operands of the same shape instead. Unlike 2 this works inside loop bodies,
        // where a node's value changes every iteration and can't become a literal.
        bypass: for (best.nodes.items, 0..) |node, j| {
            // A region's first node and control flow ops are region bounds, which must not move.
            if (node.op == .param or hasRegions(node.op) or node.scope == j + 1) continue;
            for (node.inputs.slice()) |k| {
                const operand = best.nodes.items[k];
                if (operand.scope != node.scope or !operand.shape.eqlWithTags(node.shape)) continue;
                var cand = try best.clone(allocator);
                const map = try allocator.alloc(u32, cand.nodes.items.len);
                for (map, 0..) |*m, x| m.* = @intCast(x);
                map[j] = k;
                for (cand.nodes.items[j + 1 ..]) |*user| {
                    for (user.inputs.mut()) |*in| in.* = map[in.*];
                    switch (user.op) {
                        inline else => |*attrs, tag| {
                            const I = @field(ops.impls, @tagName(tag));
                            if (comptime @hasDecl(I, "remap")) I.remap(attrs, map);
                        },
                    }
                }
                for (cand.outputs.items) |*out| out.* = map[out.*];
                cand = try deadCodeElim(allocator, &cand);
                if (cand.nodes.items.len >= best.nodes.items.len) continue;
                if (try oracle.interesting(&cand)) {
                    best = cand;
                    changed = true;
                    break :bypass;
                }
            }
        }
        if (changed) continue;

        // 4. Simplify argument data.
        const vals = try runner.reference(allocator, &best);
        for (best.nodes.items, 0..) |node, j| {
            if (!ops.isArgument(node.op)) continue;
            for ([_]i64{ 0, 1, 2 }) |fill| {
                const v = try vals[j].dupe(allocator);
                var all_same = true;
                for (0..v.count()) |e| {
                    if (Value.isFloat(v.dtype())) {
                        if (v.getF(e) != @as(f64, @floatFromInt(fill))) all_same = false;
                        v.setF(e, @floatFromInt(fill));
                    } else {
                        if (v.getI(e) != fill) all_same = false;
                        v.setI(e, fill);
                    }
                }
                if (all_same) break;
                var cand = try best.clone(allocator);
                cand.nodes.items[j].op = .{ .literal = .{ .data = v.bytes } };
                if (try oracle.interesting(&cand)) {
                    best = cand;
                    changed = true;
                    break;
                }
            }
        }
    }
    // 5. Once nothing else applies, shrink dims: replace a dim on every axis with that tag
    // and dim, largest first. Ops whose attributes depend on the dim fail the shape oracle
    // and the candidate is dropped.
    shrink: while (oracle.candidates < opts.max_candidates) {
        var pairs: std.ArrayList(struct { tag: zml.Shape.Tag, dim: i64 }) = .empty;
        for (best.nodes.items) |node| for (0..node.shape.rank()) |ax| {
            const t, const d = .{ node.shape.tag(ax), node.shape.dim(ax) };
            if (d <= 1) continue;
            for (pairs.items) |p| {
                if (p.tag == t and p.dim == d) break;
            } else try pairs.append(allocator, .{ .tag = t, .dim = d });
        };
        std.mem.sort(@TypeOf(pairs.items[0]), pairs.items, {}, struct {
            fn gt(_: void, a: @TypeOf(pairs.items[0]), b: @TypeOf(pairs.items[0])) bool {
                return a.dim > b.dim;
            }
        }.gt);
        // Bisect the smallest dim that keeps the failure (assumed monotonic).
        for (pairs.items) |p| {
            var lo: i64, var hi: i64 = .{ 1, p.dim };
            var found: ?ir.Program = null;
            while (lo < hi and oracle.candidates < opts.max_candidates) {
                const mid = lo + @divTrunc(hi - lo, 2);
                const cand = try shrinkDim(allocator, &best, p.tag, p.dim, mid);
                if (try oracle.interesting(&cand)) {
                    found = cand;
                    hi = mid;
                } else lo = mid + 1;
            }
            if (found) |f| {
                best = f;
                continue :shrink;
            }
        }
        break;
    }
    return .{ .program = best, .signature = oracle.signature, .candidates = oracle.candidates };
}

fn hasRegions(op: ops.Op) bool {
    return switch (op) {
        inline else => |_, tag| @hasDecl(@field(ops.impls, @tagName(tag)), "regions"),
    };
}

/// Sets dim `to` on every axis tagged `tag` with dim `from`; literal data keeps the
/// elements whose index is still in bounds.
fn shrinkDim(allocator: std.mem.Allocator, prog: *const ir.Program, tag: zml.Shape.Tag, from: i64, to: i64) !ir.Program {
    const cand = try prog.clone(allocator);
    for (cand.nodes.items) |*node| {
        const old = node.shape;
        var sh = old;
        for (0..old.rank()) |ax| if (old.tag(ax) == tag and old.dim(ax) == from) {
            sh = sh.setDim(ax, to);
        };
        if (sh.count() == old.count()) continue;
        node.shape = sh;
        if (node.op == .literal) {
            const src = try Value.alloc(allocator, old);
            @memcpy(src.bytes, node.op.literal.data);
            const dst = try Value.alloc(allocator, sh);
            const st = Value.strides(old);
            var it: Value.Index = .init(sh);
            var i: usize = 0;
            while (!it.done) : (i += 1) {
                dst.copyElem(i, src, Value.offset(st, it.idx[0..sh.rank()]));
                _ = it.next();
            }
            node.op.literal.data = dst.bytes;
        }
    }
    return cand;
}

test deadCodeElim {
    const allocator = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prog: ir.Program = .{};
    const sh: zml.Shape = .init(.{2}, .f32);
    _ = try prog.add(a, .{ .op = .{ .input = .{ .seed = 1, .dist = .int_range } }, .shape = sh });
    _ = try prog.add(a, .{ .op = .{ .input = .{ .seed = 2, .dist = .int_range } }, .shape = sh });
    var n: ir.Node = .{ .op = .{ .unary = .{ .kind = .abs } }, .shape = sh };
    n.inputs.append(1);
    _ = try prog.add(a, n);
    try prog.outputs.append(a, 2);
    const res = try deadCodeElim(a, &prog);
    try std.testing.expectEqual(2, res.nodes.items.len);
    try std.testing.expectEqual(1, res.outputs.items[0]);
    try std.testing.expectEqual(0, res.nodes.items[1].inputs.slice()[0]);
}
