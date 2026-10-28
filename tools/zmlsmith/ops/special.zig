//! Ops without an exact reference: random numbers, checked through properties computed in
//! the graph; and numerical ops checked against f64 references: cholesky,
//! triangularSolve, complex numbers and FFTs (observed through real and imag).

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
const List = k.List;

fn rngState(g: *Gen) k.GenError!u32 {
    const v = try Value.alloc(g.allocator, .init(.{2}, .u64));
    v.setI(0, @as(i128, g.rng.int(u64)));
    v.setI(1, @as(i128, g.rng.int(u64)));
    return g.literalInput(v);
}

fn allTrue(e: *Eval, sh: Shape) Eval.Error!Value {
    const v = try e.alloc(sh);
    @memset(v.bytes, 1);
    return v;
}

// -----------------------------------------------------------------------------

pub const rng = struct {
    pub const Kind = enum {
        /// uniform(min, max) values are in [min, max).
        uniform_range,
        /// bitGenerator returns bits that aren't constant, and different bits from the new
        /// state. (Two calls on the same state would be merged by CSE.)
        bits_vary,
        /// bitGenerator returns a new state.
        state_advances,
        /// Statistics of 1024 samples, at about 6 sigma: uniform mean (floats) or the
        /// frequency of each integer in [min, max); normal mean and stddev (with the
        /// requested ones); gumbel mean and stddev (0.5772, pi / sqrt(6)). All finite.
        uniform_stats,
        normal_stats,
        gumbel_stats,
        /// nn.sampleTokens returns a token among the top k.
        sample_topk,
        /// nn.sampleTokens with topk = 1 is the first argmax.
        sample_greedy,
        /// nn.sampleTokensDynamic returns a token among the dynamic top k.
        sample_dynamic,
    };
    pub const Attrs = struct { kind: Kind, min: f64 = 0, max: f64 = 1, topk: u32 = 1, temperature: f64 = 1, dtype: DataType = .f32 };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        var a: Attrs = .{ .kind = kind };
        const state = try rngState(g);
        switch (kind) {
            .sample_topk, .sample_greedy, .sample_dynamic => {
                const dt = g.randDtype(.float) orelse return null;
                var sh: Shape = .init(.{}, dt);
                const voc = g.intIn(i64, 2, 8);
                if (g.chance(0.5)) sh = sh.appendDim(g.intIn(i64, 1, 3), Shape.toTag(.s));
                // sampleTokens gathers along .voc with the other axes as batching axes.
                const pos = if (g.cfg.avoid.contains(.gather_batching_after)) sh.rank() else g.rng.uintLessThan(usize, sh.rank() + 1);
                sh = sh.insertTag(pos, voc, Shape.toTag(.voc));
                const x = try g.operandLike(sh, true);
                if (g.valueOf(x).err > 0) return error.Reject;
                a.topk = if (kind == .sample_greedy) 1 else g.intIn(u32, 2, @intCast(voc));
                a.temperature = g.choose(f64, &.{ 1, 0.5, 2 });
                const out = if (kind == .sample_greedy) sh.remove(sh.hasTag(Shape.toTag(.voc)).?).withDtype(.i32) else sh.remove(sh.hasTag(Shape.toTag(.voc)).?).withDtype(.bool);
                if (kind == .sample_dynamic) {
                    // Runtime knobs: top_k (i32), temperature, top_p, min_p (logits dtype).
                    // max_top_k is a compile time bound, also stored in `max`.
                    a.max = @floatFromInt(g.intIn(i64, 0, voc));
                    const kv = try Value.alloc(g.allocator, .init(.{}, .i32));
                    kv.setI(0, g.intIn(i64, 0, voc));
                    var knobs: [4]u32 = undefined;
                    // Embedded constants: as arguments they would exceed the input budget.
                    knobs[0] = try g.commit(.{ .op = .{ .constant = .{ .data = kv.bytes } }, .shape = kv.shape });
                    for ([_]f64{ g.choose(f64, &.{ 1, 0.5, 2 }), g.choose(f64, &.{ 1, 0.9, 0.5 }), g.choose(f64, &.{ 0, 0.1 }) }, 1..) |val, j| {
                        const fv = try Value.alloc(g.allocator, .init(.{}, dt));
                        fv.setF(0, val);
                        knobs[j] = try g.commit(.{ .op = .{ .constant = .{ .data = fv.bytes } }, .shape = fv.shape });
                    }
                    return .{ .op = .{ .rng = a }, .inputs = try k.dupeIds(g, &.{ state, x, knobs[0], knobs[1], knobs[2], knobs[3] }), .shape = out };
                }
                return .{ .op = .{ .rng = a }, .inputs = try k.dupeIds(g, &.{ state, x }), .shape = out };
            },
            else => {},
        }
        const dt = switch (kind) {
            .uniform_range => g.randDtype(.numeric) orelse return null,
            .normal_stats, .gumbel_stats => g.randDtype(.float) orelse return null,
            .uniform_stats => g.randDtype(.numeric) orelse return null,
            else => g.randDtype(.int) orelse return null,
        };
        var sh = g.randShape(dt, .{ .max_rank = 3 });
        if (sh.count() == 0) return error.Reject;
        a.dtype = dt;
        if (kind == .uniform_range or kind == .uniform_stats) {
            try g.avoid(.uniform_bf16, dt == .bf16);
            a.min = @floatFromInt(g.intIn(i64, -4, 4));
            a.max = a.min + @as(f64, @floatFromInt(g.intIn(i64, 1, 8)));
            if (Value.isInt(dt) and !dt.isSignedInt() and a.min < 0) {
                a.max -= a.min;
                a.min = 0;
            }
        }
        if (kind == .uniform_stats) try g.avoid(.uniform_int_bias, Value.isInt(dt) and a.min < 0);
        if (kind == .normal_stats) {
            a.min = @floatFromInt(g.intIn(i64, -2, 2));
            a.max = g.choose(f64, &.{ 0.5, 1, 2 });
        }
        switch (kind) {
            .state_advances, .bits_vary, .uniform_stats, .normal_stats, .gumbel_stats => sh = Shape.init(.{}, .bool),
            else => {},
        }
        return .{ .op = .{ .rng = a }, .inputs = try k.dupeIds(g, &.{state}), .shape = sh.withDtype(.bool) };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        if (a.kind == .sample_dynamic and ins[1].err > 0) try e.reject();
        if (a.kind != .sample_greedy) return allTrue(e, node.shape);
        // Greedy sampling is the first argmax along .voc.
        const x = ins[1];
        var v = try e.alloc(node.shape);
        const vax = x.shape.hasTag(Shape.toTag(.voc)).?;
        const st = Value.strides(x.shape);
        var it: Value.Index = .init(x.shape.setDim(vax, 1));
        var o: usize = 0;
        while (!it.done) : (o += 1) {
            const base = Value.offset(st, it.idx[0..x.shape.rank()]);
            var best: usize = 0;
            for (1..@intCast(x.shape.dim(vax))) |j| if (x.getF(base + j * st[vax]) > x.getF(base + best * st[vax])) {
                best = j;
            };
            v.setI(o, @intCast(best));
            _ = it.next();
        }
        return v;
    }

    pub fn lower(a: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const r: Tensor.Rng = .{ ._state = ins[0] };
        // The node holds the predicate; the samples have the data dtype.
        const sh = node.shape.withDtype(a.dtype);
        switch (a.kind) {
            .uniform_range => {
                _, const x = r.uniform(sh.withDtype(sh.dtype()), .{ .min = a.min, .max = a.max });
                return inRange(x, a);
            },
            .bits_vary => {
                const s: Shape = .init(.{256}, a.dtype);
                const r2, const x = r.bitGenerator(s);
                _, const y = r2.bitGenerator(s);
                return count(x.cmp(.NE, y)).cmp(.GE, Tensor.scalar(200, .i32)).logical(.AND, count(x.cmp(.NE, x.reverse(.{0}))).cmp(.GE, Tensor.scalar(200, .i32)));
            },
            .uniform_stats => {
                _, const x = r.uniform(Shape.init(.{n_stats}, a.dtype), .{ .min = a.min, .max = a.max });
                if (Value.isFloat(a.dtype)) {
                    const tol = 6 * (a.max - a.min) / @sqrt(12.0 * n_stats);
                    return x.convert(.f32).mean(0).squeeze(0).addConstant(-(a.min + a.max) / 2).abs().cmp(.LE, Tensor.scalar(tol, .f32));
                }
                const lo: i64 = @intFromFloat(a.min);
                const kk: i64 = @intFromFloat(a.max - a.min);
                var ok = x.cmp(.GE, Tensor.scalar(lo, a.dtype)).convert(.i32).sum(0).squeeze(0).cmp(.EQ, Tensor.scalar(n_stats, .i32));
                for (0..@intCast(kk)) |j| {
                    const c = count(x.cmp(.EQ, Tensor.scalar(lo + @as(i64, @intCast(j)), a.dtype)));
                    ok = ok.logical(.AND, c.cmp(.GE, Tensor.scalar(@divTrunc(n_stats, 2 * kk), .i32))).logical(.AND, c.cmp(.LE, Tensor.scalar(@divTrunc(2 * n_stats, kk), .i32)));
                }
                return ok;
            },
            .normal_stats, .gumbel_stats => {
                const s: Shape = .init(.{n_stats}, a.dtype);
                const x = (if (a.kind == .normal_stats) Tensor.Rng.normal(s, .{ .mean = a.min, .stddev = a.max }) else r.gumbel(s)[1]).convert(.f32);
                const mean_want: f64, const sd_want: f64, const sd_tol: f64 = if (a.kind == .normal_stats) .{ a.min, a.max, 0.2 * a.max } else .{ 0.5772, 1.2825, 0.25 };
                const m = x.mean(0);
                const sd = x.sub(m.broad(x.shape())).powByConst(2).mean(0).sqrt().squeeze(0);
                const finite = count(x.isFinite()).cmp(.EQ, Tensor.scalar(n_stats, .i32));
                return finite.logical(.AND, m.squeeze(0).addConstant(-mean_want).abs().cmp(.LE, Tensor.scalar(6 * sd_want / @sqrt(@as(f64, n_stats)), .f32)))
                    .logical(.AND, sd.addConstant(-sd_want).abs().cmp(.LE, Tensor.scalar(sd_tol, .f32)));
            },
            .state_advances => {
                const next, _ = r.bitGenerator(Shape.init(.{4}, .u32));
                // Tensor.any doesn't compile (KnownBug.any_compile): count the differences.
                return next._state.cmp(.NE, ins[0]).convert(.i32).sum(0).squeeze(0).cmp(.GT, Tensor.scalar(0, .i32));
            },
            .sample_dynamic => {
                const x = ins[1];
                const opts: zml.nn.DynamicSamplingStrategy = .{ .max_top_k = @intFromFloat(a.max), .top_k = ins[2], .temperature = ins[3], .top_p = ins[4], .min_p = ins[5] };
                const tokens, _ = zml.nn.sampleTokensDynamic(x, opts, r);
                // The token is among the top max(1, min(top_k, max_top_k or voc)) logits.
                const voc: i64 = x.dim(.voc);
                const bound: i64 = if (a.max > 0) @intFromFloat(a.max) else voc;
                const kth_k = ins[2].convert(.i32).minimum(Tensor.scalar(bound, .i32)).maximum(Tensor.scalar(1, .i32));
                const sorted = x.sort(.voc, .{ .descending = true }).values;
                const kth = sorted.gather(.{ .voc = kth_k.addConstant(-1) }, .{});
                const picked = x.gather(.{ .voc = tokens }, .{});
                return picked.cmp(.GE, kth);
            },
            .sample_greedy, .sample_topk => {
                const x = ins[1];
                const tokens, _ = zml.nn.sampleTokens(x, .{ .topk = a.topk, .temperature = @floatCast(a.temperature) }, r);
                if (a.kind == .sample_greedy) return tokens;
                // The sampled logit is at least the k-th largest.
                const picked = x.gather(.{ .voc = tokens }, .{});
                const kth = x.topK(.{ .voc = .voc }, a.topk, .{}).values.min(.voc).squeeze(.voc);
                return picked.cmp(.GE, kth);
            },
        }
    }

    const n_stats = 1024;

    /// Number of true elements of a rank 1 bool tensor, as an i32 scalar.
    fn count(b: Tensor) Tensor {
        return b.convert(.i32).sum(0).squeeze(0);
    }

    fn inRange(x: Tensor, a: Attrs) Tensor {
        const lo = if (x.dtype().isFloat()) Tensor.scalar(a.min, x.dtype()) else Tensor.scalar(@as(i64, @intFromFloat(a.min)), x.dtype());
        const hi = if (x.dtype().isFloat()) Tensor.scalar(a.max, x.dtype()) else Tensor.scalar(@as(i64, @intFromFloat(a.max)), x.dtype());
        // Documented interval [min, max); integers are converted from floats, so max can
        // only be reached by rounding toward it.
        const below = if (x.dtype().isFloat()) x.cmp(.LT, hi) else x.cmp(.LE, hi);
        return x.cmp(.GE, lo).logical(.AND, below);
    }

    pub fn emit(a: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        // Property ops are emitted as blocks computing the same predicate as `lower`.
        try w.print("blk: {{\n                const r: zml.Tensor.Rng = .{{ ._state = {s} }};\n                ", .{ins[0]});
        const sh = node.shape.withDtype(a.dtype);
        switch (a.kind) {
            .uniform_range => {
                try w.writeAll("_, const x = r.uniform(");
                try k.printShapeLiteral(w, sh);
                try w.print(", .{{ .min = {d}, .max = {d} }});\n                ", .{ a.min, a.max });
                if (k.Value.isFloat(sh.dtype())) {
                    try w.print("break :blk x.cmp(.GE, zml.Tensor.scalar(@as(f64, {d}), x.dtype())).logical(.AND, x.cmp(.LT, zml.Tensor.scalar(@as(f64, {d}), x.dtype())));\n", .{ a.min, a.max });
                } else {
                    try w.print("break :blk x.cmp(.GE, zml.Tensor.scalar({d}, x.dtype())).logical(.AND, x.cmp(.LE, zml.Tensor.scalar({d}, x.dtype())));\n", .{ @as(i64, @intFromFloat(a.min)), @as(i64, @intFromFloat(a.max)) });
                }
            },
            .bits_vary => {
                try w.print("const s = zml.Shape.init(.{{256}}, .{s});\n                const r2, const x = r.bitGenerator(s);\n                _, const y = r2.bitGenerator(s);\n                ", .{@tagName(a.dtype)});
                try w.writeAll("break :blk x.cmp(.NE, y).convert(.i32).sum(0).squeeze(0).cmp(.GE, zml.Tensor.scalar(200, .i32)).logical(.AND, x.cmp(.NE, x.reverse(.{0})).convert(.i32).sum(0).squeeze(0).cmp(.GE, zml.Tensor.scalar(200, .i32)));\n");
            },
            .uniform_stats => {
                try w.print("_, const x = r.uniform(zml.Shape.init(.{{{d}}}, .{s}), .{{ .min = {d}, .max = {d} }});\n                ", .{ n_stats, @tagName(a.dtype), a.min, a.max });
                if (Value.isFloat(a.dtype)) {
                    try w.print("break :blk x.convert(.f32).mean(0).squeeze(0).addConstant({d}).abs().cmp(.LE, zml.Tensor.scalar({d}, .f32));\n", .{ -(a.min + a.max) / 2, 6 * (a.max - a.min) / @sqrt(12.0 * n_stats) });
                } else {
                    const lo: i64 = @intFromFloat(a.min);
                    const kk: i64 = @intFromFloat(a.max - a.min);
                    try w.print("var ok = x.cmp(.GE, zml.Tensor.scalar({d}, x.dtype())).convert(.i32).sum(0).squeeze(0).cmp(.EQ, zml.Tensor.scalar({d}, .i32));\n                ", .{ lo, n_stats });
                    try w.print("for (0..{d}) |j| {{\n                    const c = x.cmp(.EQ, zml.Tensor.scalar({d} + @as(i64, @intCast(j)), x.dtype())).convert(.i32).sum(0).squeeze(0);\n                    ok = ok.logical(.AND, c.cmp(.GE, zml.Tensor.scalar({d}, .i32))).logical(.AND, c.cmp(.LE, zml.Tensor.scalar({d}, .i32)));\n                }}\n                break :blk ok;\n", .{ kk, lo, @divTrunc(n_stats, 2 * kk), @divTrunc(2 * n_stats, kk) });
                }
            },
            .normal_stats, .gumbel_stats => {
                const mean_want: f64, const sd_want: f64, const sd_tol: f64 = if (a.kind == .normal_stats) .{ a.min, a.max, 0.2 * a.max } else .{ 0.5772, 1.2825, 0.25 };
                if (a.kind == .normal_stats) {
                    try w.print("_ = &r;\n                const x = zml.Tensor.Rng.normal(zml.Shape.init(.{{{d}}}, .{s}), .{{ .mean = {d}, .stddev = {d} }}).convert(.f32);\n                ", .{ n_stats, @tagName(a.dtype), a.min, a.max });
                } else {
                    try w.print("const x = r.gumbel(zml.Shape.init(.{{{d}}}, .{s}))[1].convert(.f32);\n                ", .{ n_stats, @tagName(a.dtype) });
                }
                try w.writeAll("const m = x.mean(0);\n                const sd = x.sub(m.broad(x.shape())).powByConst(2).mean(0).sqrt().squeeze(0);\n                ");
                try w.print("break :blk x.isFinite().convert(.i32).sum(0).squeeze(0).cmp(.EQ, zml.Tensor.scalar({d}, .i32)).logical(.AND, m.squeeze(0).addConstant({d}).abs().cmp(.LE, zml.Tensor.scalar({d}, .f32))).logical(.AND, sd.addConstant({d}).abs().cmp(.LE, zml.Tensor.scalar({d}, .f32)));\n", .{ n_stats, -mean_want, 6 * sd_want / @sqrt(@as(f64, n_stats)), -sd_want, sd_tol });
            },
            .state_advances => try w.print("const next, _ = r.bitGenerator(zml.Shape.init(.{{4}}, .u32));\n                break :blk next._state.cmp(.NE, {s}).convert(.i32).sum(0).squeeze(0).cmp(.GT, zml.Tensor.scalar(0, .i32));\n", .{ins[0]}),
            .sample_dynamic => {
                try w.print("const t, _ = zml.nn.sampleTokensDynamic({s}, .{{ .max_top_k = {d}, .top_k = {s}, .temperature = {s}, .top_p = {s}, .min_p = {s} }}, r);\n                ", .{ ins[1], @as(u32, @intFromFloat(a.max)), ins[2], ins[3], ins[4], ins[5] });
                try w.print("const kk = {s}.convert(.i32).minimum(zml.Tensor.scalar({d}, .i32)).maximum(zml.Tensor.scalar(1, .i32));\n                ", .{ ins[2], if (a.max > 0) @as(i64, @intFromFloat(a.max)) else k.emit_shapes[1].dim(Shape.toTag(.voc)) });
                try w.print("break :blk {s}.gather(.{{ .voc = t }}, .{{}}).cmp(.GE, {s}.sort(.voc, .{{ .descending = true }}).values.gather(.{{ .voc = kk.addConstant(-1) }}, .{{}}));\n", .{ ins[1], ins[1] });
            },
            .sample_greedy, .sample_topk => {
                try w.print("const t, _ = zml.nn.sampleTokens({s}, .{{ .topk = {d}, .temperature = {d} }}, r);\n                ", .{ ins[1], a.topk, a.temperature });
                if (a.kind == .sample_greedy) {
                    try w.writeAll("break :blk t;\n");
                } else {
                    try w.print("break :blk {s}.gather(.{{ .voc = t }}, .{{}}).cmp(.GE, {s}.topK(.{{ .voc = .voc }}, {d}, .{{}}).values.min(.voc).squeeze(.voc));\n", .{ ins[1], ins[1], a.topk });
                }
            },
        }
        try w.writeAll("            }");
    }
};

// -----------------------------------------------------------------------------

pub const decomp = struct {
    pub const Kind = enum { cholesky, triangular_solve };
    pub const Transpose = enum { NO_TRANSPOSE, TRANSPOSE, ADJOINT };
    pub const Attrs = struct {
        kind: Kind,
        lower: bool,
        left_side: bool = true,
        unit_diagonal: bool = false,
        transpose: Transpose = .NO_TRANSPOSE,
    };
    pub const weight = 2;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const dt = g.choose(DataType, &.{ .f32, .f64 });
        const n = g.intIn(i64, 1, 4);
        var a: Attrs = .{ .kind = kind, .lower = g.chance(0.5) };
        switch (kind) {
            .cholesky => {
                // A = B B^T + n I, symmetric positive definite with integer entries.
                var b: [16]i64 = undefined;
                const nn: usize = @intCast(n);
                for (b[0 .. nn * nn]) |*x| x.* = g.intIn(i64, -2, 2);
                const av = try Value.alloc(g.allocator, Shape.init(.{ n, n }, dt));
                for (0..nn) |i| for (0..nn) |j| {
                    var s: i64 = if (i == j) n else 0;
                    for (0..nn) |t| s += b[i * nn + t] * b[j * nn + t];
                    av.setF(i * nn + j, @floatFromInt(s));
                };
                const x = try g.literalInput(av);
                return .{ .op = .{ .decomp = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = av.shape };
            },
            .triangular_solve => {
                a.left_side = g.chance(0.5);
                a.unit_diagonal = g.chance(0.3);
                a.transpose = g.rng.enumValue(Transpose);
                const batch = g.chance(0.3);
                const m = g.intIn(i64, 1, 3);
                const bs: i64 = if (batch) g.intIn(i64, 1, 2) else 1;
                var ash: Shape = .init(.{}, dt);
                var bsh: Shape = .init(.{}, dt);
                if (batch) {
                    ash = ash.appendDim(bs, null);
                    bsh = bsh.appendDim(bs, null);
                }
                ash = ash.appendDim(n, null).appendDim(n, null);
                bsh = if (a.left_side) bsh.appendDim(n, null).appendDim(m, null) else bsh.appendDim(m, null).appendDim(n, null);
                const av = try Value.alloc(g.allocator, ash);
                const nn: usize = @intCast(n);
                for (0..@intCast(bs)) |bi| for (0..nn) |i| for (0..nn) |j| {
                    const off = bi * nn * nn + i * nn + j;
                    // The other triangle holds garbage: it must be ignored.
                    if (i == j) av.setF(off, g.choose(f64, &.{ 1, 2, -2, 4 })) else av.setF(off, @floatFromInt(g.intIn(i64, -2, 2)));
                };
                const am = try g.literalInput(av);
                const bm = try g.freshInput(bsh);
                return .{ .op = .{ .decomp = a }, .inputs = try k.dupeIds(g, &.{ am, bm }), .shape = bsh };
            },
        }
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        const dt = node.shape.dtype();
        switch (a.kind) {
            .cholesky => {
                const x = ins[0];
                const n: usize = @intCast(x.shape.dim(0));
                var l: [16]f64 = @splat(0);
                for (0..n) |j| {
                    var d = x.getF(j * n + j);
                    for (0..j) |t| d -= l[j * n + t] * l[j * n + t];
                    l[j * n + j] = @sqrt(d);
                    for (j + 1..n) |i| {
                        var s = x.getF(i * n + j);
                        for (0..j) |t| s -= l[i * n + t] * l[j * n + t];
                        l[i * n + j] = s / l[j * n + j];
                    }
                }
                // Only the requested triangle is defined; the lowering zeroes the other one.
                for (0..n) |i| for (0..n) |j| {
                    const val = if (a.lower) l[i * n + j] else l[j * n + i];
                    _ = k.storeF(v, i * n + j, val);
                };
                v.err = 64 * @as(f64, @floatFromInt(n * n)) * Value.eps(dt) * @max(1, x.maxAbs());
            },
            .triangular_solve => {
                const am = ins[0];
                const bm = ins[1];
                const n: usize = @intCast(am.shape.dim(-1));
                const batch: usize = if (am.shape.rank() == 3) @intCast(am.shape.dim(0)) else 1;
                const m: usize = @intCast(if (a.left_side) bm.shape.dim(-1) else bm.shape.dim(-2));
                var mag: f64 = 1;
                for (0..batch) |bi| {
                    // op(A): the stored triangle, unit diagonal optionally, transposed optionally.
                    var t: [16]f64 = @splat(0);
                    for (0..n) |i| for (0..n) |j| {
                        const in_tri = if (a.lower) j <= i else j >= i;
                        var val: f64 = if (in_tri) am.getF(bi * n * n + i * n + j) else 0;
                        if (i == j and a.unit_diagonal) val = 1;
                        t[i * n + j] = val;
                    };
                    if (a.transpose != .NO_TRANSPOSE) {
                        var tt: [16]f64 = undefined;
                        for (0..n) |i| for (0..n) |j| {
                            tt[i * n + j] = t[j * n + i];
                        };
                        t = tt;
                    }
                    // Solve op(A) X = B (left) or X op(A) = B (right) by Gaussian elimination on
                    // the triangular system.
                    for (0..m) |c| {
                        var rhs: [4]f64 = undefined;
                        for (0..n) |i| rhs[i] = if (a.left_side) bm.getF(bi * n * m + i * m + c) else bm.getF(bi * m * n + c * n + i);
                        var x: [4]f64 = undefined;
                        // Right side: x A = b  <=>  A^T x^T = b^T.
                        var mat: [16]f64 = t;
                        if (!a.left_side) for (0..n) |i| for (0..n) |j| {
                            mat[i * n + j] = t[j * n + i];
                        };
                        const lower_now = blk: {
                            var lo = true;
                            for (0..n) |i| for (i + 1..n) |j| if (mat[i * n + j] != 0) {
                                lo = false;
                            };
                            break :blk lo;
                        };
                        if (lower_now) {
                            for (0..n) |i| {
                                var s = rhs[i];
                                for (0..i) |j| s -= mat[i * n + j] * x[j];
                                x[i] = s / mat[i * n + i];
                            }
                        } else {
                            var i = n;
                            while (i > 0) {
                                i -= 1;
                                var s = rhs[i];
                                for (i + 1..n) |j| s -= mat[i * n + j] * x[j];
                                x[i] = s / mat[i * n + i];
                            }
                        }
                        for (0..n) |i| {
                            mag = @max(mag, @abs(x[i]));
                            const off = if (a.left_side) bi * n * m + i * m + c else bi * m * n + c * n + i;
                            _ = k.storeF(v, off, x[i]);
                        }
                    }
                }
                v.err = 64 * @as(f64, @floatFromInt(n * n)) * Value.eps(dt) * mag * @max(1, bm.maxAbs()) + 8 * bm.err * mag;
            },
        }
        return v;
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        switch (a.kind) {
            .cholesky => {
                const l = ins[0].cholesky(a.lower);
                // Zero the implementation defined triangle.
                return if (a.lower) l.triangular(.{ 0, 1 }, 0) else l.triangular(.{ 1, 0 }, 0);
            },
            .triangular_solve => return ins[0].triangularSolve(ins[1], .{
                .left_side = a.left_side,
                .lower = a.lower,
                .unit_diagonal = a.unit_diagonal,
                .transpose_a = switch (a.transpose) {
                    .NO_TRANSPOSE => .NO_TRANSPOSE,
                    .TRANSPOSE => .TRANSPOSE,
                    .ADJOINT => .ADJOINT,
                },
            }),
        }
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (a.kind) {
            .cholesky => try w.print("{s}.cholesky({}).triangular(.{{ {s} }}, 0)", .{ ins[0], a.lower, if (a.lower) "0, 1" else "1, 0" }),
            .triangular_solve => try w.print("{s}.triangularSolve({s}, .{{ .left_side = {}, .lower = {}, .unit_diagonal = {}, .transpose_a = .{s} }})", .{ ins[0], ins[1], a.left_side, a.lower, a.unit_diagonal, @tagName(a.transpose) }),
        }
    }
};

// -----------------------------------------------------------------------------

pub const complex = struct {
    pub const Kind = enum { real, imag, abs, fft, ifft, rfft, rfft_roundtrip };
    pub const Attrs = struct { kind: Kind, dims: u8 = 1, take_imag: bool = false };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const dt = g.choose(DataType, &.{ .f32, .f64 });
        const x = try g.operand(.{ .dtype = dt, .min_rank = 1, .max_rank = 3 });
        const sh = g.shapeOf(x);
        for (0..sh.rank()) |ax| if (sh.dim(ax) == 0) return error.Reject;
        try g.avoid(.xla_fft_layout, g.prog.sharded and kind != .real and kind != .imag and kind != .abs);
        var a: Attrs = .{ .kind = kind, .take_imag = g.chance(0.5) };
        const y = try g.operandLike(sh, true);
        a.dims = g.intIn(u8, 1, @intCast(@min(2, sh.rank())));
        var out = sh;
        if (kind == .rfft and !a.take_imag) out = sh;
        if (kind == .rfft) out = sh.setDim(-1, @divTrunc(sh.dim(-1), 2) + 1);
        const needs_im = switch (kind) {
            .real, .imag, .abs, .fft, .ifft => true,
            else => false,
        };
        return .{ .op = .{ .complex = a }, .inputs = try k.dupeIds(g, if (needs_im) &.{ x, y } else &.{x}), .shape = out };
    }

    /// DFT over the last `dims` axes of (re, im); `sign` -1 forward, +1 inverse (with 1/N).
    fn dft(e: *Eval, re: Value, im: ?Value, dims: u8, inverse: bool, out_sh: Shape, take_imag: bool, half: bool) Eval.Error!Value {
        var v = try e.alloc(out_sh);
        const sh = re.shape;
        const st = Value.strides(sh);
        const r = sh.rank();
        var it: Value.Index = .init(out_sh);
        var o: usize = 0;
        var total: f64 = 0;
        for (0..re.count()) |i| total += @abs(re.getF(i)) + if (im) |m| @abs(m.getF(i)) else 0;
        var n_total: f64 = 1;
        for (r - dims..r) |ax| n_total *= @floatFromInt(sh.dim(ax));
        while (!it.done) : (o += 1) {
            var acc_re: f64 = 0;
            var acc_im: f64 = 0;
            var fsh: Shape = .init(.{}, .i32);
            for (r - dims..r) |ax| fsh = fsh.appendDim(sh.dim(ax), null);
            var fit: Value.Index = .init(fsh);
            while (!fit.done) {
                var idx: [8]i64 = undefined;
                @memcpy(idx[0..r], it.idx[0..r]);
                var phase: f64 = 0;
                for (r - dims..r, 0..) |ax, j| {
                    idx[ax] = fit.idx[j];
                    const nd: f64 = @floatFromInt(sh.dim(ax));
                    phase += @as(f64, @floatFromInt(it.idx[ax] * fit.idx[j])) / nd;
                }
                const ang = (if (inverse) @as(f64, 2) else -2) * std.math.pi * phase;
                const off = Value.offset(st, idx[0..r]);
                const xr = re.getF(off);
                const xi = if (im) |m| m.getF(off) else 0;
                acc_re += xr * @cos(ang) - xi * @sin(ang);
                acc_im += xr * @sin(ang) + xi * @cos(ang);
                _ = fit.next();
            }
            if (inverse) {
                acc_re /= n_total;
                acc_im /= n_total;
            }
            _ = k.storeF(v, o, if (take_imag) acc_im else acc_re);
            _ = it.next();
        }
        _ = half;
        // Both inputs contribute: each output sums n_total terms of each.
        v.err = 32 * n_total * Value.eps(re.dtype()) * @max(1, total) + n_total * (re.err + if (im) |m| m.err else 0);
        return v;
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        switch (a.kind) {
            .real => return x.dupe(e.allocator),
            .imag => return ins[1].dupe(e.allocator),
            .abs => {
                var v = try e.alloc(node.shape);
                for (0..x.count()) |i| _ = k.storeF(v, i, std.math.hypot(x.getF(i), ins[1].getF(i)));
                v.err = 2 * (x.err + ins[1].err) + k.approxErr(x.dtype(), v.maxAbs(), 0);
                return v;
            },
            .fft, .ifft => return dft(e, x, ins[1], a.dims, a.kind == .ifft, node.shape, a.take_imag, false),
            .rfft => {
                // The full DFT restricted to the first n/2 + 1 bins of the last axis.
                const full = try dft(e, x, null, a.dims, false, x.shape, a.take_imag, false);
                var v = try e.alloc(node.shape);
                v.err = full.err;
                const fst = Value.strides(x.shape);
                var it: Value.Index = .init(node.shape);
                var o: usize = 0;
                while (!it.done) : (o += 1) {
                    v.copyElem(o, full, Value.offset(fst, it.idx[0..x.shape.rank()]));
                    _ = it.next();
                }
                return v;
            },
            .rfft_roundtrip => {
                var v = try x.dupe(e.allocator);
                var n: f64 = 1;
                for (x.shape.rank() - a.dims..x.shape.rank()) |ax| n *= @floatFromInt(x.shape.dim(ax));
                v.err = 64 * n * Value.eps(x.dtype()) * @max(1, x.maxAbs()) * n + x.err * 2;
                return v;
            },
        }
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const sh = x.shape();
        var lens: [2]i64 = undefined;
        for (0..a.dims) |j| lens[j] = sh.dim(sh.rank() - a.dims + j);
        const len = lens[0..a.dims];
        switch (a.kind) {
            .real => return Tensor.complex(x, ins[1]).real(),
            .imag => return Tensor.complex(x, ins[1]).imag(),
            .abs => return Tensor.complex(x, ins[1]).abs(),
            .fft, .ifft => {
                const c = Tensor.complex(x, ins[1]).fft(.{ .kind = if (a.kind == .fft) .FFT else .IFFT, .length = len });
                return if (a.take_imag) c.imag() else c.real();
            },
            .rfft => {
                const c = x.fft(.{ .kind = .RFFT, .length = len });
                return if (a.take_imag) c.imag() else c.real();
            },
            .rfft_roundtrip => return x.fft(.{ .kind = .RFFT, .length = len }).fft(.{ .kind = .IRFFT, .length = len }),
        }
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const sh = k.emit_shapes[0];
        var lens: [64]u8 = undefined;
        var fbs: Writer = .fixed(&lens);
        fbs.writeAll("&[_]i64{ ") catch {};
        for (0..a.dims) |j| fbs.print("{s}{d}", .{ if (j > 0) ", " else "", sh.dim(sh.rank() - a.dims + j) }) catch {};
        fbs.writeAll(" }") catch {};
        const l = fbs.buffered();
        const part = if (a.take_imag) "imag" else "real";
        switch (a.kind) {
            .real, .imag, .abs => try w.print("zml.Tensor.complex({s}, {s}).{s}()", .{ ins[0], ins[1], @tagName(a.kind) }),
            .fft, .ifft => try w.print("zml.Tensor.complex({s}, {s}).fft(.{{ .kind = .{s}, .length = {s} }}).{s}()", .{ ins[0], ins[1], if (a.kind == .fft) "FFT" else "IFFT", l, part }),
            .rfft => try w.print("{s}.fft(.{{ .kind = .RFFT, .length = {s} }}).{s}()", .{ ins[0], l, part }),
            .rfft_roundtrip => try w.print("{s}.fft(.{{ .kind = .RFFT, .length = {s} }}).fft(.{{ .kind = .IRFFT, .length = {s} }})", .{ ins[0], l, l }),
        }
    }
};
