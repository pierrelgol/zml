//! zml.quantization: block FP8 and NVFP4 activation quantizers, global scales,
//! nn.unpackFp4, the bf16 ops.scaled_dot, and ops.composite.
//!
//! The quantizers are emulated step by step in the precision ZML computes them in, so the
//! values and scales they return are compared bit for bit. nn.scaledDot has no CPU
//! implementation (its decomposition is an unregistered custom call) and isn't generated.

const std = @import("std");

const zml = @import("zml");

const k = @import("../kit.zig");
const linalg = @import("linalg.zig");
const Shape = k.Shape;
const DataType = k.DataType;
const Tensor = k.Tensor;
const Value = k.Value;
const Gen = k.Gen;
const Eval = k.Eval;
const Node = k.Node;
const Writer = k.Writer;
const TagId = k.TagId;

const BFloat16 = zml.floats.BFloat16;
const GlobalScale = zml.Quantization.GlobalScale;

pub const quant = struct {
    pub const Kind = enum { block_fp8, nvfp4, unpack_fp4, scaled_dot, global_scale };
    /// Which field of the returned QuantizedInput the node is.
    pub const Part = enum { values, scales };
    pub const Attrs = struct {
        kind: Kind,
        part: Part = .values,
        axis: k.AxisRef = .{ .index = 0 },
        /// quantizeBlockFp8 block size.
        block: i64 = 1,
        dtype: DataType = .f8e4m3fn,
        scale_dtype: DataType = .f32,
        /// unpackFp4 result tag, scaled_dot contracting tag.
        tag: TagId = .none,
        /// quantizeNvfp4 takes the second input as input_global_scale.
        global_scale: bool = false,
        operation: GlobalScale.Operation = .multiply,
    };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        return switch (g.rng.enumValue(Kind)) {
            .block_fp8 => proposeBlockFp8(g),
            .nvfp4 => proposeNvfp4(g),
            .unpack_fp4 => proposeUnpack(g),
            .scaled_dot => proposeScaledDot(g),
            .global_scale => proposeGlobalScale(g),
        };
    }

    fn proposeBlockFp8(g: *Gen) k.GenError!?k.Proposal {
        const x = try g.operand(.{ .class = .float, .exact = true, .min_rank = 1, .max_rank = 4 });
        const sh = g.shapeOf(x);
        if (hasScratchTags(sh, &.{ .fp8_ks, .fp8_block })) return error.Reject;
        const ax = g.rng.uintLessThan(usize, sh.rank());
        const d = sh.dim(ax);
        if (d == 0) return error.Reject;
        // A random divisor of the axis.
        var divs: [64]i64 = undefined;
        var n: usize = 0;
        var b: i64 = 1;
        while (b <= d and n < divs.len) : (b += 1) if (@mod(d, b) == 0) {
            divs[n] = b;
            n += 1;
        };
        const block = divs[g.rng.uintLessThan(usize, n)];
        const a: Attrs = .{
            .kind = .block_fp8,
            .part = g.rng.enumValue(Part),
            .axis = .random(g, sh, ax),
            .block = block,
            .dtype = g.choose(DataType, &.{ .f8e4m3fn, .f8e4m3fnuz }),
            .scale_dtype = g.choose(DataType, &.{ .f32, .f8e8m0 }),
        };
        const out = switch (a.part) {
            .values => sh.withDtype(a.dtype),
            .scales => sh.setDim(ax, @divExact(d, block)).withDtype(a.scale_dtype),
        };
        return .{ .op = .{ .quant = a }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    fn proposeNvfp4(g: *Gen) k.GenError!?k.Proposal {
        // The block is 16 values: build an input whose quantized axis holds 1 or 2 blocks.
        const dt = g.choose(DataType, &.{ .f32, .bf16 });
        const rank = g.intIn(usize, 1, 3);
        var dims: [3]i64 = undefined;
        var tags: [3]Shape.Tag = undefined;
        g.randTags(tags[0..rank]);
        for (0..rank) |i| dims[i] = g.intIn(i64, 1, 3);
        const ax = g.rng.uintLessThan(usize, rank);
        dims[ax] = 16 * g.intIn(i64, 1, 2);
        const sh = k.ir.makeShape(dt, dims[0..rank], tags[0..rank]);
        const x = try g.operandLike(sh, true);
        // The global scale divides x in x's dtype, a second rounding: f32 only.
        const with_gs = dt == .f32 and g.chance(0.4);
        var ids: [2]u32 = .{ x, 0 };
        if (with_gs) ids[1] = try g.freshInputRange(Shape.init(.{}, .f32), 1, 4);
        const a: Attrs = .{ .kind = .nvfp4, .part = g.rng.enumValue(Part), .axis = .{ .index = @intCast(ax), .tag = TagId.of(sh.tag(ax)) }, .global_scale = with_gs };
        const out = switch (a.part) {
            .values => sh.withDtype(.f4e2m1),
            .scales => blk: {
                var s = sh.setDim(ax, @divExact(sh.dim(ax), 16)).withDtype(.f8e4m3fn);
                s._tags.set(ax, Shape.toTag(.sc));
                break :blk s;
            },
        };
        return .{ .op = .{ .quant = a }, .inputs = try k.dupeIds(g, ids[0 .. @as(usize, 1) + @intFromBool(with_gs)]), .shape = out };
    }

    fn proposeUnpack(g: *Gen) k.GenError!?k.Proposal {
        // Packed weights: bytes along the last (tagged) axis, two f4e2m1 codes each.
        const w = try g.operand(.{ .dtype = g.choose(DataType, &.{ .u8, .i8 }), .exact = true, .min_rank = 1, .max_rank = 3, .tagged = true });
        const sh = g.shapeOf(w);
        if (sh.dtype() != .u8 and sh.dtype() != .i8) return error.Reject;
        if (sh.hasTag(.bitcast) != null) return error.Reject;
        const last = sh.rank() - 1;
        // The contracted tag is the packed one renamed, or kept.
        const t = if (g.chance(0.5)) sh.tag(last) else g.freshTag(sh);
        const a: Attrs = .{ .kind = .unpack_fp4, .axis = .{ .index = @intCast(last), .tag = TagId.of(sh.tag(last)) }, .tag = TagId.of(t) };
        var out = sh.setDim(last, 2 * sh.dim(last)).withDtype(.f4e2m1);
        out._tags.set(last, t);
        return .{ .op = .{ .quant = a }, .inputs = try k.dupeIds(g, &.{w}), .shape = out };
    }

    fn proposeScaledDot(g: *Gen) k.GenError!?k.Proposal {
        const lhs = try g.operand(.{ .class = .float, .exact = true, .min_rank = 1, .max_rank = 3, .tagged = true });
        const ls = g.shapeOf(lhs);
        const c_ax = g.rng.uintLessThan(usize, ls.rank());
        // rhs: the contracting axis, maybe a batching axis, up to two free axes.
        var dims: [4]i64 = undefined;
        var tags: [4]Shape.Tag = undefined;
        dims[0] = ls.dim(c_ax);
        tags[0] = ls.tag(c_ax);
        var n: usize = 1;
        if (ls.rank() > 1 and g.chance(0.4)) {
            const b_ax = (c_ax + 1 + g.rng.uintLessThan(usize, ls.rank() - 1)) % ls.rank();
            dims[n] = ls.dim(b_ax);
            tags[n] = ls.tag(b_ax);
            n += 1;
        }
        for (0..g.intIn(usize, 0, 2)) |_| {
            const t = g.freshTag(ls);
            for (tags[0..n]) |u| if (u == t) return error.Reject;
            dims[n] = g.intIn(i64, 1, 4);
            tags[n] = t;
            n += 1;
        }
        var perm: [4]u8 = .{ 0, 1, 2, 3 };
        g.rng.shuffle(u8, perm[0..n]);
        var rd: [4]i64 = undefined;
        var rt: [4]Shape.Tag = undefined;
        for (0..n) |i| {
            rd[i] = dims[perm[i]];
            rt[i] = tags[perm[i]];
        }
        const rdt = g.randDtype(.float) orelse return error.Reject;
        const rhs = try g.operandLike(k.ir.makeShape(rdt, rd[0..n], rt[0..n]), true);
        // Scales have the operand's shape, in any float dtype.
        const lsc = try g.operandLike(ls.withDtype(g.randDtype(.float) orelse return error.Reject), true);
        const rsc = try g.operandLike(g.shapeOf(rhs).withDtype(g.randDtype(.float) orelse return error.Reject), true);
        const ids = [_]u32{ lhs, rhs, lsc, rsc };
        const t = TagId.of(ls.tag(c_ax));
        var c: [1][2]u8 = undefined;
        var b: [8][2]u8 = undefined;
        const nb = dotPairs(ls, g.shapeOf(rhs), t, &c, &b);
        const out = dotShape(ls, g.shapeOf(rhs), &c, b[0..nb]).withDtype(.bf16);
        if (out.rank() > Shape.MAX_RANK) return error.Reject;
        return .{ .op = .{ .quant = .{ .kind = .scaled_dot, .tag = t } }, .inputs = try k.dupeIds(g, &ids), .shape = out };
    }

    fn proposeGlobalScale(g: *Gen) k.GenError!?k.Proposal {
        const dt = g.randDtype(.float) orelse return error.Reject;
        const shapes = [_]Shape{ .init(.{}, dt), .init(.{1}, dt), .init(.{ 1, 1 }, dt), .init(.{g.intIn(i64, 2, 4)}, dt) };
        const sh = g.choose(Shape, &shapes);
        const x = try g.operandLike(sh, false);
        const op = g.rng.enumValue(GlobalScale.Operation);
        // A single value is flattened to a scalar, other shapes are kept.
        const out = if (sh.count() == 1) Shape.init(.{}, .f32) else g.shapeOf(x).withDtype(.f32);
        return .{ .op = .{ .quant = .{ .kind = .global_scale, .operation = op } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    fn hasScratchTags(sh: Shape, comptime names: []const @EnumLiteral()) bool {
        inline for (names) |n| if (sh.hasTag(n) != null) return true;
        return false;
    }

    /// Documented tag semantics of dot: the tag contracts, other shared tags batch.
    fn dotPairs(lhs: Shape, rhs: Shape, t: TagId, c: *[1][2]u8, b: *[8][2]u8) usize {
        c[0] = .{ lhs.hasTag(t.tag()).?, rhs.hasTag(t.tag()).? };
        var nb: usize = 0;
        for (0..lhs.rank()) |ax| {
            if (ax == c[0][0]) continue;
            if (rhs.hasTag(lhs.tag(ax))) |r| {
                b[nb] = .{ @intCast(ax), r };
                nb += 1;
            }
        }
        return nb;
    }

    fn dotShape(lhs: Shape, rhs: Shape, c: []const [2]u8, b: []const [2]u8) Shape {
        var res: Shape = .init(.{}, lhs.dtype());
        for (b) |p| res = res.appendDim(lhs.dim(p[0]), lhs.tag(p[0]));
        for (0..lhs.rank()) |ax| {
            if (inPairs(c, 0, ax) or inPairs(b, 0, ax)) continue;
            res = res.appendDim(lhs.dim(ax), lhs.tag(ax));
        }
        for (0..rhs.rank()) |ax| {
            if (inPairs(c, 1, ax) or inPairs(b, 1, ax)) continue;
            res = res.appendDim(rhs.dim(ax), rhs.tag(ax));
        }
        return res;
    }

    fn inPairs(pairs: []const [2]u8, side: usize, ax: usize) bool {
        for (pairs) |p| if (p[side] == ax) return true;
        return false;
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        for (ins) |v| if (v.err > 0) try e.reject();
        return switch (a.kind) {
            .block_fp8 => evalBlockFp8(e, a, node, ins[0]),
            .nvfp4 => evalNvfp4(e, a, node, ins),
            .unpack_fp4 => {
                // The bitcast splits each byte into two codes, the low nibble first: the
                // packed layout the interpreter stores f4e2m1 in.
                const v = try e.alloc(node.shape);
                @memcpy(v.bytes, ins[0].bytes);
                return v;
            },
            .scaled_dot => evalScaledDot(e, a, node, ins),
            .global_scale => {
                var v = try e.alloc(node.shape);
                for (0..v.count()) |i| {
                    const s: f32 = @floatCast(ins[0].getF(i));
                    const r: f32 = switch (a.operation) {
                        .multiply => s,
                        .divide => 1.0 / s,
                    };
                    if (!std.math.isFinite(r)) try e.reject();
                    v.setF(i, r);
                }
                return v;
            },
        };
    }

    /// Strides and index helpers over the blocks of `axis`.
    fn blockIndex(sh: Shape, idx: []const i64, axis: usize, block: i64, j: i64) usize {
        var full: [Shape.MAX_RANK]i64 = undefined;
        @memcpy(full[0..idx.len], idx);
        full[axis] = idx[axis] * block + j;
        return Value.offset(Value.strides(sh), full[0..sh.rank()]);
    }

    fn evalBlockFp8(e: *Eval, a: Attrs, node: *const Node, x: Value) Eval.Error!Value {
        const ax = a.axis.resolve(x.shape);
        const fp8_max: f32 = if (a.dtype == .f8e4m3fn) 448.0 else 224.0;
        const inv: f32 = 1.0 / fp8_max;
        const floor_: f32 = if (a.scale_dtype == .f32) 1e-10 else 1e-4;
        // Scales as f32, one per block.
        const ssh = x.shape.setDim(ax, @divExact(x.shape.dim(ax), a.block)).withDtype(.f32);
        var scales = try e.alloc(ssh);
        var it: Value.Index = .init(ssh);
        var i: usize = 0;
        while (!it.done) : (i += 1) {
            var m: f32 = 0;
            for (0..@intCast(a.block)) |j| {
                const xi: f32 = @floatCast(x.getF(blockIndex(x.shape, it.idx[0..ssh.rank()], ax, a.block, @intCast(j))));
                m = @max(m, @max(@abs(xi), floor_) * inv);
            }
            if (a.scale_dtype == .f8e8m0) {
                // Rounded up to a power of two.
                const bits: u32 = @bitCast(m);
                const exponent = bits >> 23;
                const fractional: u32 = @intFromBool(bits & 0x7fffff != 0);
                m = @bitCast((exponent + fractional) << 23);
            }
            scales.setF(i, m);
            _ = it.next();
        }
        if (a.part == .scales) {
            var v = try e.alloc(node.shape);
            for (0..v.count()) |s| v.setF(s, scales.getF(s));
            return v;
        }
        var v = try e.alloc(node.shape);
        var xit: Value.Index = .init(x.shape);
        i = 0;
        const sst = Value.strides(ssh);
        while (!xit.done) : (i += 1) {
            var sidx: [Shape.MAX_RANK]i64 = xit.idx;
            sidx[ax] = @divFloor(sidx[ax], a.block);
            const s: f32 = @floatCast(scales.getF(Value.offset(sst, sidx[0..ssh.rank()])));
            const xi: f32 = @floatCast(x.getF(i));
            const q = std.math.clamp(xi / s, -fp8_max, fp8_max);
            v.setF(i, q);
            _ = xit.next();
        }
        return v;
    }

    /// Rounds to the dtype ZML computes in, or keeps f32 when XLA skips the rounding.
    fn inDtype(dt: DataType, x: f32, round: bool) f32 {
        if (!round or dt == .f32) return x;
        return Value.bf16ToF32(Value.bf16Rne(x));
    }

    /// Fills the scales and values quantizeNvfp4 returns. A scale rounded down puts values
    /// slightly above 6: the conversion to f4e2m1 saturates them (observed, as XLA does
    /// for this format without inf or NaN).
    fn nvfp4(x: Value, gs: ?f32, axis: usize, round: bool, scales: Value, values: Value) void {
        const dt = x.dtype();
        const e4m3 = Value.MiniFloat.of(.f8e4m3fn);
        // `scale(1.0 / 6.0)` is a constant in x's dtype.
        const inv6: f32 = if (dt == .f32) @floatCast(1.0 / 6.0) else BFloat16.fromF32(@floatCast(1.0 / 6.0)).toF32(); // as Tensor.scalar rounds it
        const min_normal: f32 = 0x1p-6;
        const scaledAt = struct {
            fn f(xv: Value, i: usize, g: ?f32) f32 {
                const xi: f32 = @floatCast(xv.getF(i));
                return if (g) |s| xi / s else xi;
            }
        }.f;
        const ssh = scales.shape;
        var it: Value.Index = .init(ssh);
        var i: usize = 0;
        while (!it.done) : (i += 1) {
            var m: f32 = 0;
            for (0..16) |j| {
                const xi = scaledAt(x, blockIndex(x.shape, it.idx[0..ssh.rank()], axis, 16, @intCast(j)), gs);
                m = @max(m, std.math.clamp(inDtype(dt, @abs(xi) * inv6, round), min_normal, 448.0));
            }
            scales.bytes[i] = e4m3.encode(m);
            _ = it.next();
        }
        const sst = Value.strides(ssh);
        var xit: Value.Index = .init(x.shape);
        i = 0;
        while (!xit.done) : (i += 1) {
            var sidx: [Shape.MAX_RANK]i64 = xit.idx;
            sidx[axis] = @divFloor(sidx[axis], 16);
            const divisor: f32 = @max(@as(f32, @floatCast(e4m3.decode(scales.bytes[Value.offset(sst, sidx[0..ssh.rank()])]))), min_normal);
            const q = inDtype(dt, scaledAt(x, i, gs) / divisor, round);
            values.setF(i, q);
            _ = xit.next();
        }
    }

    fn evalNvfp4(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const ax = a.axis.resolve(x.shape);
        const gs: ?f32 = if (a.global_scale) @floatCast(ins[1].getF(0)) else null;
        const ssh = x.shape.setDim(ax, @divExact(x.shape.dim(ax), 16)).withDtype(.f8e4m3fn);
        var results: [2][2]Value = undefined;
        for (0..2) |r| {
            results[r] = .{ try e.alloc(ssh), try e.alloc(x.shape.withDtype(.f4e2m1)) };
            nvfp4(x, gs, ax, r == 0, results[r][0], results[r][1]);
            if (r == 1 and x.dtype() == .f32) break;
        }
        const rounded = results[0];
        if (x.dtype() != .f32) {
            // Results that depend on whether bf16 intermediates are rounded.
            if (!std.mem.eql(u8, rounded[0].bytes, results[1][0].bytes) or !std.mem.eql(u8, rounded[1].bytes, results[1][1].bytes)) try e.reject();
        }
        const v = try e.alloc(node.shape);
        @memcpy(v.bytes, switch (a.part) {
            .values => rounded[1].bytes,
            .scales => rounded[0].bytes,
        });
        return v;
    }

    fn evalScaledDot(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        // Each operand times its scale, both rounded to bf16, and the product rounded to bf16
        // unless XLA keeps it in f32.
        var real: [2]Value = undefined;
        for (0..2) |s| {
            const x = ins[s];
            const sc = ins[s + 2];
            var v = try e.alloc(x.shape.withDtype(.bf16));
            var err: f64 = 0;
            for (0..x.count()) |i| {
                const p = Value.bf16ToF32(Value.bf16Rne(@floatCast(x.getF(i)))) * Value.bf16ToF32(Value.bf16Rne(@floatCast(sc.getF(i))));
                v.setF(i, p);
                err = @max(err, @abs(v.getF(i) - p));
            }
            v.err = err;
            real[s] = v;
        }
        var c: [1][2]u8 = undefined;
        var b: [8][2]u8 = undefined;
        const nb = dotPairs(real[0].shape, real[1].shape, a.tag, &c, &b);
        return linalg.evalDot(e, node.shape, real[0], real[1], &c, b[0..nb]);
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        return switch (a.kind) {
            .block_fp8 => {
                const q = switch (a.axis.pass()) {
                    .tag => |t| zml.quantization.quantizeBlockFp8(ins[0], t, a.block, a.dtype, a.scale_dtype),
                    .index => |i| zml.quantization.quantizeBlockFp8(ins[0], i, a.block, a.dtype, a.scale_dtype),
                };
                return if (a.part == .values) q.values else q.scales;
            },
            .nvfp4 => {
                const q = zml.quantization.quantizeNvfp4(ins[0], if (a.global_scale) ins[1] else null, a.axis.tag.tag());
                return if (a.part == .values) q.values else q.scales;
            },
            .unpack_fp4 => zml.nn.unpackFp4(ins[0], a.axis.tag.tag(), a.tag.tag()),
            .scaled_dot => zml.ops.scaled_dot(ins[0], ins[1], ins[2], ins[3], a.tag.tag()),
            .global_scale => (GlobalScale{ .value = ins[0], .operation = a.operation }).asMultiplier(),
        };
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        switch (a.kind) {
            .block_fp8 => {
                try w.print("zml.quantization.quantizeBlockFp8({s}, ", .{ins[0]});
                try a.axis.print(w);
                try w.print(", {d}, .{s}, .{s}).{t}", .{ a.block, @tagName(a.dtype), @tagName(a.scale_dtype), a.part });
            },
            .nvfp4 => try w.print("zml.quantization.quantizeNvfp4({s}, {s}, .{s}).{t}", .{ ins[0], if (a.global_scale) ins[1] else "null", a.axis.tag.name(), a.part }),
            .unpack_fp4 => try w.print("zml.nn.unpackFp4({s}, .{s}, .{s})", .{ ins[0], a.axis.tag.name(), a.tag.name() }),
            .scaled_dot => try w.print("zml.ops.scaled_dot({s}, {s}, {s}, {s}, .{s})", .{ ins[0], ins[1], ins[2], ins[3], a.tag.name() }),
            .global_scale => try w.print("(zml.Quantization.GlobalScale{{ .value = {s}, .operation = .{t} }}).asMultiplier()", .{ ins[0], a.operation }),
        }
    }
};

/// zml.ops.composite with a ZML decomposition: CPU inlines it. Integer elementwise bodies
/// with one or two results, so the reference is exact.
pub const composite = struct {
    pub const Body = enum { add, sub, mul, max, min };
    pub const Attrs = struct {
        body: Body,
        /// The body also returns `b - a` as a second result.
        two_results: bool,
        /// Which result the node is.
        result: u1 = 0,
        /// Several composites may share a name.
        name: u2 = 0,
        version: i32 = 0,
    };
    pub const weight = 2;

    const names = [_][:0]const u8{ "zmlsmith.op", "zmlsmith.op", "zmlsmith.other", "zmlsmith.nested.op_$1" };

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const a = try g.operand(.{ .class = .int, .exact = true });
        const sh = g.shapeOf(a);
        const b = try g.operandLike(sh, false);
        const two = g.chance(0.5);
        const attrs: Attrs = .{
            .body = g.rng.enumValue(Body),
            .two_results = two,
            .result = if (two) g.rng.int(u1) else 0,
            .name = g.rng.int(u2),
            .version = g.intIn(i32, 0, 2),
        };
        // CPU inlines the decomposition, so XLA's clamp simplification applies to it too.
        if (attrs.body == .max or attrs.body == .min) {
            const nodes = g.prog.nodes.items;
            for ([2][2]u32{ .{ a, b }, .{ b, a } }) |p| {
                if (nodes[p[0]].op == .clamp) for (nodes[p[0]].inputs.slice()[1..]) |bound| try g.avoid(.xla_clamp_simplify, bound == p[1]);
            }
        }
        return .{ .op = .{ .composite = attrs }, .inputs = try k.dupeIds(g, &.{ a, b }), .shape = sh };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        var v = try e.alloc(node.shape);
        const dt = node.shape.dtype();
        for (0..v.count()) |i| {
            const x = ins[0].getI(i);
            const y = ins[1].getI(i);
            const r: i128 = if (a.result == 1) y - x else switch (a.body) {
                .add => x + y,
                .sub => x - y,
                .mul => x *% y,
                .max => @max(x, y),
                .min => @min(x, y),
            };
            v.setI(i, Value.wrapInt(dt, r));
        }
        return v;
    }

    fn decomposition(in: []const Tensor, ctx: Attrs) [2]Tensor {
        const x = in[0];
        const y = in[1];
        const r = switch (ctx.body) {
            .add => x.add(y),
            .sub => x.sub(y),
            .mul => x.mul(y),
            .max => x.maximum(y),
            .min => x.minimum(y),
        };
        return .{ r, y.sub(x) };
    }

    fn decomposition1(in: []const Tensor, ctx: Attrs) Tensor {
        return decomposition(in, ctx)[0];
    }

    pub fn lower(a: Attrs, node: *const Node, ins: []const Tensor) Tensor {
        const opts: zml.ops.CompositeOpts = .{ .version = a.version };
        if (a.two_results) return zml.ops.composite(names[a.name], ins[0..2], &.{ node.shape, node.shape }, decomposition, a, opts)[a.result];
        return zml.ops.composite(names[a.name], ins[0..2], &.{node.shape}, decomposition1, a, opts)[0];
    }

    pub fn emit(a: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const body = switch (a.body) {
            .add => "add",
            .sub => "sub",
            .mul => "mul",
            .max => "maximum",
            .min => "minimum",
        };
        try w.print("zml.ops.composite(\"{s}\", &.{{ {s}, {s} }}, &.{{ ", .{ names[a.name], ins[0], ins[1] });
        try k.printShapeLiteral(w, node.shape);
        if (a.two_results) {
            try w.writeAll(", ");
            try k.printShapeLiteral(w, node.shape);
        }
        try w.print(" }}, struct {{\n                fn f(in: []const zml.Tensor, _: void) {s} {{\n                    return .{{ in[0].{s}(in[1]){s} }};\n                }}\n            }}.f, {{}}, .{{ .version = {d} }})[{d}]", .{
            if (a.two_results) "[2]zml.Tensor" else "[1]zml.Tensor",
            body,
            if (a.two_results) ", in[1].sub(in[0])" else "",
            a.version,
            a.result,
        });
    }
};
