//! Host-only property tests: zml.floats conversions against an independent codec, and
//! zml.Shape invariants on random shapes. No device needed, so they run fast.

const std = @import("std");

const zml = @import("zml");
const Shape = zml.Shape;
const DataType = zml.DataType;

const ir = @import("ir.zig");
const Value = @import("Value.zig");

const Failures = struct {
    allocator: std.mem.Allocator,
    seen: std.StringArrayHashMapUnmanaged(usize) = .empty,
    examples: std.StringArrayHashMapUnmanaged([]const u8) = .empty,

    fn add(f: *Failures, sig: []const u8, comptime fmt: []const u8, args: anytype) !void {
        const gop = try f.seen.getOrPut(f.allocator, sig);
        if (!gop.found_existing) {
            gop.value_ptr.* = 0;
            try f.examples.put(f.allocator, sig, try std.fmt.allocPrint(f.allocator, fmt, args));
        }
        gop.value_ptr.* += 1;
    }
};

/// Every code of every f8 and f4 format, through zml.floats toF32 and fromF32.
fn checkFloats(f: *Failures) !void {
    inline for (.{ .f8e4m3fn, .f8e4m3fnuz, .f8e4m3b11fnuz, .f8e5m2, .f8e5m2fnuz, .f8e8m0, .f8e4m3, .f8e3m4 }) |dt| {
        const T = DataType.toZigType(dt);
        const mf = Value.MiniFloat.of(dt);
        for (0..256) |c| {
            const code: u8 = @intCast(c);
            const want = mf.decode(code);
            const got: f64 = @as(T, @bitCast(code)).toF32();
            const same = (std.math.isNan(want) and std.math.isNan(got)) or want == got;
            if (!same) try f.add("floats.toF32:" ++ @tagName(dt), "{s}: code 0x{x:0>2} decodes to {d}, expected {d}", .{ @tagName(dt), code, got, want });
            if (std.math.isFinite(want)) {
                const back: u8 = @bitCast(T.fromF32(@floatCast(want)));
                const back_v = mf.decode(back);
                if (back_v != want and !(want == 0 and back_v == 0)) try f.add("floats.fromF32:" ++ @tagName(dt), "{s}: fromF32({d}) gives code 0x{x:0>2} = {d}", .{ @tagName(dt), want, back, back_v });
            }
        }
        // -0.0 has no encoding in FNUZ formats, where 0x80 is NaN.
        if (mf.fnuz) {
            const code: u8 = @bitCast(T.fromF32(-0.0));
            if (code == 0x80) try f.add("floats.negzero:" ++ @tagName(dt), "{s}: fromF32(-0.0) gives 0x80, the NaN encoding", .{@tagName(dt)});
        }
    }
}

/// fromF32 on values that need rounding: every midpoint between adjacent finite codes and
/// its neighbors, against round to nearest even. Also bf16, through random f32 values.
fn checkFloatRounding(f: *Failures, seed: u64) !void {
    inline for (.{ .f8e4m3fn, .f8e4m3fnuz, .f8e4m3b11fnuz, .f8e5m2, .f8e5m2fnuz, .f8e4m3, .f8e3m4, .f4e2m1 }) |dt| {
        const T = DataType.toZigType(dt);
        const Bits = @typeInfo(T).@"struct".backing_integer.?;
        const mf = Value.MiniFloat.of(dt);
        const n_codes: usize = @as(usize, 1) << @bitSizeOf(Bits);
        // Positive finite values in increasing order.
        var vals: [256]f64 = undefined;
        var nv: usize = 0;
        for (0..n_codes / 2) |c| {
            const v = mf.decode(@intCast(c));
            if (std.math.isFinite(v)) {
                vals[nv] = v;
                nv += 1;
            }
        }
        std.mem.sort(f64, vals[0..nv], {}, std.sort.asc(f64));
        for (0..nv - 1) |j| {
            const mid = (vals[j] + vals[j + 1]) / 2;
            for ([_]f64{ mid, mid * (1 - 0x1p-20), mid * (1 + 0x1p-20) }) |x| {
                const x32: f32 = @floatCast(x);
                if (@as(f64, x32) != x) continue;
                const want = mf.decode(mf.encode(x));
                const got = mf.decode(@as(Bits, @bitCast(T.fromF32(x32))));
                if (got != want) try f.add("floats.fromF32_rounding:" ++ @tagName(dt), "{s}: fromF32({d}) = {d}, round to nearest even gives {d}", .{ @tagName(dt), x, got, want });
            }
        }
    }
    var prng: std.Random.DefaultPrng = .init(seed);
    const rng = prng.random();
    for (0..20000) |j| {
        var bits: u32 = rng.int(u32) & 0x7fffffff;
        // Exact ties half of the time.
        if (j % 2 == 0) bits = (bits & 0xffff0000) | 0x8000;
        const x: f32 = @bitCast(bits);
        if (!std.math.isFinite(x) or @abs(x) > 1e38) continue;
        const lsb = (bits >> 16) & 1;
        const want: u16 = @intCast((bits + 0x7fff + lsb) >> 16);
        const got: u16 = @bitCast(zml.floats.BFloat16.fromF32(x));
        if (got != want) try f.add("floats.fromF32_rounding:bf16", "bf16: fromF32({d}) gives 0x{x:0>4}, round to nearest even gives 0x{x:0>4}", .{ x, got, want });
    }
}

fn randShape(rng: std.Random) Shape {
    const rank = rng.intRangeAtMost(usize, 0, Shape.MAX_RANK);
    var dims: [8]i64 = undefined;
    var tags: [8]Shape.Tag = undefined;
    var pool = ir.tag_pool;
    rng.shuffle(Shape.Tag, &pool);
    for (0..rank) |i| {
        dims[i] = rng.intRangeAtMost(i64, 0, 5);
        tags[i] = if (rng.boolean()) pool[i] else Shape.TagUnknown;
    }
    const dts = [_]DataType{ .f32, .bf16, .i4, .u2, .bool, .f64, .u8, .f4e2m1 };
    return ir.makeShape(dts[rng.uintLessThan(usize, dts.len)], dims[0..rank], tags[0..rank]);
}

/// Invariants of the pure Shape API.
fn checkShapes(f: *Failures, seed: u64, count: usize) !void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const rng = prng.random();
    for (0..count) |_| {
        const sh = randShape(rng);
        const r = sh.rank();
        // count and byteSize.
        var c: usize = 1;
        for (sh.dims()) |d| c *= @intCast(d);
        if (sh.count() != c) try f.add("shape.count", "{f}.count() = {d}, expected {d}", .{ sh, sh.count(), c });
        const want_bytes = std.math.divCeil(usize, c * sh.dtype().bitSizeOf(), 8) catch unreachable;
        if (sh.byteSize() != want_bytes) try f.add("shape.byteSize", "{f}.byteSize() = {d}, expected {d}", .{ sh, sh.byteSize(), want_bytes });
        // Negative and positive axes.
        for (0..r) |ax| {
            const neg: i8 = @intCast(@as(i64, @intCast(ax)) - r);
            if (sh.axis(neg) != ax) try f.add("shape.axis_negative", "{f}.axis({d}) = {d}, expected {d}", .{ sh, neg, sh.axis(neg), ax });
            const t = sh.tag(ax);
            if (t != Shape.TagUnknown and sh.axis(t) != ax) try f.add("shape.axis_tag", "{f}.axis(.{s}) = {d}, expected {d}", .{ sh, t, sh.axis(t), ax });
        }
        if (r == 0) continue;
        // transpose then the inverse permutation is the identity.
        var perm: [8]i64 = undefined;
        for (0..r) |i| perm[i] = @intCast(i);
        rng.shuffle(i64, perm[0..r]);
        var inv: [8]i64 = undefined;
        for (0..r) |i| inv[@intCast(perm[i])] = @intCast(i);
        const tr = sh.transpose(@as([]const i64, perm[0..r])).transpose(@as([]const i64, inv[0..r]));
        if (!tr.eqlWithTags(sh)) try f.add("shape.transpose_inverse", "{f}.transpose(p).transpose(p^-1) = {f}", .{ sh, tr });
        // insert then remove is the identity. At max rank insert hits `catch unreachable`
        // instead of a clear error: skipped.
        if (r == Shape.MAX_RANK) continue;
        const at = rng.uintLessThan(usize, r + 1);
        const ins = if (at == r) sh.appendDim(7, null) else sh.insert(@as(i8, @intCast(at)), .{7});
        if (ins.dim(at) != 7 or !ins.remove(@as(i8, @intCast(at))).eqlWithTags(sh)) try f.add("shape.insert_remove", "{f}.insert({d}, 7) = {f}", .{ sh, at, ins });
        // reshape to the flat shape keeps the count. With a zero dim, inferMissingAxis
        // compares the product of the non zero dims to 0 and panics "TooBig": found by this
        // test, skipped so the other checks run.
        if (sh.count() > 0 and sh.reshape(.{-1}).count() != sh.count()) try f.add("shape.reshape_auto", "{f}.reshape(.{{-1}})", .{sh});
        // packedShape: the bytes the device holds.
        if (sh.dtype().bitSizeOf() < 8) {
            const per = 8 / sh.dtype().bitSizeOf();
            if (@mod(sh.dim(-1), per) == 0) {
                const p = sh.packedShape();
                if (p.byteSize() != sh.byteSize()) try f.add("shape.packedShape_bytes", "{f}.packedShape() = {f} holds {d} bytes, {d} needed", .{ sh, p, p.byteSize(), sh.byteSize() });
            }
        }
        // splitAxis then merge restores the shape.
        const ax = rng.uintLessThan(usize, r);
        const t = sh.tag(ax);
        if (t != Shape.TagUnknown and sh.dim(ax) > 0 and @mod(sh.dim(ax), 1) == 0) {
            const split = sh.splitAxis(@as(i8, @intCast(ax)), ir.makeShape(.i32, &.{ 1, sh.dim(ax) }, &.{ Shape.toTag(.coord), Shape.toTag(.bitcast) }));
            const merged = split.mergeAxis(t, @as([]const Shape.Tag, &.{ Shape.toTag(.coord), Shape.toTag(.bitcast) }));
            if (!merged.eqlWithTags(sh)) try f.add("shape.split_merge", "{f}.splitAxis then mergeAxis = {f}", .{ sh, merged });
        }
        // `._N` names axis N: it exists only below the rank.
        {
            const lits = [_]Shape.Tag{ Shape.toTag(._0), Shape.toTag(._1), Shape.toTag(._2), Shape.toTag(._3), Shape.toTag(._4), Shape.toTag(._5), Shape.toTag(._6), Shape.toTag(._7) };
            for (lits, 0..) |lt, n| {
                const got = sh.hasTag(lt);
                const want: ?u3 = if (n < r) @intCast(n) else null;
                if (got != want) try f.add("shape.hasTag_literal_int", "{f}.hasTag(._{d}) = {?d}, expected {?d}", .{ sh, n, got, want });
            }
        }
        // mergeAxes of two adjacent tagged axes, then splitAxes back.
        if (r >= 2) {
            const m0 = rng.uintLessThan(usize, r - 1);
            const t0 = sh.tag(m0);
            const t1 = sh.tag(m0 + 1);
            if (t0 != Shape.TagUnknown and t1 != Shape.TagUnknown and sh.hasTag(.sc) == null) {
                const merged = sh.mergeAxes(.{ .sc = @as([]const Shape.Tag, &.{ t0, t1 }) });
                if (merged.rank() != r - 1 or merged.dim(m0) != sh.dim(m0) * sh.dim(m0 + 1) or merged.tag(m0) != Shape.toTag(.sc))
                    try f.add("shape.mergeAxes", "{f}.mergeAxes(.{{ .sc = .{{ {d}, {d} }} }}) = {f}", .{ sh, m0, m0 + 1, merged });
                if (sh.dim(m0) * sh.dim(m0 + 1) > 0) {
                    const back = merged.splitAxes(.{ .sc = ir.makeShape(.i32, &.{ sh.dim(m0), sh.dim(m0 + 1) }, &.{ t0, t1 }) });
                    if (!back.eqlWithTags(sh)) try f.add("shape.mergeAxes_splitAxes", "{f}.mergeAxes then splitAxes = {f}", .{ sh, back });
                }
            }
        }
        // Partition specs: lookup and renaming of mesh axes.
        {
            var p = sh.withDefaultPartitioning();
            p._partitioning.set(0, .init(.data));
            const mapped = p.mapPartitioningAxes(.{ .data = .model, .model = .data });
            if (!mapped.partition(0).eql(.init(.model))) try f.add("shape.mapPartitioningAxes", "{f} .data mapped to .model gives {any}", .{ p, mapped.partition(0) });
            if (!p.containsPartitionSpec(.init(.data)) or p.containsPartitionSpec(.init(.model))) try f.add("shape.containsPartitionSpec", "{f}", .{p});
            if (mapped.containsPartitionSpec(.init(.data))) try f.add("shape.mapPartitioningAxes_swap", "{f} swapping .data and .model maps .data back", .{p});
            if (!sh.withReplicatedPartitioning().containsPartitionSpec(.replicated) or sh.withReplicatedPartitioning().containsPartitionSpec(.open))
                try f.add("shape.containsPartitionSpec_replicated", "{f}", .{sh});
            if (!Shape.PartitionSpec.init(.data).isClosed() or (@as(Shape.PartitionSpec, .open)).isClosed()) try f.add("shape.isClosed", "", .{});
        }
        // Strides: row major, in elements and bytes.
        {
            const es = sh.computeElementStrides();
            var acc: i64 = 1;
            var j: usize = r;
            while (j > 0) {
                j -= 1;
                if (es.get(j) != acc) try f.add("shape.computeElementStrides", "{f}: stride[{d}] = {d}, expected {d}", .{ sh, j, es.get(j), acc });
                acc *= sh.dim(j);
            }
            if (sh.dtype().bitSizeOf() >= 8) {
                const bs = sh.computeByteStrides();
                for (0..r) |jj| if (bs.get(jj) != es.get(jj) * sh.dtype().sizeOf()) try f.add("shape.computeByteStrides", "{f}: byte stride[{d}] = {d}, expected {d}", .{ sh, jj, bs.get(jj), es.get(jj) * sh.dtype().sizeOf() });
            }
        }
        // Tag queries.
        {
            var all = true;
            for (0..r) |jj| if (sh.tag(jj) == Shape.TagUnknown) {
                all = false;
            };
            if (sh.isFullyTagged() != all) try f.add("shape.isFullyTagged", "{f}.isFullyTagged() = {}", .{ sh, sh.isFullyTagged() });
            for (0..r) |jj| {
                const tg = sh.tag(jj);
                if (tg != Shape.TagUnknown and !sh.hasTags(@as([]const Shape.Tag, &.{tg}))) try f.add("shape.hasTags", "{f}.hasTags(.{s}) is false", .{ sh, tg });
                const dbg = sh.debugTag(jj);
                if (tg == Shape.TagUnknown and !(dbg.len == 1 and dbg[0] == '0' + @as(u8, @intCast(jj)))) try f.add("shape.debugTag", "{f}.debugTag({d}) = {s}", .{ sh, jj, dbg });
            }
            const fresh = Shape.toTag(.voc);
            if (sh.setTag(0, fresh).tag(0) != fresh) try f.add("shape.setTag", "{f}.setTag(0, .voc)", .{sh});
        }
        // removeMany removes exactly the given axes, in any order.
        if (r >= 2) {
            const rm = sh.removeMany(.{ @as(i8, @intCast(r - 1)), 0 });
            if (rm.rank() != r - 2) try f.add("shape.removeMany", "{f}.removeMany(.{{-1, 0}}) = {f}", .{ sh, rm });
            for (0..r - 2) |jj| if (rm.dim(jj) != sh.dim(jj + 1)) try f.add("shape.removeMany", "{f}.removeMany(.{{-1, 0}}) = {f}", .{ sh, rm });
        }
        // eqlDims ignores tags and dtype.
        if (!sh.eqlDims(sh.withDtype(.u8).withPartialTags(@as([]const Shape.Tag, &.{})))) try f.add("shape.eqlDims", "{f}.eqlDims(itself as u8)", .{sh});
        // canBroadcastTo: same rank, size-1 or equal dims, is documented as broadcastable.
        {
            var target = sh;
            for (0..r) |jj| if (sh.dim(jj) == 1) {
                target = target.setDim(jj, 3);
            };
            if (!sh.canBroadcastTo(target)) try f.add("shape.canBroadcastTo", "{f}.canBroadcastTo({f}) is false", .{ sh, target });
        }
        // contiguousPerm moves the given axes last, keeping the others in order.
        if (r >= 2) {
            const perm_c = sh.contiguousPerm(.{@as(i8, 0)});
            if (perm_c.get(r - 1) != 0) try f.add("shape.contiguousPerm", "{f}.contiguousPerm(.{{0}}) = {any}", .{ sh, perm_c.constSlice() });
        }
        // Partitioning predicates on a fresh shape: nothing partitioned.
        if (sh.hasAtLeastOnePartitionedAxis()) try f.add("shape.hasAtLeastOnePartitionedAxis", "{f} without partitioning", .{sh});
        {
            const rep = sh.withReplicatedPartitioning();
            if (!rep.isFullyReplicated()) try f.add("shape.withReplicatedPartitioning", "{f}.withReplicatedPartitioning() is not fully replicated", .{sh});
            if (rep.isFullyPartitioned() and r > 0) try f.add("shape.isFullyPartitioned", "{f} replicated reported fully partitioned", .{sh});
        }
        // range(n) is untagged 0..n.
        {
            const rg = Shape.range(r, .i32);
            for (0..r) |jj| if (rg.dim(jj) != jj) try f.add("shape.range", "Shape.range({d}) = {f}", .{ r, rg });
        }
        // An axis index equal to the rank is out of bounds.
        if (rng.uintLessThan(u8, 8) == 0) {
            // Shape.axisFromInt accepts it (KnownBug: off by one); only recorded once.
            const res = sh.axisFromInt(@intCast(r));
            if (res == r) try f.add("shape.axisFromInt_rank", "{f}.axisFromInt({d}) returns {d} instead of failing", .{ sh, r, res });
        }
    }
}

/// zml.Slice views (chains of subSlice and squeeze) against an independent model: each
/// view axis maps to an original axis and start, squeezed axes to a fixed coordinate.
fn checkSlices(allocator: std.mem.Allocator, f: *Failures, seed: u64, count: usize) !void {
    var prng: std.Random.DefaultPrng = .init(seed ^ 0x511ce);
    const rng = prng.random();
    inline for (.{ u8, u16, u32, u64 }) |T| {
        for (0..count / 4) |_| try checkSlice(T, allocator, f, rng);
    }
}

fn checkSlice(comptime T: type, allocator: std.mem.Allocator, f: *Failures, rng: std.Random) !void {
    const dt = DataType.fromZigType(T);
    const rank = rng.intRangeAtMost(usize, 1, 4);
    var dims: [4]i64 = undefined;
    for (dims[0..rank]) |*d| d.* = rng.intRangeAtMost(i64, 1, 5);
    var full: Shape = .init(.{}, dt);
    for (dims[0..rank]) |d| full = full.appendDim(d, null);
    const n = full.count();
    const data = try allocator.alloc(T, n);
    defer allocator.free(data);
    for (data, 0..) |*x, i| x.* = @truncate(i *% 2654435761 +% 7);
    var view = zml.Slice.init(full, std.mem.sliceAsBytes(data));

    // Model: for each original axis, the view axis it became (or null), its start and,
    // when squeezed, its fixed coordinate.
    var view_axis: [4]?usize = .{ 0, 1, 2, 3 };
    var start: [4]i64 = @splat(0);
    var ops: [64]u8 = undefined;
    var n_ops: usize = 0;
    for (0..rng.intRangeAtMost(usize, 0, 4)) |_| {
        if (view.shape.rank() == 0) break;
        const ax = rng.uintLessThan(usize, view.shape.rank());
        const d = view.shape.dim(ax);
        if (d == 0) break;
        var orig: usize = 0;
        for (view_axis[0..rank], 0..) |va, o| if (va == ax) {
            orig = o;
        };
        if (d == 1 and rng.boolean()) {
            view = view.squeeze(ax);
            for (&view_axis) |*va| if (va.*) |v| {
                if (v == ax) va.* = null else if (v > ax) va.* = v - 1;
            };
            ops[n_ops] = 'q';
        } else {
            const st = rng.intRangeAtMost(i64, 0, d - 1);
            const len = rng.intRangeAtMost(i64, 0, d - st);
            view = view.subSlice(ax, st, len);
            start[orig] += st;
            ops[n_ops] = 's';
        }
        n_ops += 1;
    }

    // Expected elements of the view, row-major, from the model.
    const vn = view.shape.count();
    const want = try allocator.alloc(T, vn);
    defer allocator.free(want);
    const vr = view.shape.rank();
    var idx: [4]i64 = @splat(0);
    for (0..vn) |i| {
        var o: usize = 0;
        for (0..rank) |a| {
            const c = start[a] + if (view_axis[a]) |v| idx[v] else 0;
            o = o * @as(usize, @intCast(dims[a])) + @as(usize, @intCast(c));
        }
        want[i] = data[o];
        var a = vr;
        while (a > 0) {
            a -= 1;
            idx[a] += 1;
            if (idx[a] < view.shape.dim(a)) break;
            idx[a] = 0;
        }
    }

    // contiguousItemsIterator: the runs concatenate to the view in row-major order.
    var got: std.ArrayList(T) = .empty;
    defer got.deinit(allocator);
    var it = view.contiguousItemsIterator(T);
    while (it.next()) |chunk| try got.appendSlice(allocator, chunk);
    if (!std.mem.eql(T, got.items, want)) try f.add("slice.contiguousItemsIterator", "{f} after {s}: {d} items, expected {d}", .{ view.shape, ops[0..n_ops], got.items.len, want.len });

    // constData agrees when contiguous.
    if (view.isContiguous() and vn > 0 and !std.mem.eql(u8, view.constData(), std.mem.sliceAsBytes(want)))
        try f.add("slice.constData", "{f} after {s}: contiguous data differs", .{ view.shape, ops[0..n_ops] });

    // elementStrides is byte_strides over the element size.
    for (view.elementStrides().slice(), view.byte_strides.slice()) |es, bs| {
        if (es * @as(i64, @sizeOf(T)) != bs) try f.add("slice.elementStrides", "{f}: {d} vs byte stride {d}", .{ view.shape, es, bs });
    }

    // copy into the same view of a zeroed buffer writes exactly the view's positions.
    const zeros = try allocator.alloc(T, n);
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var dst = zml.Slice.init(full, std.mem.sliceAsBytes(zeros));
    dst.offset_bytes = view.offset_bytes;
    dst.shape = view.shape;
    dst.byte_strides = view.byte_strides;
    if (vn > 0 and (vr > 0 or view.isContiguous())) {
        dst.copy(std.mem.sliceAsBytes(want));
        var back: std.ArrayList(T) = .empty;
        defer back.deinit(allocator);
        var it2 = dst.contiguousItemsIterator(T);
        while (it2.next()) |chunk| try back.appendSlice(allocator, chunk);
        var written: usize = 0;
        for (zeros, data) |z, d| written += @intFromBool(z != 0 and z == d);
        if (!std.mem.eql(T, back.items, want)) try f.add("slice.copy", "{f} after {s}: copied view reads back differently", .{ view.shape, ops[0..n_ops] });
        if (written > vn) try f.add("slice.copy_outside", "{f} after {s}: copy wrote outside the view", .{ view.shape, ops[0..n_ops] });
    }
}

/// Formatting zml.Slice for every dtype, rank 0 to 3, empty or not: it must not panic.
fn checkSliceFormat(allocator: std.mem.Allocator, f: *Failures, seed: u64, count: usize) !void {
    var prng: std.Random.DefaultPrng = .init(seed ^ 0xf0f0);
    const rng = prng.random();
    for (0..count / 8) |_| {
        const dts = comptime std.enums.values(DataType);
        const dt = dts[rng.uintLessThan(usize, dts.len)];
        const rank = rng.intRangeAtMost(usize, 0, 3);
        var dims: [3]i64 = undefined;
        for (dims[0..rank]) |*d| d.* = rng.intRangeAtMost(i64, 0, 4);
        // Sub-byte dtypes pack along the last axis.
        if (dt.bitSizeOf() < 8 and rank > 0) dims[rank - 1] = 2 * rng.intRangeAtMost(i64, 0, 2);
        if (dt.bitSizeOf() < 8 and rank == 0) continue;
        // Sub-byte byte strides are wrong from rank 2 (found/slice_subbyte.zig).
        if (dt.bitSizeOf() < 8 and rank >= 2) continue;
        var sh: Shape = .init(.{}, dt);
        for (dims[0..rank]) |d| sh = sh.appendDim(d, null);
        const bytes = try allocator.alloc(u8, sh.byteSize());
        defer allocator.free(bytes);
        rng.bytes(bytes);
        const s: zml.Slice = .init(sh, bytes);
        var w: std.Io.Writer.Allocating = .init(allocator);
        defer w.deinit();
        w.writer.print("{f}", .{s}) catch |err| try f.add("slice.format", "{f}: {t}", .{ sh, err });
        w.writer.print("{d}", .{s}) catch |err| try f.add("slice.formatNumber", "{f}: {t}", .{ sh, err });
    }
}

pub fn run(allocator: std.mem.Allocator, seed: u64, count: usize) !usize {
    var f: Failures = .{ .allocator = allocator };
    try checkFloats(&f);
    try checkFloatRounding(&f, seed);
    try checkShapes(&f, seed, count);
    try checkSlices(allocator, &f, seed, count);
    try checkSliceFormat(allocator, &f, seed, count);
    var it = f.seen.iterator();
    while (it.next()) |e| {
        std.debug.print("RESULT host kind=host sig={s} count={d}\n    {s}\n", .{ e.key_ptr.*, e.value_ptr.*, f.examples.get(e.key_ptr.*).? });
    }
    std.debug.print("host: {d} shapes, {d} distinct failures\n", .{ count, f.seen.count() });
    return f.seen.count();
}
