//! Random program generation.
//!
//! The generator evaluates the reference interpreter while it builds the program, so every
//! decision can look at concrete values: an op whose result would be non-finite, too large,
//! too imprecise to compare, or sitting on a discontinuity (floor, compare, argmax) of an
//! imprecise input is rejected and another one is tried. Programs are valid by construction.

const std = @import("std");

const zml = @import("zml");
const Shape = zml.Shape;
const DataType = zml.DataType;

const ir = @import("ir.zig");
const ops = @import("ops.zig");
const Value = @import("Value.zig");

pub const Error = error{ OutOfMemory, Reject };

/// ZML bugs found by zmlsmith. While a bug is open the generator avoids triggering it, so
/// it doesn't mask new ones; `--known` generates them again, to check fixes.
pub const KnownBug = enum {
    /// Tensor.Slice.absolute ignores `.len` because `.end` defaults to `to_the_end`.
    slice_len,
    /// linspace scales iota by `steps` instead of (end - start) / (steps - 1).
    linspace,
    /// outer broadcasts with `broad`, which needs tags when ranks differ.
    outer_untagged,
    /// dynamicUpdateSlice1d mixes i32 placeholder offsets with the offset dtype.
    dus1d_offset_dtype,
    /// unflatten shifts negative axes by one.
    unflatten_negative_axis,
    /// gather with a batching axis after the gathered axis trips ZML's shape check.
    gather_batching_after,
    /// cartesianProduct broadcasts rank 0 vectors with a one axis mapping.
    cartesian_scalar,
    /// fmod builds an f32 divisor whatever the input dtype.
    fmod_non_f32,
    /// chunkAllowTrailing computes the tail as d % chunk_size and drops elements.
    chunk_trailing,
    /// maxPool indices pick the init index 0 when the max equals the dtype minimum.
    pool_min_index,
    /// scatterSlices doesn't transpose updates whose window axes aren't in operand order
    /// (TODO in scatterPrepareIndices): invalid MLIR, or silently wrong values.
    scatter_update_order,
    /// The untagged scatterSlices branch assumes updates.rank() >= number of indices.
    scatter_untagged_rank,
    /// Scalar indices are rewritten to dynamicUpdateSlice, which ignores update_fn and
    /// requires the update to have the operand rank and axis order.
    scatter_scalar_indices,
    /// nn.rope slices axis -1 instead of .hd (splitRealImg) and merges assuming .hd is last.
    rope_hd_not_last,
    /// nn.nearest and nn.upsample drop the tags of the resized axes.
    nearest_drops_tags,
    /// nn.resizeCubic1d gathers 4 neighbors: it crashes on axes smaller than 4.
    cubic_small_axis,
    /// nn.resizeCubic1d moves the axis back with swapAxes(0, axis): wrong for axis >= 2.
    cubic_axis_order,
    /// Rng.uniform on bf16 shifts u16 random bits by 8 - 7 and keeps garbage in the
    /// exponent: values fall outside [min, max).
    uniform_bf16,
    /// broad, broadcastLeft and broadcastRight return self unchanged when the dims already
    /// match, keeping self's tags instead of the target's ones.
    broad_equal_dims_tags,
    /// ops.while registers loop state block arguments by tensor id: the same tensor in two
    /// state slots aliases them and the loop returns wrong values.
    while_aliased_state,
    /// Upstream XLA: two chained narrowing bitcasts fail the HLO verifier after
    /// simplification (expects u8[.., 8], gets u8[.., 4, 2]).
    xla_bitcast_chain,
    /// Partitioning metadata survives ops that change the partitioned dim (reductions, pad):
    /// the result still claims a sharded axis and Buffer.toSliceAlloc panics. Avoiding it
    /// disables sharded arguments, which also hides: reuseBuffer aliasing a sharded input to
    /// a replicated output (compile error), and upstream XLA SPMD failures on strided
    /// slices (HLO verifier) and `or` reductions (Unimplemented). See regressions/found.
    sharded_reduced_axis,
    /// Upstream XLA SPMD: gathers with 8 or 16 bit indices abort the compiler
    /// (gather_scatter_handler.cc "Unsupported indices type").
    xla_spmd_small_indices,
    /// Shape.packedShape uses divExact on the last dim: sub-byte arguments and results
    /// whose last dim isn't a multiple of the items per byte (or of rank 0) panic.
    subbyte_last_dim,
    /// Sharded buffers of sub-byte dtypes overrun the host slice in Buffer.toSliceAlloc:
    /// shard offsets are computed per element without the packing.
    sharded_subbyte,
    /// Returning a reuseBuffer result twice panics ("reused twice"): both outputs claim
    /// the same donation although the program is valid.
    reuse_duplicate_output,
    /// Tensor.any calls Tensor.constant with two arguments: it doesn't compile.
    any_compile,
    /// Tensor.all's reduce body returns a bare Tensor instead of a tuple: it doesn't compile.
    all_compile,
    /// zml.ops.gather treats a `.coord` axis of the indices as the index vector dim even
    /// when the operand also has a `.coord` axis: it becomes a batching dim and MLIR
    /// rejects the gather ("expects start_indices_batching_dims not to include
    /// index_vector_dim").
    gather_coord_tag,
    /// Sharded reduce_window with base dilation and padding returns shifted values once the
    /// result sharding is propagated from a sharded argument.
    sharded_reduce_window,
    /// Empty buffers with a sharded axis overflow in Sharding.shardSlice (Buffer.toSliceAlloc).
    sharded_empty,
    /// Compiler.repack splits the last dim into `.{ -1, items_per_byte }`: sub-byte
    /// arguments and results with no elements panic ("Can't split ... TooBig").
    subbyte_empty,
    /// Upstream XLA: converting f4e2m1 to any integer type saturates +-6.0 (the largest
    /// code) to the integer type's max/min instead of 6/-6.
    xla_f4_to_int,
    /// Shape.inferMissingAxis panics "TooBig" when a -1 / .auto dim is inferred for a
    /// shape with a zero dim (host check; the reshape family doesn't build empty reshapes).
    reshape_zero,
    /// Shape.axisFromInt accepts `a == rank` (host check).
    axis_from_int_rank,
    /// The `._N` axis syntax isn't checked against the rank: hasTag(._5) on rank 2 is 5.
    literal_axis_rank,
    /// leakyReLU computes relu(x) + min(slope * x, 0): wrong for a negative slope. The
    /// activation family only uses positive slopes.
    leaky_negative_slope,
    /// Tensor.slices mixing static and dynamic slices sets static dims to `to_the_end`. The
    /// generator only builds dynamic slices on one axis.
    slices_mixed_static,
    /// Tensor.broadcast with dims already equal ignores a permuting `axes_`. The broadcast
    /// family always adds an axis.
    broadcast_equal_dims_perm,
    /// Rng.uniform on integers samples floats in [min, max) and truncates toward zero: a
    /// negative min is never produced and 0 comes out twice as often.
    uniform_int_bias,
    /// ZML doesn't disable XLA CPU's fast min/max (xla_cpu_enable_fast_min_max, on by
    /// default): maximum, minimum, clamp and max/min reductions drop NaN on CPU.
    cpu_fast_min_max,
    /// Upstream XLA: maximum(clamp(lo, x, hi), lo) (and minimum with hi) is simplified to
    /// the clamp, assuming lo <= hi, which StableHLO doesn't require.
    xla_clamp_simplify,
    /// Upstream XLA: `x + 0` with a constant zero is simplified to `x`, so -0 + 0 gives -0
    /// instead of +0 (addConstant(0)).
    xla_add_zero_sign,
    /// Upstream XLA CPU: a convolution with window reversal, a stride > 1 and padding writes a
    /// result of the wrong size; reading it back fails (regressions/found/xla_conv_rev.zig).
    xla_conv_reversal,
    /// nn.resizeBilinear / resizeBicubic drop ResizeOpts.precision. ResizeOpts isn't
    /// generated.
    resize_drops_precision,
    /// nn.GatedDeltaNet.forward validates inputs by tags only, but its body combines them
    /// with shape-matching mul/sub and dynamicUpdateSlice: any axis order other than
    /// .{ .s, .h, .k } / .{ .h, .v, .k } panics.
    gdn_axis_order,
    /// Tensor.mean divides by the axis length as a constant of the tensor's dtype: an
    /// integer axis longer than the dtype's max panics (regressions/found/mean_int_len.zig).
    mean_int_len,
    /// Upstream XLA SPMD: a reduce with a bitwise body (and, or, xor) over a partitioned
    /// axis becomes an all-reduce that CPU can't compile ("Unsupported reduction
    /// computation", regressions/found/xla_spmd_or.zig). Partitioned axes also come from
    /// manualComputation, not only from sharded arguments.
    xla_spmd_bitwise_reduce,
    /// Compiler.repack packs a sub-byte argument or result by splitting its last axis in
    /// two: at Shape.MAX_RANK the split overflows and panics ("attempt to unwrap error:
    /// Overflow", regressions/found/subbyte_rank8.zig).
    subbyte_max_rank,
    /// Upstream XLA CPU: a padded reduce_window feeding another padded reduce_window
    /// (Tensor.cumulativeSum is one) gives wrong values, e.g. the cumsum of [0, 0, 1, 0, 0]
    /// comes out [1, 1, 1, 1, 1] (regressions/found/xla_rw_cumsum.zig).
    xla_reduce_window_chain,
    /// Upstream XLA CPU: chained float additions of constants are folded, so
    /// `x.subConstant(c1).addConstant(c2)` is rounded once, as x + (c2 - c1)
    /// (regressions/found/xla_const_fold.zig).
    xla_const_reassoc,
    /// Upstream XLA SPMD: a pad with interior padding and a negative edge on a partitioned
    /// axis aborts the partitioner (spmd_partitioner.cc CHECK
    /// get_first_valid_element_offset_on_dilated_shard, regressions/found/xla_spmd_pad.zig).
    xla_spmd_interior_pad,
    /// With the GSPMD partitioner (`--gspmd`), ops.partitionId inside manualComputation
    /// doesn't compile ("PartitionId instruction is not supported for SPMD partitioning",
    /// regressions/found/gspmd_part_id.zig).
    gspmd_partition_id,
    /// Upstream XLA GSPMD: a strided pool over a size 1 axis feeding a manualComputation
    /// aborts the partitioner (CHECK !target.IsReplicatedOrSingleDevice(),
    /// regressions/found/gspmd_pool_manual.zig).
    gspmd_pool_manual,
    /// Upstream XLA GSPMD on the two level mesh: manualComputation next to other results
    /// aborts the partitioner (hlo_sharding.cc CHECK !IsManualLeaf() && !IsUnknownLeaf(),
    /// regressions/found/gspmd_mask_manual.zig).
    gspmd_mesh2d_manual,
    /// nn.resizeLinear1d / resizeCubic1d compute sample coordinates in the image dtype:
    /// f16 and bf16 coordinates are coarser than a pixel on long axes, so the wrong
    /// pixels are read with weights outside [0, 1] (regressions/found/resize_coords.zig).
    resize_half_coords,
    /// Upstream XLA CPU: a convolution with negative padding and non default dimension
    /// numbers gives wrong values (regressions/found/xla_conv_neg.zig).
    xla_conv_neg_pad_layout,
    /// Upstream XLA CPU: an FFT in a sharded program may get a non default input layout
    /// and fail at run time (fft_thunk.cc RET_CHECK, regressions/found/xla_fft_layout.zig).
    xla_fft_layout,
    /// Upstream XLA CPU: convert(bool) times a negative constant gives +0 for false
    /// instead of -0 (regressions/found/xla_bool_mul.zig).
    xla_bool_mul_sign,
};

/// Whether node `id` is a convert from bool (KnownBug.xla_bool_mul_sign).
pub fn fromBool(prog: *const ir.Program, id: u32) bool {
    const n = prog.nodes.items[id];
    return n.op == .convert and n.inputs.len > 0 and prog.nodes.items[n.inputs.slice()[0]].shape.dtype() == .bool;
}

/// A reduce_window node with padding (KnownBug.xla_reduce_window_chain).
pub fn paddedWindow(node: ir.Node) bool {
    if (node.op != .reduce_window) return false;
    for (node.op.reduce_window.win.pad.slice()) |p| if (p[0] != 0 or p[1] != 0) return true;
    return false;
}

/// Top level proposal outcomes per family, for `zmlsmith stats`: a family rejected most
/// of the time explores little.
pub const stats = struct {
    pub var attempts: [ops.family_count]u64 = @splat(0);
    pub var commits: [ops.family_count]u64 = @splat(0);
    /// Rejected by commit: the reference value, avoided shapes or dtypes.
    pub var commit_rejects: [ops.family_count]u64 = @splat(0);
};

pub const Config = struct {
    /// Probability that a program runs on the sharded mesh.
    sharded_prob: f32 = 0.3,
    /// Known bugs the generator steers around.
    avoid: std.EnumSet(KnownBug) = .initFull(),
    min_nodes: u32 = 3,
    max_nodes: u32 = 20,
    max_inputs: u32 = 6,
    max_rank: u4 = 5,
    max_dim: i64 = 7,
    max_elems: usize = 1024,
    /// Probability that a fresh shape is fully tagged.
    tagged_prob: f32 = 0.8,
    /// Probability of a size-1 axis in fresh shapes, the broadcasting edge case.
    one_dim_prob: f32 = 0.2,
    /// Probability of a size-0 axis in fresh shapes.
    zero_dim_prob: f32 = 0.0,
    dtypes: []const DataType = &default_dtypes,
    /// Per-family weights, indexed like `ops.Family`. Zero disables a family.
    weights: [ops.family_count]u32 = ops.default_weights,

    pub const default_dtypes = [_]DataType{ .f32, .f32, .f32, .bf16, .f16, .f64, .i32, .i32, .i64, .i8, .i16, .u8, .u32, .u16, .u64, .bool };
    /// Picked by `convert` as a destination, and then moved around by layout ops.
    pub const storage_dtypes = [_]DataType{ .i4, .u4, .i2, .u2, .f4e2m1, .f8e3m4, .f8e4m3, .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .f8e8m0 };
};

pub const Class = enum {
    any,
    float,
    int,
    signed_int,
    unsigned_int,
    bool,
    int_or_bool,
    /// Integers and floats, not bool.
    numeric,
    /// Signed integers and floats.
    signed,
    /// Sub-byte integers, f8 and f4: storage only.
    storage,

    pub fn has(c: Class, dt: DataType) bool {
        if (Value.isStorage(dt)) return c == .storage;
        return switch (c) {
            .storage => false,
            .any => true,
            .float => Value.isFloat(dt),
            .int => Value.isInt(dt),
            .signed_int => Value.isInt(dt) and dt.isSignedInt(),
            .unsigned_int => Value.isInt(dt) and !dt.isSignedInt(),
            .bool => dt == .bool,
            .int_or_bool => Value.isInt(dt) or dt == .bool,
            .numeric => dt != .bool,
            .signed => Value.isFloat(dt) or (Value.isInt(dt) and dt.isSignedInt()),
        };
    }
};

pub const Filter = struct {
    class: Class = .any,
    dtype: ?DataType = null,
    min_rank: u4 = 0,
    max_rank: u4 = Shape.MAX_RANK,
    /// Only accept values whose error bound is zero.
    exact: bool = false,
    /// Require every axis to be tagged with distinct tags.
    tagged: bool = false,
    min_count: usize = 0,
    /// Exclude size-0 tensors.
    nonempty: bool = true,
    /// Accept values holding infinities on purpose.
    allow_inf: bool = false,
    /// Also accept storage-only dtypes (sub-byte, f8, f4), for data movement ops.
    allow_storage: bool = false,
};

pub const Proposal = struct {
    op: ops.Op,
    inputs: []const u32 = &.{},
    shape: Shape,
};

pub const Gen = struct {
    allocator: std.mem.Allocator,
    rng: std.Random,
    cfg: *const Config,
    prog: ir.Program = .{},
    values: std.ArrayList(Value) = .empty,
    n_inputs: u32 = 0,
    /// The region new nodes go to, 0 at top level. Operands are only picked in it.
    scope: u32 = 0,
    /// Values bound to the `param` nodes of the region being generated.
    bindings: []const Value = &.{},
    /// Control flow nesting depth.
    depth: u8 = 0,
    /// The family being proposed handles size-0 tensors (`pub const empty_ok`).
    empty_ok: bool = false,

    /// A random family according to the configured weights.
    pub fn pickFamily(g: *Gen) ops.Family {
        var total: u32 = 0;
        for (g.cfg.weights) |w| total += w;
        var r = g.rng.uintLessThan(u32, total);
        for (g.cfg.weights, 0..) |w, i| {
            if (r < w) return @enumFromInt(i);
            r -= w;
        }
        unreachable;
    }

    /// Drops the nodes added after `len`, when a control flow op is rejected.
    pub fn rollback(g: *Gen, len: usize) void {
        g.prog.nodes.shrinkRetainingCapacity(len);
        g.values.shrinkRetainingCapacity(len);
        g.n_inputs = 0;
        for (g.prog.nodes.items) |n| if (ops.isArgument(n.op)) {
            g.n_inputs += 1;
        };
    }

    /// Rejects the current proposal if it would hit a known, still avoided, bug.
    pub fn avoid(g: *Gen, bug: KnownBug, hits: bool) Error!void {
        if (hits and g.cfg.avoid.contains(bug)) return error.Reject;
    }

    pub fn chance(g: *Gen, p: f32) bool {
        return g.rng.float(f32) < p;
    }

    pub fn intIn(g: *Gen, comptime T: type, lo: T, hi: T) T {
        return g.rng.intRangeAtMost(T, lo, hi);
    }

    pub fn choose(g: *Gen, comptime T: type, items: []const T) T {
        return items[g.rng.uintLessThan(usize, items.len)];
    }

    pub fn shapeOf(g: *const Gen, id: u32) Shape {
        return g.prog.nodes.items[id].shape;
    }

    pub fn valueOf(g: *const Gen, id: u32) Value {
        return g.values.items[id];
    }

    pub fn randDtype(g: *Gen, class: Class) ?DataType {
        var cands: [32]DataType = undefined;
        var n: usize = 0;
        for (g.cfg.dtypes) |dt| {
            if (class.has(dt) and n < cands.len) {
                cands[n] = dt;
                n += 1;
            }
        }
        if (n == 0) return null;
        return cands[g.rng.uintLessThan(usize, n)];
    }

    pub fn randDim(g: *Gen) i64 {
        if (g.empty_ok and g.chance(g.cfg.zero_dim_prob)) return 0;
        if (g.chance(g.cfg.one_dim_prob)) return 1;
        // Large presets favor sizes around vector and tile widths, where backends switch
        // between main loops and remainder code.
        if (g.cfg.max_dim >= 16 and g.chance(0.3)) {
            const edges = [_]i64{ 8, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 127, 128, 129 };
            var n: usize = 0;
            while (n < edges.len and edges[n] <= g.cfg.max_dim) n += 1;
            return edges[g.rng.uintLessThan(usize, n)];
        }
        return g.intIn(i64, 2, g.cfg.max_dim);
    }

    /// Distinct random tags.
    pub fn randTags(g: *Gen, out: []Shape.Tag) void {
        var pool = ir.tag_pool;
        g.rng.shuffle(Shape.Tag, &pool);
        @memcpy(out, pool[0..out.len]);
    }

    /// A tag not used by `sh`.
    pub fn freshTag(g: *Gen, sh: Shape) Shape.Tag {
        while (true) {
            const t = g.choose(Shape.Tag, &ir.tag_pool);
            if (sh.hasTag(t) == null) return t;
        }
    }

    pub const ShapeOpts = struct {
        min_rank: u4 = 0,
        max_rank: ?u4 = null,
        tagged: ?bool = null,
    };

    /// Size-0 tensors may flow into the current family.
    pub fn edge(g: *const Gen) bool {
        return g.cfg.zero_dim_prob > 0 and g.empty_ok;
    }

    pub fn randShape(g: *Gen, dt: DataType, opts: ShapeOpts) Shape {
        const max_rank = @min(opts.max_rank orelse g.cfg.max_rank, g.cfg.max_rank);
        const min_rank = @min(opts.min_rank, max_rank);
        // Bias toward ranks 1 to 3.
        var rank: u4 = g.intIn(u4, min_rank, max_rank);
        if (rank > 3 and g.chance(0.5)) rank = @max(min_rank, g.intIn(u4, min_rank, 3));
        var dims: [Shape.MAX_RANK]i64 = undefined;
        for (dims[0..rank]) |*d| d.* = g.randDim();
        // Keep tensors small enough for the interpreter.
        while (true) {
            var c: usize = 1;
            for (dims[0..rank]) |d| c *= @intCast(@max(d, 1));
            if (c <= g.cfg.max_elems) break;
            var biggest: usize = 0;
            for (dims[0..rank], 0..) |d, i| if (d > dims[biggest]) {
                biggest = i;
            };
            dims[biggest] = @divFloor(dims[biggest], 2);
        }
        var tags: [Shape.MAX_RANK]Shape.Tag = @splat(Shape.TagUnknown);
        const tagged = opts.tagged orelse g.chance(g.cfg.tagged_prob);
        if (tagged) {
            g.randTags(tags[0..rank]);
        } else if (g.chance(0.15) and rank > 1) {
            // Partially tagged.
            g.randTags(tags[0..rank]);
            for (tags[0..rank]) |*t| if (g.chance(0.5)) {
                t.* = Shape.TagUnknown;
            };
        }
        return ir.makeShape(dt, dims[0..rank], tags[0..rank]);
    }

    fn matches(g: *const Gen, id: u32, f: Filter) bool {
        if (g.prog.nodes.items[id].scope != g.scope) return false;
        const sh = g.shapeOf(id);
        if (f.dtype) |dt| if (sh.dtype() != dt) return false;
        if (!f.class.has(sh.dtype()) and !(f.allow_storage and Value.isStorage(sh.dtype()))) return false;
        if (sh.rank() < f.min_rank or sh.rank() > f.max_rank) return false;
        if (f.nonempty and sh.count() == 0 and !g.edge()) return false;
        if (sh.count() < f.min_count) return false;
        if (f.exact and g.values.items[id].err != 0) return false;
        if (f.tagged and !fullyTagged(sh)) return false;
        if (g.values.items[id].allow_inf and !f.allow_inf) return false;
        return true;
    }

    /// Picks an existing node matching the filter, favoring recent ones so that
    /// programs grow deep instead of wide.
    pub fn pick(g: *Gen, f: Filter) ?u32 {
        const n = g.prog.nodes.items.len;
        var best: ?u32 = null;
        var best_score: f32 = -1;
        for (0..n) |i| {
            const id: u32 = @intCast(i);
            if (!g.matches(id, f)) continue;
            const recency = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(n));
            const score = g.rng.float(f32) * (0.3 + recency);
            if (score > best_score) {
                best_score = score;
                best = id;
            }
        }
        return best;
    }

    /// Picks a node with exactly these dims and dtype, ignoring tags unless `same_tags`.
    pub fn pickLike(g: *Gen, sh: Shape, same_tags: bool) ?u32 {
        const n = g.prog.nodes.items.len;
        var best: ?u32 = null;
        var best_score: f32 = -1;
        for (0..n) |i| {
            if (g.values.items[i].allow_inf) continue;
            if (g.prog.nodes.items[i].scope != g.scope) continue;
            const other = g.prog.nodes.items[i].shape;
            if (!other.eql(sh)) continue;
            if (same_tags and !std.mem.eql(Shape.Tag, other.tags(), sh.tags())) continue;
            const score = g.rng.float(f32);
            if (score > best_score) {
                best_score = score;
                best = @intCast(i);
            }
        }
        return best;
    }

    /// Creates a new program argument with random data suited to its dtype.
    pub fn freshInput(g: *Gen, sh: Shape) Error!u32 {
        if (g.scope != 0) {
            // Regions can't have arguments: embed the data instead.
            const v = try Value.alloc(g.allocator, sh);
            fillInput(v, .{ .seed = g.rng.int(u64), .dist = .int_range, .lo = if (sh.dtype() == .bool or (Value.isInt(sh.dtype()) and !sh.dtype().isSignedInt())) 0 else -3, .hi = 3 });
            return g.commit(.{ .op = .{ .constant = .{ .data = v.bytes } }, .shape = sh });
        }
        if (g.n_inputs >= g.cfg.max_inputs) return error.Reject;
        const dt = sh.dtype();
        var attrs: ops.Input = .{ .seed = g.rng.int(u64), .dist = .int_range, .lo = -4, .hi = 4 };
        if (dt == .f8e8m0) {
            // Powers of two only, no zero and no sign.
            attrs.lo = 1;
            attrs.hi = 2;
        } else if (dt == .bool) {
            attrs.lo = 0;
            attrs.hi = 1;
        } else if (Value.isInt(dt)) {
            const r = g.rng.uintLessThan(u8, 10);
            if (r < 2) {
                attrs.dist = .bits;
            } else if (r < 4) {
                attrs.lo = 0;
                attrs.hi = 9;
            } else if (!dt.isSignedInt()) {
                attrs.lo = 0;
                attrs.hi = 12;
            }
        } else {
            const r = g.rng.uintLessThan(u8, 10);
            if (r < 4) {
                attrs.dist = .uniform;
                attrs.lo = -2;
                attrs.hi = 2;
            } else if (r < 6) {
                attrs.dist = .dyadic;
            } else if (r < 7) {
                attrs.lo = 1;
                attrs.hi = 6;
            } else if (g.edge()) {
                attrs.dist = .signed_zero;
            }
        }
        return g.commit(.{ .op = .{ .input = attrs }, .shape = sh });
    }

    /// A fresh input with values in [lo, hi], for ops with a restricted domain.
    pub fn freshInputRange(g: *Gen, sh: Shape, lo: i64, hi: i64) Error!u32 {
        if (g.scope != 0) {
            const v = try Value.alloc(g.allocator, sh);
            fillInput(v, .{ .seed = g.rng.int(u64), .dist = .int_range, .lo = lo, .hi = hi });
            return g.commit(.{ .op = .{ .constant = .{ .data = v.bytes } }, .shape = sh });
        }
        if (g.n_inputs >= g.cfg.max_inputs) return error.Reject;
        const dist: ops.impls.input.Dist = if (Value.isFloat(sh.dtype()) and g.chance(0.5)) .uniform else .int_range;
        return g.commit(.{ .op = .{ .input = .{ .seed = g.rng.int(u64), .dist = dist, .lo = lo, .hi = hi } }, .shape = sh });
    }

    /// Creates a program argument with explicit data, typically indices.
    pub fn literalInput(g: *Gen, v: Value) Error!u32 {
        if (g.scope != 0) return g.commit(.{ .op = .{ .constant = .{ .data = v.bytes } }, .shape = v.shape });
        if (g.n_inputs >= g.cfg.max_inputs) return error.Reject;
        return g.commit(.{ .op = .{ .literal = .{ .data = v.bytes } }, .shape = v.shape });
    }

    /// An existing node matching the filter, or else a fresh input of a random shape.
    pub fn operand(g: *Gen, f: Filter) Error!u32 {
        if (!g.chance(0.15)) {
            if (g.pick(f)) |id| return id;
        }
        const dt = f.dtype orelse g.randDtype(f.class) orelse return error.Reject;
        const sh = g.randShape(dt, .{
            .min_rank = f.min_rank,
            .max_rank = @min(f.max_rank, g.cfg.max_rank),
            .tagged = if (f.tagged) true else null,
        });
        if (f.nonempty and sh.count() == 0 and !g.edge()) return error.Reject;
        if (sh.count() < f.min_count) return error.Reject;
        return g.freshInput(sh);
    }

    /// An existing node with the same dims and dtype as `sh`, or else a fresh input.
    /// The tags of the result follow `sh` when a fresh input is created.
    pub fn operandLike(g: *Gen, sh: Shape, same_tags: bool) Error!u32 {
        if (!g.chance(0.2)) {
            if (g.pickLike(sh, same_tags)) |id| return id;
        }
        return g.freshInput(sh);
    }

    /// Evaluates and appends a node. Inputs and literals are evaluated here too.
    pub fn commit(g: *Gen, prop: Proposal) Error!u32 {
        if (g.cfg.avoid.contains(.subbyte_last_dim) and badPacking(prop.shape)) return error.Reject;
        if (g.cfg.avoid.contains(.subbyte_empty) and prop.shape.dtype().bitSizeOf() < 8 and prop.shape.count() == 0) return error.Reject;
        if (g.cfg.avoid.contains(.subbyte_max_rank) and prop.shape.dtype().bitSizeOf() < 8 and prop.shape.rank() == Shape.MAX_RANK) return error.Reject;
        var node: ir.Node = .{ .op = prop.op, .shape = prop.shape, .scope = g.scope };
        // Regions only read their own nodes; a family picking across scopes is a bug.
        for (prop.inputs) |in| if (g.prog.nodes.items[in].scope != g.scope) return error.Reject;
        for (prop.inputs) |in| node.inputs.append(in);
        var ins: [ir.max_operands]Value = undefined;
        for (prop.inputs, 0..) |in, i| ins[i] = g.values.items[in];
        var e: ops.Eval = .{ .allocator = g.allocator, .strict = true, .prog = &g.prog, .bindings = g.bindings };
        const v = try ops.eval(&e, &node, ins[0..prop.inputs.len]);
        std.debug.assert(v.shape.eql(node.shape));
        if (!acceptable(v)) return error.Reject;
        const id = try g.prog.add(g.allocator, node);
        try g.values.append(g.allocator, v);
        if (ops.isArgument(prop.op)) g.n_inputs += 1;
        return id;
    }
};

/// Shapes Shape.packedShape can't pack (KnownBug.subbyte_last_dim).
fn badPacking(sh: Shape) bool {
    const bits = sh.dtype().bitSizeOf();
    if (bits >= 8) return false;
    if (sh.rank() == 0) return true;
    return @mod(sh.dim(-1), @divExact(8, bits)) != 0;
}

pub fn fullyTagged(sh: Shape) bool {
    for (sh.tags()) |t| if (t == Shape.TagUnknown) return false;
    return true;
}

/// Values the oracle can compare meaningfully.
fn acceptable(v: Value) bool {
    const dt = v.dtype();
    if (!Value.isFloat(dt)) return true;
    if (v.allow_inf) return true;
    if (!v.allFinite()) return false;
    // Storage floats: only the range matters, their conversions handle subnormals.
    if (Value.isStorage(dt)) return v.maxAbs() <= Value.floatLimit(dt);
    const m = v.maxAbs();
    if (m > Value.floatLimit(dt)) return false;
    // Devices may flush subnormals to zero.
    const min_normal: f64 = switch (dt) {
        .f16 => std.math.floatMin(f16),
        .bf16, .f32 => std.math.floatMin(f32),
        else => std.math.floatMin(f64),
    };
    for (0..v.count()) |i| {
        const x = @abs(v.getF(i));
        if (x != 0 and x < min_normal) return false;
    }
    // Too much accumulated imprecision makes any comparison meaningless.
    if (v.err > 0.05 * @max(1.0, m)) return false;
    return true;
}

/// Input space presets, picked from the seed so that every worker and the supervisor
/// agree on the program a seed produces.
pub const Preset = enum {
    default,
    /// High ranks, size-0 and size-1 axes, signed zeros.
    edge,
    /// Large dims to reach the size dependent paths of XLA (fusions, tiling, gemm).
    big,
    /// Long programs, for interactions between many ops and deeper error accumulation.
    long,

    pub fn of(seed: u64) Preset {
        return switch (seed % 10) {
            3 => .edge,
            5 => .long,
            7 => .big,
            else => .default,
        };
    }
};

/// The known bug a seed generates without avoiding, if any: 15% of seeds lift one, so
/// the input space behind each avoidance keeps being exercised. The supervisor buckets
/// their failures apart.
pub fn liftedBug(seed: u64) ?KnownBug {
    var prng: std.Random.DefaultPrng = .init(seed ^ 0x6b6e_6f77_6e00);
    const r = prng.random();
    if (r.float(f32) >= 0.15) return null;
    return r.enumValue(KnownBug);
}

/// Generates a program from a seed.
pub fn generate(allocator: std.mem.Allocator, seed: u64, base_cfg: *const Config) !struct { ir.Program, []Value } {
    var preset_cfg = base_cfg.*;
    if (liftedBug(seed)) |bug| preset_cfg.avoid.remove(bug);
    switch (Preset.of(seed)) {
        .default => {},
        .edge => {
            preset_cfg.max_rank = Shape.MAX_RANK;
            preset_cfg.zero_dim_prob = 0.06;
            preset_cfg.one_dim_prob = 0.4;
            preset_cfg.max_dim = 4;
        },
        .big => {
            preset_cfg.max_dim = 129;
            preset_cfg.max_elems = 32 * 1024;
            preset_cfg.max_nodes = 10;
            preset_cfg.max_rank = 4;
        },
        .long => {
            preset_cfg.min_nodes = 30;
            preset_cfg.max_nodes = 80;
            preset_cfg.max_inputs = 10;
            preset_cfg.max_elems = 256;
        },
    }
    // Swarm testing: half the programs drop a random subset of families, so the remaining
    // ones get longer chains and combinations a uniform mix rarely builds.
    var swarm_prng: std.Random.DefaultPrng = .init(seed ^ 0x5eed_5a4d);
    const swarm = swarm_prng.random();
    if (swarm.boolean()) {
        var kept = preset_cfg.weights;
        var any = false;
        for (&kept) |*w| {
            if (swarm.float(f32) < 0.4) w.* = 0;
            any = any or w.* > 0;
        }
        if (any) preset_cfg.weights = kept;
    }
    const cfg = &preset_cfg;
    var prng: std.Random.DefaultPrng = .init(seed);
    var g: Gen = .{ .allocator = allocator, .rng = prng.random(), .cfg = cfg };
    g.prog.seed = seed;
    g.prog.sharded = g.chance(cfg.sharded_prob);
    // Sharded arguments hit the reduced axis bug as soon as any op shrinks them.
    g.prog.shard_args = g.prog.sharded and !cfg.avoid.contains(.sharded_reduced_axis);

    var total_weight: u32 = 0;
    for (cfg.weights) |w| total_weight += w;
    std.debug.assert(total_weight > 0);

    const target = g.intIn(u32, cfg.min_nodes, cfg.max_nodes);
    var attempts: u32 = 0;
    while (g.prog.nodes.items.len < target and attempts < target * 40) : (attempts += 1) {
        const fam = g.pickFamily();
        stats.attempts[@intFromEnum(fam)] += 1;
        const before = g.prog.nodes.items.len;

        const prop = ops.propose(fam, &g) catch |err| switch (err) {
            error.Reject => continue,
            error.OutOfMemory => return error.OutOfMemory,
        } orelse {
            // Region families commit their nodes themselves.
            if (g.prog.nodes.items.len > before) stats.commits[@intFromEnum(fam)] += 1;
            continue;
        };
        _ = g.commit(prop) catch |err| switch (err) {
            error.Reject => {
                stats.commit_rejects[@intFromEnum(fam)] += 1;
                continue;
            },
            error.OutOfMemory => return error.OutOfMemory,
        };
        stats.commits[@intFromEnum(fam)] += 1;
    }

    if (g.prog.nodes.items.len == 0) {
        const dt = g.randDtype(.any) orelse .f32;
        _ = try g.freshInput(g.randShape(dt, .{}));
    }
    try pickOutputs(&g);
    dedupeDonated(&g);
    return .{ g.prog, g.values.items };
}

/// The node a value aliases on CPU: placement ops and reuseBuffer return their input.
fn aliasRoot(nodes: []const ir.Node, id: u32) u32 {
    var cur = id;
    while (true) {
        const n = nodes[cur];
        if (n.op == .memory) {
            cur = n.inputs.slice()[0];
            continue;
        }
        // Ops that may return their input as is (repeat by 1, renames, same-shape
        // broadcast or reshape): conservatively, any single input op keeping the dims.
        if (n.inputs.len == 1 and n.shape.eqlDims(nodes[n.inputs.slice()[0]].shape) and n.shape.dtype() == nodes[n.inputs.slice()[0]].shape.dtype()) {
            cur = n.inputs.slice()[0];
            continue;
        }
        return cur;
    }
}

/// Whether the alias class of `id` is donated by a reuseBuffer.
fn isDonated(nodes: []const ir.Node, id: u32) bool {
    const root = aliasRoot(nodes, id);
    for (nodes) |n| if (n.op == .memory and n.op.memory.kind == .reuse_buffer and aliasRoot(nodes, n.inputs.slice()[0]) == root) return true;
    return false;
}

/// Drops outputs that alias an already returned donated tensor (KnownBug.reuse_duplicate_output).
fn dedupeDonated(g: *Gen) void {
    if (!g.cfg.avoid.contains(.reuse_duplicate_output)) return;
    const nodes = g.prog.nodes.items;
    var outs = &g.prog.outputs;
    // Tag-only ops (rename, partial tags) return the donated buffer under another node:
    // with any donation, an exact duplicate output may be one too.
    var any_reuse = false;
    for (nodes) |n| any_reuse = any_reuse or (n.op == .memory and n.op.memory.kind == .reuse_buffer);
    var i: usize = 0;
    while (i < outs.items.len) {
        const o = outs.items[i];
        var dup = any_reuse and std.mem.indexOfScalar(u32, outs.items[0..i], o) != null;
        if (!dup and isDonated(nodes, o)) {
            for (outs.items[0..i]) |p| if (aliasRoot(nodes, p) == aliasRoot(nodes, o)) {
                dup = true;
            };
        }
        if (dup) _ = outs.orderedRemove(i) else i += 1;
    }
}

/// Outputs are the nodes nobody consumes, so every computation is observable, plus
/// occasionally an argument or a duplicate, which exercise aliasing in the runtime.
fn pickOutputs(g: *Gen) !void {
    const nodes = g.prog.nodes.items;
    const used = try g.allocator.alloc(bool, nodes.len);
    @memset(used, false);
    for (nodes) |n| for (n.inputs.slice()) |in| {
        used[in] = true;
    };
    var sinks: std.ArrayList(u32) = .empty;
    var i: usize = nodes.len;
    while (i > 0) {
        i -= 1;
        if (!used[i] and !ops.isArgument(nodes[i].op) and nodes[i].scope == 0) try sinks.append(g.allocator, @intCast(i));
    }
    // Every sink must be observed, otherwise its computation is dead code.
    for (sinks.items) |s| try g.prog.outputs.append(g.allocator, s);
    if (g.prog.outputs.items.len == 0) try g.prog.outputs.append(g.allocator, @intCast(nodes.len - 1));
    if (g.chance(0.1)) {
        const o = g.prog.outputs.items[0];
        const donated = nodes[o].op == .memory and nodes[o].op.memory.kind == .reuse_buffer;
        if (!(donated and g.cfg.avoid.contains(.reuse_duplicate_output))) try g.prog.outputs.append(g.allocator, o);
    }
    if (g.chance(0.1)) {
        for (nodes, 0..) |n, j| if (ops.isArgument(n.op)) {
            try g.prog.outputs.append(g.allocator, @intCast(j));
            break;
        };
    }
    if (g.chance(0.1) and nodes.len > 1) {
        const o = g.rng.uintLessThan(u32, @intCast(nodes.len));
        const donated = nodes[o].op == .memory and nodes[o].op.memory.kind == .reuse_buffer;
        if (nodes[o].scope == 0 and !(donated and g.cfg.avoid.contains(.reuse_duplicate_output))) try g.prog.outputs.append(g.allocator, o);
    }
}

/// Fills an input value from its seeded distribution.
pub fn fillInput(v: Value, attrs: ops.Input) void {
    var prng: std.Random.DefaultPrng = .init(attrs.seed);
    const rng = prng.random();
    const dt = v.dtype();
    for (0..v.count()) |i| {
        switch (attrs.dist) {
            .bits => {
                if (Value.isFloat(dt)) {
                    v.setF(i, @floatFromInt(rng.intRangeAtMost(i64, attrs.lo, attrs.hi)));
                } else {
                    v.setI(i, @bitCast(@as(u128, rng.int(u64)) | (@as(u128, rng.int(u64)) << 64)));
                }
            },
            .int_range => {
                const x = rng.intRangeAtMost(i64, attrs.lo, attrs.hi);
                if (Value.isFloat(dt)) v.setF(i, @floatFromInt(x)) else v.setI(i, x);
            },
            .uniform => {
                const lo: f64 = @floatFromInt(attrs.lo);
                const hi: f64 = @floatFromInt(attrs.hi);
                const x = lo + (hi - lo) * rng.float(f64);
                if (Value.isFloat(dt)) v.setF(i, x) else v.setI(i, @intFromFloat(@round(x)));
            },
            .dyadic => {
                const x = rng.intRangeAtMost(i64, attrs.lo * 4, attrs.hi * 4);
                if (Value.isFloat(dt)) v.setF(i, @as(f64, @floatFromInt(x)) / 4) else v.setI(i, x);
            },
            .signed_zero => {
                // Mostly zeros of both signs, the edge case of min/max/sign/compare.
                const x: f64 = switch (rng.uintLessThan(u8, 4)) {
                    0 => 0.0,
                    1 => -0.0,
                    else => @floatFromInt(rng.intRangeAtMost(i64, -2, 2)),
                };
                if (Value.isFloat(dt)) v.setF(i, x) else v.setI(i, @intFromFloat(x));
            },
        }
    }
}
