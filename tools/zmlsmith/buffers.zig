//! zml.Buffer API checks on the device: host <-> device round trips for random shapes,
//! dtypes and shardings, views, scalars, shard bookkeeping, device pointers, uninitialized
//! buffers and memory kinds.

const std = @import("std");

const zml = @import("zml");
const Shape = zml.Shape;
const DataType = zml.DataType;

const ir = @import("ir.zig");
const runner = @import("runner.zig");

const dtypes = [_]DataType{ .f32, .bf16, .f16, .f64, .i8, .i32, .i64, .u8, .u16, .u64, .bool, .i4, .u4, .f8e4m3fn, .f8e5m2 };

pub fn run(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, seed: u64, count: usize) !usize {
    var prng: std.Random.DefaultPrng = .init(seed);
    const rng = prng.random();
    var failures: usize = 0;
    var seen: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (0..count) |_| {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const a = arena.allocator();

        const dt = dtypes[rng.uintLessThan(usize, dtypes.len)];
        const rank = rng.uintLessThan(usize, 4) + 1;
        var dims: [4]i64 = undefined;
        var tags: [4]Shape.Tag = undefined;
        var pool = ir.tag_pool;
        rng.shuffle(Shape.Tag, &pool);
        for (0..rank) |i| {
            dims[i] = rng.intRangeAtMost(i64, 1, 5);
            tags[i] = pool[i];
        }
        // Sharded on the mesh: a random axis on `data` and, on the two level mesh, maybe
        // another on `model`, made divisible by the device count.
        // Sharded sub-byte buffers crash in toSliceAlloc (KnownBug.sharded_subbyte).
        const sharded = runner.mesh != null and rng.boolean() and dt.bitSizeOf() >= 8;
        const data_ax = rng.uintLessThan(usize, rank);
        const model_ax: ?usize = if (sharded and runner.mesh2d and rank > 1 and rng.boolean()) (data_ax + 1 + rng.uintLessThan(usize, rank - 1)) % rank else null;
        if (sharded) dims[data_ax] = 4 * rng.intRangeAtMost(i64, 1, 2);
        if (model_ax) |m| dims[m] = 4 * rng.intRangeAtMost(i64, 1, 2);
        // Sub-byte dtypes need a last dim the packing divides.
        if (dt.bitSizeOf() < 8) dims[rank - 1] = 2 * rng.intRangeAtMost(i64, 1, 3);
        var sh = ir.makeShape(dt, dims[0..rank], tags[0..rank]);
        if (sharded) {
            sh = sh.withDefaultPartitioning();
            sh._partitioning.set(data_ax, .{ .axis = Shape.toTag(.data) });
            if (model_ax) |m| sh._partitioning.set(m, .{ .axis = Shape.toTag(.model) });
        }
        const sharding: zml.Sharding = if (sharded) runner.mesh.? else .replicated;

        const data = try a.alloc(u8, sh.byteSize());
        rng.bytes(data);
        if (dt == .bool) for (data) |*b| {
            b.* &= 1;
        };

        const check = struct {
            fn f(s: *std.StringArrayHashMapUnmanaged(void), al: std.mem.Allocator, n: *usize, name: []const u8, ok: bool, shape: Shape) !void {
                if (ok) return;
                n.* += 1;
                const gop = try s.getOrPut(al, name);
                if (!gop.found_existing) std.debug.print("RESULT buffers kind=buffers sig=buffer.{s}\n    {f}\n", .{ name, shape });
            }
        }.f;

        // fromBytes -> toSliceAlloc.
        var b = try zml.Buffer.fromBytes(io, platform, sh, sharding, data);
        defer b.deinit();
        const back = try b.toSliceAlloc(a, io);
        try check(&seen, allocator, &failures, "fromBytes_toSliceAlloc", std.mem.eql(u8, back.constData(), data), sh);

        // Shard bookkeeping.
        try check(&seen, allocator, &failures, "numShards", b.numShards() == platform.devices.len or !sharded, sh);
        try b.await(io);
        var it = b.shards();
        var n_shards: usize = 0;
        var remaining_ok = it.remaining() == b.numShards();
        while (it.next()) |shard| : (n_shards += 1) {
            remaining_ok = remaining_ok and it.remaining() == b.numShards() - n_shards - 1;
            // Both device pointer accessors agree on the shard order.
            if (sh.byteSize() > 0) try check(&seen, allocator, &failures, "devicePtr_opaqueDevicePtr", shard.devicePtr() == b.opaqueDevicePtr(n_shards), sh);
        }
        try check(&seen, allocator, &failures, "shards_iterator", n_shards == b.numShards(), sh);
        try check(&seen, allocator, &failures, "ShardIterator_remaining", remaining_ok and it.remaining() == 0, sh);

        // uninitialized: same metadata as a buffer holding data.
        if (zml.Buffer.uninitialized(io, platform, sh, sharding, .{})) |u_| {
            var u = u_;
            defer u.deinit();
            try u.await(io);
            try check(&seen, allocator, &failures, "uninitialized_metadata", u.shape().eqlWithTags(sh) and u.numShards() == b.numShards(), sh);
        } else |err| {
            try check(&seen, allocator, &failures, try std.fmt.allocPrint(a, "uninitialized_error_{t}", .{err}), false, sh);
        }

        // fromSlice -> toSlice into an existing slice.
        const src: zml.Slice = .init(sh, data);
        var b2 = try zml.Buffer.fromSlice(io, platform, src, sharding);
        defer b2.deinit();
        const dst = try zml.Slice.alloc(a, sh);
        // toSlice can't read shards that aren't contiguous in the destination (an axis
        // other than the first is sharded) and says so; toSliceAlloc copies them.
        const read_ok = if (b2.toSlice(io, dst)) true else |err| blk: {
            if (err != error.NonContiguousShardRead or !(sharded and (data_ax != 0 or model_ax != null)))
                try check(&seen, allocator, &failures, try std.fmt.allocPrint(a, "toSlice_error_{t}", .{err}), false, sh);
            break :blk false;
        };
        if (read_ok) try check(&seen, allocator, &failures, "fromSlice_toSlice", std.mem.eql(u8, dst.constData(), data), sh);

        // A view of the device memory reads the same data.
        if (!sharded) {
            if (b.createView()) |view_| {
                var view = view_;
                defer view.deinit();
                const vb = try view.toSliceAlloc(a, io);
                try check(&seen, allocator, &failures, "createView", std.mem.eql(u8, vb.constData(), data), sh);
            } else |err| {
                try check(&seen, allocator, &failures, try std.fmt.allocPrint(a, "createView_error_{t}", .{err}), false, sh);
            }
        }

        // Memory kinds, from bytes and from a slice.
        const kinds = [_]zml.Memory.Kind{ .default, .device, .host_pinned, .host_unpinned };
        const kind = kinds[rng.uintLessThan(usize, kinds.len)];
        const from_slice = rng.boolean();
        const name = if (from_slice) "fromSliceOpts" else "fromBytesOpts";
        if (if (from_slice) zml.Buffer.fromSliceOpts(io, platform, src, sharding, .{ .memory = kind }) else zml.Buffer.fromBytesOpts(io, platform, sh, sharding, data, .{ .memory = kind })) |bm_| {
            var bm = bm_;
            defer bm.deinit();
            const mb = try bm.toSliceAlloc(a, io);
            try check(&seen, allocator, &failures, try std.fmt.allocPrint(a, "{s}_{t}", .{ name, kind }), std.mem.eql(u8, mb.constData(), data), sh);
        } else |err| {
            try check(&seen, allocator, &failures, try std.fmt.allocPrint(a, "{s}_{t}_error_{t}", .{ name, kind, err }), false, sh);
        }
    }

    // Releasing the buffers of a struct of tensors.
    {
        const Pair = struct { x: zml.Tensor, y: zml.Tensor };
        const sh: Shape = .init(.{ .a = 3 }, .f32);
        var pair: zml.Bufferized(Pair) = .{ .x = try .uninitialized(io, platform, sh, .replicated, .{}), .y = try .uninitialized(io, platform, sh, .replicated, .{}) };
        zml.Buffer.deinitAll(Pair, &pair);
        var pair2: zml.Bufferized(Pair) = .{ .x = try .uninitialized(io, platform, sh, .replicated, .{}), .y = try .uninitialized(io, platform, sh, .replicated, .{}) };
        zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Pair, &pair2);
    }

    // Scalars and getValue.
    {
        var s = try zml.Buffer.scalar(io, platform, 42, .i32);
        defer s.deinit();
        const v = try s.getValue(i32, io);
        if (v != 42) {
            failures += 1;
            std.debug.print("RESULT buffers kind=buffers sig=buffer.scalar_getValue\n    got {d}\n", .{v});
        }
        var sf = try zml.Buffer.scalar(io, platform, 1.5, .bf16);
        defer sf.deinit();
        const vf = try sf.getValue(zml.floats.BFloat16, io);
        if (vf.toF32() != 1.5) {
            failures += 1;
            std.debug.print("RESULT buffers kind=buffers sig=buffer.scalar_bf16\n    got {d}\n", .{vf.toF32()});
        }
    }
    std.debug.print("buffers: {d} cases, {d} failures, {d} distinct\n", .{ count, failures, seen.count() });
    return failures;
}
