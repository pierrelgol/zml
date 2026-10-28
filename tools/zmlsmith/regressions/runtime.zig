//! Runtime properties of executables, on random data: inputs that aren't donated are
//! left unchanged, donation computes the right result, repeated, unawaited and concurrent
//! calls don't share state, concurrent compiles work, and FnExe runners with a baked
//! prefix agree with the raw executable.

const std = @import("std");
const zml = @import("zml");

const n = 64;
const shape: zml.Shape = .init(.{ .a = n }, .f32);

fn randomData(seed: u64) [n]f32 {
    var prng: std.Random.DefaultPrng = .init(seed);
    var out: [n]f32 = undefined;
    // Small integers: every result below is exact.
    for (&out) |*x| x.* = @floatFromInt(prng.random().intRangeAtMost(i32, -50, 50));
    return out;
}

fn upload(io: std.Io, platform: *const zml.Platform, data: *const [n]f32) !zml.Buffer {
    return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(data));
}

fn download(allocator: std.mem.Allocator, io: std.Io, b: zml.Buffer) ![n]f32 {
    const s = try b.toSliceAlloc(allocator, io);
    defer s.free(allocator);
    var out: [n]f32 = undefined;
    @memcpy(&out, s.constItems(f32));
    return out;
}

const Fns = struct {
    fn axpy(x: zml.Tensor, y: zml.Tensor) zml.Tensor {
        return x.scale(2).add(y);
    }
    fn incDonated(x: zml.Tensor) zml.Tensor {
        return x.addConstant(1).reuseBuffer(x);
    }
    fn neg(x: zml.Tensor) zml.Tensor {
        return x.negate();
    }
};

fn axpyRef(x: [n]f32, y: [n]f32) [n]f32 {
    var out: [n]f32 = undefined;
    for (&out, x, y) |*o, a, b| o.* = 2 * a + b;
    return out;
}

test "runtime: inputs that aren't donated are unchanged" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const x: zml.Tensor = .fromShape(shape);
    var exe = try zml.module.compile(allocator, io, Fns.axpy, .{ x, zml.Tensor.fromShape(shape) }, platform, .{});
    defer exe.deinit();
    for (0..8) |seed| {
        const xd = randomData(seed);
        const yd = randomData(seed + 100);
        var bx = try upload(io, platform, &xd);
        defer bx.deinit();
        var by = try upload(io, platform, &yd);
        defer by.deinit();
        var res = try zml.testing.autoCall(allocator, io, &exe, Fns.axpy, .{ bx, by });
        defer res.deinit();
        try std.testing.expectEqual(axpyRef(xd, yd), try download(allocator, io, res));
        try std.testing.expectEqual(xd, try download(allocator, io, bx));
        try std.testing.expectEqual(yd, try download(allocator, io, by));
    }
}

test "runtime: a donated input gives the right result" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var exe = try zml.module.compile(allocator, io, Fns.incDonated, .{zml.Tensor.fromShape(shape)}, platform, .{});
    defer exe.deinit();
    for (0..8) |seed| {
        const xd = randomData(seed);
        var bx = try upload(io, platform, &xd);
        defer bx.deinit();
        var res = try zml.testing.autoCall(allocator, io, &exe, Fns.incDonated, .{bx});
        defer res.deinit();
        var want = xd;
        for (&want) |*w| w.* += 1;
        try std.testing.expectEqual(want, try download(allocator, io, res));
    }
}

test "runtime: repeated calls through one Runner see their own inputs" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const x: zml.Tensor = .fromShape(shape);
    var exe = try zml.module.compile(allocator, io, Fns.axpy, .{ x, zml.Tensor.fromShape(shape) }, platform, .{});
    defer exe.deinit();
    var r = try exe.runner(allocator);
    defer r.deinit(allocator);
    for (0..16) |seed| {
        const xd = randomData(seed);
        const yd = randomData(seed + 7);
        var bx = try upload(io, platform, &xd);
        defer bx.deinit();
        var by = try upload(io, platform, &yd);
        defer by.deinit();
        var out: zml.Buffer = undefined;
        r.runOpts(io, .{ bx, by }, .{&out}, .{ .wait = true });
        defer out.deinit();
        try std.testing.expectEqual(axpyRef(xd, yd), try download(allocator, io, out));
    }
}

test "runtime: calls that aren't awaited, read back later" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const x: zml.Tensor = .fromShape(shape);
    var exe = try zml.module.compile(allocator, io, Fns.axpy, .{ x, zml.Tensor.fromShape(shape) }, platform, .{});
    defer exe.deinit();
    const k = 6;
    var ins: [k][2]zml.Buffer = undefined;
    var outs: [k]zml.Buffer = undefined;
    var datas: [k][2][n]f32 = undefined;
    for (0..k) |i| {
        datas[i] = .{ randomData(i), randomData(i + 50) };
        ins[i] = .{ try upload(io, platform, &datas[i][0]), try upload(io, platform, &datas[i][1]) };
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ ins[i][0], ins[i][1] });
        exe.callOpts(io, args, &results, .{ .wait = false });
        results.fill(.{&outs[i]});
    }
    for (0..k) |i| {
        try std.testing.expectEqual(axpyRef(datas[i][0], datas[i][1]), try download(allocator, io, outs[i]));
        outs[i].deinit();
        ins[i][0].deinit();
        ins[i][1].deinit();
    }
}

const Concurrent = struct {
    exe: *const zml.Exe,
    platform: *const zml.Platform,
    bad: std.atomic.Value(u32) = .init(0),

    fn worker(c: *Concurrent, id: usize) void {
        c.workerImpl(id) catch {
            _ = c.bad.fetchAdd(1000, .monotonic);
        };
    }

    fn workerImpl(c: *Concurrent, id: usize) !void {
        const allocator = std.heap.c_allocator;
        const io = std.testing.io;
        var r = try c.exe.runner(allocator);
        defer r.deinit(allocator);
        for (0..50) |it| {
            const xd = randomData(id * 1000 + it);
            const yd = randomData(id * 1000 + it + 500);
            var bx = try upload(io, c.platform, &xd);
            defer bx.deinit();
            var by = try upload(io, c.platform, &yd);
            defer by.deinit();
            var out: zml.Buffer = undefined;
            r.runOpts(io, .{ bx, by }, .{&out}, .{ .wait = true });
            defer out.deinit();
            const got = try download(allocator, io, out);
            if (!std.meta.eql(got, axpyRef(xd, yd))) _ = c.bad.fetchAdd(1, .monotonic);
        }
    }
};

test "runtime: concurrent calls of one executable" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const x: zml.Tensor = .fromShape(shape);
    var exe = try zml.module.compile(allocator, io, Fns.axpy, .{ x, zml.Tensor.fromShape(shape) }, platform, .{});
    defer exe.deinit();
    var c: Concurrent = .{ .exe = &exe, .platform = platform };
    var ts: [6]std.Thread = undefined;
    for (&ts, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Concurrent.worker, .{ &c, i });
    for (ts) |t| t.join();
    try std.testing.expectEqual(@as(u32, 0), c.bad.load(.monotonic));
}

const Compiles = struct {
    platform: *const zml.Platform,
    bad: std.atomic.Value(u32) = .init(0),

    fn worker(c: *Compiles, id: usize) void {
        c.workerImpl(id) catch {
            _ = c.bad.fetchAdd(1, .monotonic);
        };
    }

    fn workerImpl(c: *Compiles, id: usize) !void {
        const allocator = std.heap.c_allocator;
        const io = std.testing.io;
        for (0..4) |it| {
            const sh: zml.Shape = .init(.{ .a = @as(i64, @intCast(8 + id * 4 + it)) }, .f32);
            var exe = try zml.module.compile(allocator, io, Fns.neg, .{zml.Tensor.fromShape(sh)}, c.platform, .{});
            defer exe.deinit();
            var data: [64]f32 = undefined;
            for (&data, 0..) |*d, i| d.* = @floatFromInt(i + id);
            const len: usize = @intCast(sh.dim(0));
            var b = try zml.Buffer.fromBytes(io, c.platform, sh, .replicated, std.mem.sliceAsBytes(data[0..len]));
            defer b.deinit();
            var res = try zml.testing.autoCall(allocator, io, &exe, Fns.neg, .{b});
            defer res.deinit();
            const s = try res.toSliceAlloc(allocator, io);
            defer s.free(allocator);
            for (s.constItems(f32), data[0..len]) |g, d| if (g != -d) {
                _ = c.bad.fetchAdd(1, .monotonic);
                break;
            };
        }
    }
};

test "runtime: concurrent compiles" {
    const platform = zml.testing.env();
    var c: Compiles = .{ .platform = platform };
    var ts: [4]std.Thread = undefined;
    for (&ts, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Compiles.worker, .{ &c, i });
    for (ts) |t| t.join();
    try std.testing.expectEqual(@as(u32, 0), c.bad.load(.monotonic));
}

const Model = struct {
    const Input = struct { w: zml.Tensor, x: zml.Tensor };
    // A struct result: with a bare Tensor, FnExe.Output points into Buffer's own fields
    // (found/fnexe_tensor_out.zig).
    const Output = struct { y: zml.Tensor };
    fn forward(in: Input) Output {
        return .{ .y = in.w.scale(2).add(in.x) };
    }
};

test "runtime: FnExe runner with a baked prefix agrees with the raw call" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const T = zml.FnExe(Model.forward);
    const x: zml.Tensor = .fromShape(shape);
    var exe = try T.compile(allocator, io, platform, .{}, .{.{ .w = x, .x = zml.Tensor.fromShape(shape) }});
    defer exe.deinit();
    const wd = randomData(1);
    var bw = try upload(io, platform, &wd);
    defer bw.deinit();
    var r = try T.Runner(.{.w}).init(&exe, allocator, .{ .w = bw });
    defer r.deinit(allocator);
    for (0..8) |seed| {
        const xd = randomData(seed + 10);
        var bx = try upload(io, platform, &xd);
        defer bx.deinit();
        var out: zml.Buffer = undefined;
        r.run(io, .{ .inputs = .{ .x = bx }, .outputs = .{ .y = &out }, .opts = .{ .wait = true } });
        defer out.deinit();
        try std.testing.expectEqual(axpyRef(wd, xd), try download(allocator, io, out));
    }
}

const RngFns = struct {
    const Out = struct { rng: zml.Tensor.Rng, bits: zml.Tensor };
    fn draw(rng: zml.Tensor.Rng) Out {
        const next, const bits = rng.bitGenerator(.init(.{ .a = 32 }, .u32));
        return .{ .rng = next, .bits = bits };
    }
};

test "runtime: Rng.init state through initBuffer, for each algorithm" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    inline for (.{ .DEFAULT, .THREE_FRY, .PHILOX }) |alg| {
        var rng = zml.Tensor.Rng.init();
        rng.algorithm = alg;
        var exe = try zml.module.compile(allocator, io, RngFns.draw, .{rng}, platform, .{});
        defer exe.deinit();
        var draws: [3][32]u32 = undefined;
        var states: [3][2]u64 = undefined;
        for ([_]u128{ 1, 1, 2 }, 0..) |seed, i| {
            var s = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
            defer zml.Tensor.Rng.deinitBuffer(&s);
            var res = try zml.testing.autoCall(allocator, io, &exe, RngFns.draw, .{s});
            defer zml.Buffer.deinitAll(RngFns.Out, &res);
            const bits = try res.bits.toSliceAlloc(allocator, io);
            defer bits.free(allocator);
            @memcpy(&draws[i], bits.constItems(u32));
            const st = try res.rng._state.toSliceAlloc(allocator, io);
            defer st.free(allocator);
            @memcpy(&states[i], st.constItems(u64));
        }
        // Same seed, same bits; another seed, other bits; the state moves.
        try std.testing.expectEqual(draws[0], draws[1]);
        try std.testing.expect(!std.meta.eql(draws[0], draws[2]));
        try std.testing.expect(!std.meta.eql(states[0], [2]u64{ 1, 0 }));
        var distinct: usize = 0;
        for (draws[0], 0..) |x, i| {
            if (std.mem.indexOfScalar(u32, draws[0][0..i], x) == null) distinct += 1;
        }
        try std.testing.expect(distinct > 28);
    }
}
