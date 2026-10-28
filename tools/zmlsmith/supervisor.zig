//! The fuzzing loop: workers are child processes running batches of seeds, so panics and
//! hangs in ZML or XLA only cost one seed. Each new failure signature is saved, reduced
//! and turned into a Zig regression test.

const std = @import("std");

const child = @import("child.zig");
const emit = @import("emit.zig");
const gen = @import("gen.zig");
const ir = @import("ir.zig");
const reduce = @import("reduce.zig");

pub const Options = struct {
    seed: u64,
    count: u64,
    jobs: u32,
    batch: u64,
    out: []const u8,
    reduce: bool,
    timeout_s: i64,
    only: ?[]const u8 = null,
    exclude: ?[]const u8 = null,
    known: bool = false,
    mutate: ?[]const u8 = null,
};

const State = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    opts: Options,
    exe: []const u8,
    mutex: std.Io.Mutex = .init,
    next_seed: u64,
    end_seed: u64,
    seen: std.StringHashMapUnmanaged(u64) = .empty,
    done: u64 = 0,
    failing: u64 = 0,

    /// Claims the next batch of seeds, or null when done.
    fn claim(s: *State) ?struct { u64, u64 } {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        if (s.next_seed >= s.end_seed) return null;
        const start = s.next_seed;
        const n = @min(s.opts.batch, s.end_seed - start);
        s.next_seed += n;
        return .{ start, n };
    }

    /// Records a result; returns true when its signature is new.
    fn record(s: *State, seed: u64, sig: []const u8) !bool {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.done += 1;
        if (std.mem.eql(u8, sig, "ok")) return false;
        s.failing += 1;
        const gop = try s.seen.getOrPut(s.allocator, sig);
        if (gop.found_existing) {
            gop.value_ptr.* += 1;
            return false;
        }
        gop.key_ptr.* = try s.allocator.dupe(u8, sig);
        gop.value_ptr.* = 1;
        std.debug.print("[zmlsmith] new failure #{d} at seed {d}: {s}\n", .{ s.seen.count(), seed, sig });
        return true;
    }
};

pub fn fuzz(allocator: std.mem.Allocator, io: std.Io, opts: Options) !void {
    var state: State = .{
        .allocator = allocator,
        .io = io,
        .opts = opts,
        .exe = try child.selfExe(allocator, io),
        .next_seed = opts.seed,
        .end_seed = opts.seed + opts.count,
    };
    try std.Io.Dir.cwd().createDirPath(io, opts.out);
    const threads = try allocator.alloc(std.Thread, opts.jobs);
    for (threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, worker, .{ &state, i });
    for (threads) |t| t.join();

    var summary: std.Io.Writer.Allocating = .init(allocator);
    try summary.writer.print("seeds: {d}, failing: {d}, distinct signatures: {d}\n", .{ state.done, state.failing, state.seen.count() });
    var it = state.seen.iterator();
    while (it.next()) |e| try summary.writer.print("{d: >6}  {s}\n", .{ e.value_ptr.*, e.key_ptr.* });
    std.debug.print("{s}", .{summary.written()});
    const path = try std.fs.path.join(allocator, &.{ opts.out, "summary.txt" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = summary.written() });
}

fn worker(s: *State, id: usize) void {
    workerImpl(s, id) catch |err| std.debug.print("[zmlsmith] worker {d} failed: {t}\n", .{ id, err });
}

fn workerImpl(s: *State, id: usize) !void {
    const status = try std.fmt.allocPrint(s.allocator, "{s}/.worker-{d}.status", .{ s.opts.out, id });
    while (s.claim()) |batch| try runRange(s, status, batch[0], batch[1], false);
}

/// Seed the child was running when it was killed, from its `--status` file.
fn statusSeed(s: *State, a: std.mem.Allocator, path: []const u8) ?u64 {
    const data = std.Io.Dir.cwd().readFileAlloc(s.io, path, a, .limited(64)) catch return null;
    return std.fmt.parseInt(u64, std.mem.trim(u8, data, " \n"), 10) catch null;
}

/// Runs seeds [start, start + n) in child processes and records their results. A crash
/// costs one seed: the run resumes after it. A batch gets `timeout_s` plus 30 s per seed;
/// when it times out, the seed in progress is rerun alone with the full `timeout_s`, so a
/// slow batch isn't mistaken for a hang, and the seeds before it are rerun since their
/// results were lost with the child.
fn runRange(s: *State, status: []const u8, start_: u64, n_: u64, alone: bool) !void {
    var start, var n = .{ start_, n_ };
    while (n > 0) {
        var arena: std.heap.ArenaAllocator = .init(s.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(a, &.{ s.exe, "run", "--seed", try std.fmt.allocPrint(a, "{d}", .{start}), "--count", try std.fmt.allocPrint(a, "{d}", .{n}), "--status", status });
        if (s.opts.only) |o| try argv.appendSlice(a, &.{ "--only", o });
        if (s.opts.exclude) |o| try argv.appendSlice(a, &.{ "--exclude", o });
        if (s.opts.known) try argv.append(a, "--known");
        if (@import("runner.zig").mesh2d) try argv.append(a, "--mesh2d");
        if (@import("runner.zig").gspmd) try argv.append(a, "--gspmd");
        if (s.opts.mutate) |m| try argv.appendSlice(a, &.{ "--mutate", m });
        const timeout = if (alone) s.opts.timeout_s else s.opts.timeout_s + 30 * @as(i64, @intCast(n));
        const out = try child.run(a, s.io, argv.items, timeout);
        if (out.crashed and std.mem.eql(u8, out.signature, "timeout") and !alone) {
            const h = statusSeed(s, a, status) orelse start;
            const hung = if (h >= start and h < start + n) h else start;
            if (hung > start) try runRange(s, status, start, hung - start, false);
            try runRange(s, status, hung, 1, true);
            if (hung + 1 >= start + n) return;
            n -= hung + 1 - start;
            start = hung + 1;
            continue;
        }
        // Results of the seeds that completed.
        var finished: u64 = 0;
        while (finished < n) : (finished += 1) {
            const seed = start + finished;
            const marker = try std.fmt.allocPrint(a, "RESULT seed={d} ", .{seed});
            if (std.mem.indexOf(u8, out.log, marker) == null) break;
            const sig = try child.worstResult(a, out.log, seed);
            if (try s.record(seed, sig)) try saveFailure(s, a, seed, sig, out.log);
        }
        if (!out.crashed) return;
        // The seed after the last completed one crashed or hung.
        const crashed_seed = start + finished;
        if (try s.record(crashed_seed, out.signature)) try saveFailure(s, a, crashed_seed, out.signature, out.log);
        const consumed = crashed_seed - start + 1;
        if (consumed >= n) return;
        start += consumed;
        n -= consumed;
    }
}

fn saveFailure(s: *State, a: std.mem.Allocator, seed: u64, sig: []const u8, log: []const u8) !void {
    const dir = try std.fmt.allocPrint(a, "{s}/{s}", .{ s.opts.out, try child.slug(a, sig) });
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(s.io, dir);
    // The hang may be in the generator itself: don't regenerate in this process.
    if (std.mem.eql(u8, sig, "timeout")) {
        try cwd.writeFile(s.io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/seed-{d}.zs", .{ dir, seed }), .data = try std.fmt.allocPrint(a, "# signature: timeout\n# zmlsmith gen --seed {d}, then repro\n", .{seed}) });
        try cwd.writeFile(s.io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/seed-{d}.log", .{ dir, seed }), .data = log });
        return;
    }
    const cfg = try @import("main.zig").configFrom(s.opts.only, s.opts.exclude, s.opts.known);
    const prog, _ = try gen.generate(a, seed, &cfg);
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.print("# signature: {s}\n", .{sig});
    // The first failure of a signature may come from a seed that stopped avoiding a known
    // bug: say so, it is the first suspect.
    if (!s.opts.known) if (gen.liftedBug(seed)) |bug| try w.writer.print("# known bug not avoided in this seed: {t}\n", .{bug});
    try prog.format(&w.writer);
    try cwd.writeFile(s.io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/seed-{d}.zs", .{ dir, seed }), .data = w.written() });
    try cwd.writeFile(s.io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/seed-{d}.log", .{ dir, seed }), .data = log });
    if (!s.opts.reduce) return;
    // A hang costs the timeout per candidate.
    if (std.mem.eql(u8, sig, "timeout")) return;

    const res = reduce.reduce(a, s.io, &prog, .{ .signature = sig, .timeout_s = @min(s.opts.timeout_s, 120), .max_candidates = 400, .mutate = s.opts.mutate, .tmp_dir = s.opts.out }) catch |err| {
        std.debug.print("[zmlsmith] reducing seed {d} failed: {t}\n", .{ seed, err });
        return;
    };
    var rw: std.Io.Writer.Allocating = .init(a);
    try rw.writer.print("# signature: {s}\n# reduced from seed {d} in {d} candidates\n", .{ sig, seed, res.candidates });
    try res.program.format(&rw.writer);
    try cwd.writeFile(s.io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/reduced.zs", .{dir}), .data = rw.written() });
    const src = try emit.zigTest(a, &res.program);
    try cwd.writeFile(s.io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/reduced_test.zig", .{dir}), .data = src });
    std.debug.print("[zmlsmith] reduced {s}: {d} -> {d} nodes\n", .{ sig, prog.nodes.items.len, res.program.nodes.items.len });
}
