//! zmlsmith: a random program generator and differential tester for the ZML API.
//!
//!   zmlsmith gen --seed S                 print the program generated from a seed
//!   zmlsmith run --seed S [--count N]     run seeds in process, print one result per seed
//!   zmlsmith repro FILE                   run a program file in process
//!   zmlsmith fuzz [--jobs J] [--seed S] [--count N] [--out DIR]
//!                                         run seeds in crash isolated workers, reduce
//!                                         each new failure and write it to DIR
//!   zmlsmith reduce FILE|--seed S [--out FILE]
//!                                         shrink a failing program, keeping its signature

const std = @import("std");

const zml = @import("zml");

const gen = @import("gen.zig");
const ir = @import("ir.zig");
const lower = @import("lower.zig");
const ops = @import("ops.zig");
const reduce = @import("reduce.zig");
const runner = @import("runner.zig");
const emit = @import("emit.zig");

pub const std_options: std.Options = .{
    .log_level = .warn,
    .logFn = logFn,
};

/// The default log, plus the last PJRT compile message kept for compile error signatures.
fn logFn(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (level == .warn or level == .err) {
        var buf: [512]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, format, args) catch buf[0..];
        if (std.mem.indexOf(u8, msg, "[PJRT_Client_Compile] ")) |i| runner.setCompileMessage(msg[i + "[PJRT_Client_Compile] ".len ..]);
    }
    std.log.defaultLog(level, scope, format, args);
}

pub const panic = std.debug.FullPanic(panicHandler);

/// Prints which node was being lowered before the default panic output, so crashes can
/// be attributed and deduplicated by the supervisor.
/// The label of crashes outside lowering (parser fuzzing sets its target).
var crash_op: []const u8 = "none";
/// Skip the stack trace: symbolizing it in a Debug build takes about 12 s, and the reducer
/// only needs the signature, which doesn't use it (except for parser crashes).
var no_trace = false;

fn panicHandler(msg: []const u8, ret_addr: ?usize) noreturn {
    if (lower.current_node) |id| {
        if (lower.current_prog) |prog| {
            var buf: [256]u8 = undefined;
            var fba: std.heap.FixedBufferAllocator = .init(&buf);
            const label = runner.opLabel(fba.allocator(), &prog.nodes.items[id]) catch "?";
            std.debug.print("ZMLSMITH_CRASH node=%{d} op={s}\n", .{ id, label });
        }
    } else {
        std.debug.print("ZMLSMITH_CRASH node=none op={s}\n", .{crash_op});
    }
    std.debug.print("ZMLSMITH_PANIC {s}\n", .{msg});
    if (no_trace) std.process.abort();
    std.debug.defaultPanic(msg, ret_addr);
}

const Args = struct {
    cmd: []const u8 = "",
    seed: u64 = 1,
    count: u64 = 1,
    jobs: u32 = 4,
    batch: u64 = 20,
    out: []const u8 = "zmlsmith-out",
    file: ?[]const u8 = null,
    all_nodes: bool = false,
    no_reduce: bool = false,
    status: ?[]const u8 = null,
    timeout_s: i64 = 600,
    verbose: bool = false,
    only: ?[]const u8 = null,
    exclude: ?[]const u8 = null,
    known: bool = false,
    mutate: ?[]const u8 = null,
    target: ?[]const u8 = null,
};

/// Generation config from `--only a,b` and `--exclude c,d` family lists.
pub fn configFrom(only: ?[]const u8, exclude: ?[]const u8, known: bool) !gen.Config {
    var cfg: gen.Config = .{};
    if (known) cfg.avoid = .initEmpty();
    if (only) |list| {
        cfg.weights = @splat(0);
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |name| {
            const fam = std.meta.stringToEnum(ops.Family, name) orelse return error.UnknownFamily;
            cfg.weights[@intFromEnum(fam)] = @max(1, ops.default_weights[@intFromEnum(fam)]);
        }
    }
    if (exclude) |list| {
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |name| {
            const fam = std.meta.stringToEnum(ops.Family, name) orelse return error.UnknownFamily;
            cfg.weights[@intFromEnum(fam)] = 0;
        }
    }
    return cfg;
}

fn parseArgs(argv: []const [:0]const u8) !Args {
    var a: Args = .{};
    if (argv.len < 2) return error.Usage;
    a.cmd = argv[1];
    var i: usize = 2;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        const val = if (i + 1 < argv.len) argv[i + 1] else "";
        if (std.mem.eql(u8, arg, "--seed")) {
            a.seed = try std.fmt.parseInt(u64, val, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--count")) {
            a.count = try std.fmt.parseInt(u64, val, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--jobs")) {
            a.jobs = try std.fmt.parseInt(u32, val, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--batch")) {
            a.batch = try std.fmt.parseInt(u64, val, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--timeout")) {
            a.timeout_s = try std.fmt.parseInt(i64, val, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--out")) {
            a.out = val;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--only")) {
            a.only = val;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--exclude")) {
            a.exclude = val;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--status")) {
            a.status = val;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--all-nodes")) {
            a.all_nodes = true;
        } else if (std.mem.eql(u8, arg, "--no-trace")) {
            no_trace = true;
        } else if (std.mem.eql(u8, arg, "--mesh2d")) {
            runner.mesh2d = true;
        } else if (std.mem.eql(u8, arg, "--gspmd")) {
            runner.gspmd = true;
        } else if (std.mem.eql(u8, arg, "--mutate")) {
            a.mutate = val;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--target")) {
            a.target = val;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--known")) {
            a.known = true;
        } else if (std.mem.eql(u8, arg, "--no-reduce")) {
            a.no_reduce = true;
        } else if (std.mem.eql(u8, arg, "-v")) {
            a.verbose = true;
        } else {
            a.file = arg;
        }
    }
    return a;
}

pub fn main(init: std.process.Init) !void {
    // Crashes are expected in every child: writing core files costs seconds each.
    std.posix.setrlimit(.CORE, .{ .cur = 0, .max = 0 }) catch {};
    const allocator = init.gpa;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const args = parseArgs(argv) catch {
        std.debug.print(
            \\usage: zmlsmith gen|run|repro|fuzz|reduce [--seed S] [--count N] [--jobs J] [--out DIR] [FILE]
            \\
        , .{});
        std.process.exit(2);
    };

    const cfg = configFrom(args.only, args.exclude, args.known) catch {
        std.debug.print("unknown family in --only/--exclude\n", .{});
        std.process.exit(2);
    };
    if (args.mutate) |m| ops.mutation = std.meta.stringToEnum(ops.Mutation, m) orelse {
        std.debug.print("unknown mutation {s}\n", .{m});
        std.process.exit(2);
    };
    if (std.mem.eql(u8, args.cmd, "gen")) {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const prog, _ = try gen.generate(arena.allocator(), args.seed, &cfg);
        try printStdout(io, arena.allocator(), "{f}", .{&prog});
        return;
    }
    if (std.mem.eql(u8, args.cmd, "selfcheck")) {
        // Every generated program must survive print -> parse -> print, and replaying it
        // must give the values seen during generation. The reducer and `repro` rely on it.
        var failures: usize = 0;
        for (0..args.count) |i| {
            var arena: std.heap.ArenaAllocator = .init(allocator);
            defer arena.deinit();
            const a = arena.allocator();
            const seed = args.seed + i;
            const prog, const values = try gen.generate(a, seed, &cfg);
            var w1: std.Io.Writer.Allocating = .init(a);
            try prog.format(&w1.writer);
            const parsed = ir.parse(a, w1.written()) catch |err| {
                std.debug.print("seed {d}: parse failed: {t}\n", .{ seed, err });
                failures += 1;
                continue;
            };
            var w2: std.Io.Writer.Allocating = .init(a);
            try parsed.format(&w2.writer);
            if (!std.mem.eql(u8, w1.written(), w2.written())) {
                std.debug.print("seed {d}: roundtrip differs\n", .{seed});
                failures += 1;
                continue;
            }
            const replay = try runner.reference(a, &parsed);
            for (values, replay, 0..) |v, r, j| {
                if (parsed.nodes.items[j].scope != 0) continue;
                if (!std.mem.eql(u8, v.bytes, r.bytes)) {
                    std.debug.print("seed {d}: replay differs at %{d}\n", .{ seed, j });
                    failures += 1;
                    break;
                }
            }
        }
        std.debug.print("selfcheck: {d} seeds, {d} failures\n", .{ args.count, failures });
        if (failures > 0) std.process.exit(1);
        return;
    }
    if (std.mem.eql(u8, args.cmd, "stats")) {
        // How often each family, kind and dtype is generated.
        var counts: std.StringArrayHashMapUnmanaged(usize) = .empty;
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var nodes: usize = 0;
        for (0..args.count) |i| {
            const prog, _ = try gen.generate(a, args.seed + i, &cfg);
            for (prog.nodes.items) |*n| {
                nodes += 1;
                const label = try runner.opLabel(a, n);
                const gop = try counts.getOrPut(a, label);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* += 1;
                const dk = try std.fmt.allocPrint(a, "dtype.{t}", .{n.shape.dtype()});
                const g2 = try counts.getOrPut(a, dk);
                if (!g2.found_existing) g2.value_ptr.* = 0;
                g2.value_ptr.* += 1;
            }
        }
        const Ctx = struct {
            keys: [][]const u8,
            pub fn lessThan(c: @This(), l: usize, r: usize) bool {
                return std.mem.lessThan(u8, c.keys[l], c.keys[r]);
            }
        };
        counts.sort(Ctx{ .keys = counts.keys() });
        var out: std.Io.Writer.Allocating = .init(a);
        try out.writer.print("{d} programs, {d} nodes\n", .{ args.count, nodes });
        var it = counts.iterator();
        while (it.next()) |e| try out.writer.print("{d: >8}  {s}\n", .{ e.value_ptr.*, e.key_ptr.* });
        try out.writer.writeAll("\nfamily: committed / attempted (rejected by propose, by commit)\n");
        for (0..@import("ops.zig").family_count) |f| {
            const at = gen.stats.attempts[f];
            if (at == 0) continue;
            const c = gen.stats.commits[f];
            const cr = gen.stats.commit_rejects[f];
            try out.writer.print("{d: >6} / {d: >6} {d: >4}%  ({d}, {d})  {t}\n", .{ c, at, c * 100 / at, at - c - cr, cr, @as(@import("ops.zig").Family, @enumFromInt(f)) });
        }
        try printStdout(io, a, "{s}", .{out.written()});
        return;
    }
    if (std.mem.eql(u8, args.cmd, "robust")) {
        const robust = @import("robust.zig");
        const exe_path = try @import("child.zig").selfExe(init.arena.allocator(), io);
        const bad = try robust.audit(allocator, io, exe_path, args.timeout_s);
        std.debug.print("robust: {d} cases, {d} not cleanly rejected\n", .{ @typeInfo(@TypeOf(robust.cases)).@"struct".fields.len + @typeInfo(@TypeOf(robust.runtime_cases)).@"struct".fields.len, bad });
        return;
    }
    if (std.mem.eql(u8, args.cmd, "host")) {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const n = try @import("host.zig").run(arena.allocator(), args.seed, args.count);
        if (n > 0) std.process.exit(1);
        return;
    }
    if (std.mem.eql(u8, args.cmd, "emit")) {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const prog = try loadProgram(arena.allocator(), io, args, &cfg);
        const src = try emit.zigTest(arena.allocator(), &prog);
        try printStdout(io, arena.allocator(), "{s}", .{src});
        return;
    }
    if (std.mem.eql(u8, args.cmd, "fuzz")) {
        return @import("supervisor.zig").fuzz(std.heap.smp_allocator, io, .{
            .seed = args.seed,
            .count = args.count,
            .jobs = args.jobs,
            .batch = args.batch,
            .out = args.out,
            .reduce = !args.no_reduce,
            .timeout_s = args.timeout_s,
            .only = args.only,
            .exclude = args.exclude,
            .known = args.known,
            .mutate = args.mutate,
        });
    }
    if (std.mem.eql(u8, args.cmd, "reduce")) {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const prog = try loadProgram(arena.allocator(), io, args, &cfg);
        const res = try reduce.reduce(arena.allocator(), io, &prog, .{ .timeout_s = @min(args.timeout_s, 120), .verbose = args.verbose, .mutate = args.mutate });
        try printStdout(io, arena.allocator(), "# signature: {s}\n{f}", .{ res.signature, &res.program });
        return;
    }

    if (std.mem.startsWith(u8, args.cmd, "parse-")) {
        const parse = @import("parse.zig");
        const target = std.meta.stringToEnum(parse.Target, args.target orelse "") orelse {
            std.debug.print("--target must be one of safetensors, tokenizer, rope\n", .{});
            std.process.exit(2);
        };
        crash_op = switch (target) {
            inline else => |t| "parse." ++ @tagName(t),
        };
        if (std.mem.eql(u8, args.cmd, "parse-gen")) {
            var arena: std.heap.ArenaAllocator = .init(allocator);
            defer arena.deinit();
            const bytes = try parse.generate(arena.allocator(), target, args.seed);
            try printStdout(io, arena.allocator(), "{s}", .{bytes});
            return;
        }
        if (std.mem.eql(u8, args.cmd, "parse-batch")) return parse.batch(io, target, args.seed, args.count, args.out);
        if (std.mem.eql(u8, args.cmd, "parse-run")) {
            const path = args.file orelse return error.MissingFile;
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
            defer allocator.free(bytes);
            const dir = std.fs.path.dirname(path) orelse ".";
            parse.run(io, target, bytes, dir);
            return;
        }
        if (std.mem.eql(u8, args.cmd, "parse-fuzz")) return parse.fuzz(io, .{
            .target = target,
            .seed = args.seed,
            .count = args.count,
            .jobs = args.jobs,
            .out = args.out,
            .timeout_s = @min(args.timeout_s, 60),
            .exe = try @import("child.zig").selfExe(allocator, io),
        });
        std.debug.print("unknown command {s}\n", .{args.cmd});
        std.process.exit(2);
    }

    // In process commands need a platform.
    const create: zml.platform.CreateOptions = if (runner.mesh2d) .{ .physical_mesh = .{ .custom = runner.twoLevelMesh } } else .{};
    const platform: *zml.Platform = try .auto(allocator, io, create);
    defer platform.deinit(allocator, io);
    // On CPU the suggested strategy binds every logical axis to the bus: with the two level
    // mesh, bind them to distinct physical axes explicitly.
    runner.mesh = if (runner.mesh2d)
        try platform.registerShardingWithStrategy("zmlsmith", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }), .parseBindings(.{ .data = .bus, .model = .link }))
    else
        try platform.registerSharding("zmlsmith", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }));

    if (std.mem.eql(u8, args.cmd, "buffers")) {
        const n = try @import("buffers.zig").run(allocator, io, platform, args.seed, args.count);
        if (n > 0) std.process.exit(1);
        return;
    }
    if (std.mem.eql(u8, args.cmd, "robust-case")) {
        try @import("robust.zig").runCase(allocator, io, platform, args.seed);
        return;
    }
    if (std.mem.eql(u8, args.cmd, "repro")) {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const prog = try loadProgram(arena.allocator(), io, args, &cfg);
        if (args.verbose) std.debug.print("{f}", .{&prog});
        const report = try runner.run(arena.allocator(), io, platform, &prog, .{ .all_nodes = args.all_nodes });
        try printReport(io, arena.allocator(), prog.seed, report);
        return;
    }
    if (std.mem.eql(u8, args.cmd, "emit-batch")) {
        // Regression tests for every passing seed, to check that emitted code compiles
        // and reproduces the interpreter results through the public API.
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();
        try out.writer.writeAll("const std = @import(\"std\");\nconst zml = @import(\"zml\");\n");
        var kept: usize = 0;
        for (0..args.count) |i| {
            var arena: std.heap.ArenaAllocator = .init(allocator);
            defer arena.deinit();
            const prog, _ = try gen.generate(arena.allocator(), args.seed + i, &cfg);
            const report = try runner.run(arena.allocator(), io, platform, &prog, .{});
            if (report.findings.items.len > 0) continue;
            const src = try emit.zigTest(arena.allocator(), &prog);
            // Drop the per-test imports, the file has them once.
            const body = src[std.mem.indexOf(u8, src, "\n// Generated").? + 1 ..];
            try out.writer.writeAll(body);
            kept += 1;
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args.out, .data = out.written() });
        std.debug.print("wrote {d} tests to {s}\n", .{ kept, args.out });
        return;
    }
    if (std.mem.eql(u8, args.cmd, "run")) {
        for (0..args.count) |i| {
            const seed = args.seed + i;
            if (args.status) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = try std.fmt.allocPrint(init.arena.allocator(), "{d}\n", .{seed}) });
            var arena: std.heap.ArenaAllocator = .init(allocator);
            defer arena.deinit();
            const prog, _ = try gen.generate(arena.allocator(), seed, &cfg);
            if (args.verbose) std.debug.print("{f}", .{&prog});
            const report = try runner.run(arena.allocator(), io, platform, &prog, .{ .all_nodes = args.all_nodes });
            try printReport(io, arena.allocator(), seed, report);
        }
        return;
    }
    std.debug.print("unknown command {s}\n", .{args.cmd});
    std.process.exit(2);
}

fn loadProgram(allocator: std.mem.Allocator, io: std.Io, args: Args, cfg: *const gen.Config) !ir.Program {
    if (args.file) |path| {
        const src = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 << 20));
        return ir.parse(allocator, src);
    }
    const prog, _ = try gen.generate(allocator, args.seed, cfg);
    return prog;
}

test {
    _ = ir;
    _ = reduce;
    _ = gen;
    _ = ops;
    _ = @import("Value.zig");
}

pub fn printStdout(io: std.Io, allocator: std.mem.Allocator, comptime fmt: []const u8, fmt_args: anytype) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, fmt_args);
    try std.Io.File.stdout().writeStreamingAll(io, s);
}

/// One line per finding, `RESULT seed=S kind=K sig=SIG`, then indented details. The
/// supervisor parses these lines.
fn printReport(io: std.Io, allocator: std.mem.Allocator, seed: u64, report: runner.Report) !void {
    if (report.findings.items.len == 0) return printStdout(io, allocator, "RESULT seed={d} kind=ok\n", .{seed});
    for (report.findings.items) |f| {
        try printStdout(io, allocator, "RESULT seed={d} kind={t} node={d} sig={s}\n    {s}\n", .{ seed, f.kind, f.node, f.signature, f.detail });
    }
}
