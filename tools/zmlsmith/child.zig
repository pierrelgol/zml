//! Runs zmlsmith itself in a child process and classifies the outcome, so that ZML panics
//! and hangs are contained and turned into findings.

const std = @import("std");

pub const Outcome = struct {
    /// "ok", or the signature of the most severe finding, crash or timeout.
    signature: []const u8,
    /// Stdout and stderr, for the failure log.
    log: []const u8,
    crashed: bool,
    /// The node the finding is about, when known.
    node: ?u32 = null,
};

const severity = [_][]const u8{ "crash", "timeout", "shape", "compile_error", "output_shape", "value", "nondeterministic", "tags", "downstream" };

fn rank(sig: []const u8) usize {
    for (severity, 0..) |s, i| if (std.mem.startsWith(u8, sig, s)) return i;
    return severity.len;
}

pub fn selfExe(allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
    return std.process.executablePathAlloc(io, allocator);
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8, timeout_s: i64) !Outcome {
    const res = std.process.run(allocator, io, .{
        .argv = argv,
        .timeout = .{ .duration = .{ .raw = .fromSeconds(timeout_s), .clock = .awake } },
        .stdout_limit = .limited(64 << 20),
        .stderr_limit = .limited(64 << 20),
    }) catch |err| switch (err) {
        error.Timeout => return .{ .signature = "timeout", .log = "timeout", .crashed = true },
        else => return err,
    };
    const log = try std.mem.concat(allocator, u8, &.{ res.stdout, "\n--- stderr ---\n", res.stderr });
    const exited_ok = switch (res.term) {
        .exited => |c| c == 0,
        else => false,
    };
    if (!exited_ok) return .{ .signature = try crashSignature(allocator, res.stderr), .log = log, .crashed = true, .node = crashNode(res.stderr) };
    const sig = try worstResult(allocator, res.stdout, null);
    return .{ .signature = sig, .log = log, .crashed = false, .node = resultNode(res.stdout, sig) };
}

/// The most severe `sig=` among `RESULT` lines, optionally only for one seed.
pub fn worstResult(allocator: std.mem.Allocator, stdout: []const u8, seed: ?u64) ![]const u8 {
    var best: []const u8 = "ok";
    var lines = std.mem.splitScalar(u8, stdout, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "RESULT ")) continue;
        if (seed) |s| {
            const tag = try std.fmt.allocPrint(allocator, "seed={d} ", .{s});
            if (std.mem.indexOf(u8, line, tag) == null) continue;
        }
        const i = std.mem.indexOf(u8, line, "sig=") orelse continue;
        const sig = line[i + 4 ..];
        if (std.mem.startsWith(u8, sig, "downstream")) continue;
        if (std.mem.eql(u8, best, "ok") or rank(sig) < rank(best)) best = sig;
    }
    return allocator.dupe(u8, best);
}

fn resultNode(stdout: []const u8, sig: []const u8) ?u32 {
    var lines = std.mem.splitScalar(u8, stdout, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "RESULT ") or !std.mem.endsWith(u8, line, sig)) continue;
        const i = std.mem.indexOf(u8, line, " node=") orelse continue;
        const end = std.mem.indexOfScalarPos(u8, line, i + 6, ' ') orelse continue;
        return std.fmt.parseInt(u32, line[i + 6 .. end], 10) catch null;
    }
    return null;
}

fn crashNode(stderr: []const u8) ?u32 {
    const i = std.mem.indexOf(u8, stderr, "ZMLSMITH_CRASH node=%") orelse return null;
    const start = i + "ZMLSMITH_CRASH node=%".len;
    var end = start;
    while (end < stderr.len and std.ascii.isDigit(stderr[end])) end += 1;
    return std.fmt.parseInt(u32, stderr[start..end], 10) catch null;
}

/// `crash:<op>:<panic message with numbers and shapes elided>`.
pub fn crashSignature(allocator: std.mem.Allocator, stderr: []const u8) ![]const u8 {
    var op: []const u8 = "none";
    var msg: []const u8 = "unknown";
    var lines = std.mem.splitScalar(u8, stderr, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "ZMLSMITH_CRASH ")) |i| {
            if (std.mem.indexOf(u8, line[i..], "op=")) |j| op = line[i + j + 3 ..];
        }
        if (std.mem.indexOf(u8, line, "ZMLSMITH_PANIC ")) |i| msg = line[i + 15 ..];
        // XLA and absl fatal errors: `F0000 <time> <tid> file.cc:123] message`.
        if (line.len > 2 and line[0] == 'F' and std.ascii.isDigit(line[1])) {
            if (std.mem.indexOf(u8, line, "] ")) |i| {
                const head = line[0..i];
                const file_start = (std.mem.lastIndexOfScalar(u8, head, ' ') orelse 0) + 1;
                const file = head[file_start..];
                const colon = std.mem.indexOfScalar(u8, file, ':') orelse file.len;
                return std.fmt.allocPrint(allocator, "crash:xla:{s}:{s}", .{ file[0..colon], line[i + 2 ..][0..@min(70, line.len - i - 2)] });
            }
        }
    }
    // Which inactive union field was read depends on the data, not on the bug.
    if (std.mem.startsWith(u8, msg, "access of union field")) msg = "access of inactive union field";
    // Keep the words, drop values: shapes in braces and parenthesis, and any word holding
    // a digit (numbers, pointers, generated names) becomes '#'.
    var out: std.ArrayList(u8) = .empty;
    var depth: usize = 0;
    var i: usize = 0;
    while (i < msg.len and out.items.len < 90) {
        const c = msg[i];
        switch (c) {
            '{', '(', '[' => {
                depth += 1;
                i += 1;
                continue;
            },
            '}', ')', ']' => {
                depth -|= 1;
                i += 1;
                continue;
            },
            else => {},
        }
        if (depth > 0 or c == '\'' or c == '"') {
            i += 1;
            continue;
        }
        if (std.ascii.isAlphanumeric(c) or c == '@' or c == '_') {
            var j = i;
            var digit = false;
            while (j < msg.len and (std.ascii.isAlphanumeric(msg[j]) or msg[j] == '@' or msg[j] == '_')) : (j += 1) {
                if (std.ascii.isDigit(msg[j])) digit = true;
            }
            if (digit) try out.append(allocator, '#') else try out.appendSlice(allocator, msg[i..j]);
            i = j;
            continue;
        }
        try out.append(allocator, c);
        i += 1;
    }
    const elided = std.mem.trim(u8, out.items, " ");
    // Parser crashes are bucketed by the first ZML frame: the same message comes from many
    // unchecked accesses.
    if (std.mem.startsWith(u8, op, "parse.")) {
        if (zmlFrame(stderr)) |site| return std.fmt.allocPrint(allocator, "crash:{s}:{s}:{s}", .{ op, site, elided });
    }
    return std.fmt.allocPrint(allocator, "crash:{s}:{s}", .{ op, elided });
}

/// `zml/file.zig:line` of the first stack frame in ZML sources (zml/, vfs/, stdx/).
fn zmlFrame(stderr: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, stderr, '\n');
    while (lines.next()) |line| {
        for ([_][]const u8{ "/_main/zml/", "/_main/vfs/", "/_main/stdx/" }) |root| {
            const i = std.mem.indexOf(u8, line, root) orelse continue;
            const rest = line[i + "/_main/".len ..];
            // The assert helpers: the caller is the site.
            if (std.mem.startsWith(u8, rest, "stdx/debug.zig")) continue;
            const z = std.mem.indexOf(u8, rest, ".zig:") orelse continue;
            var end = z + ".zig:".len;
            while (end < rest.len and std.ascii.isDigit(rest[end])) end += 1;
            return rest[0..end];
        }
    }
    return null;
}

/// A directory friendly version of a signature.
pub fn slug(allocator: std.mem.Allocator, sig: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (sig) |c| {
        if (out.items.len >= 80) break;
        try out.append(allocator, if (std.ascii.isAlphanumeric(c) or c == '.' or c == '-') c else '_');
    }
    return out.items;
}
