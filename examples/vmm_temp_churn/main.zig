const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

// Each size gets its own executable whose preallocated temp holds the m x m f32 scores.
const row_counts = [_]u64{ 12288, 14336, 16384 };
const feature_count = 256;
const mib = 1024 * 1024;

const AllocatorKind = enum { vmm, bfc };

pub fn churn(x: zml.Tensor) zml.Tensor {
    const scores = x.dot(x.rename(.{ .m = .n }), .k);

    return scores.exp().sum(.n);
}

fn gpuAllocator(kind: AllocatorKind) zml.platform.CreateOptions.XlaGpu.Allocator {
    return switch (kind) {
        .vmm => .{ .vmm = .{ .memory_fraction = 0.90 } },
        .bfc => .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } },
    };
}

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\ vmm_temp_churn --allocator=vmm --headroom-mib=1200 --iterations=30
        ;
        allocator: AllocatorKind = .vmm,
        headroom_mib: u64 = 1200,
        iterations: usize = 30,
    };

    const allocator = init.gpa;
    const io = init.io;
    const cli_args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    const platform: *zml.Platform = try .auto(allocator, io, .{ .cuda = .{ .allocator = gpuAllocator(cli_args.allocator) } });
    defer platform.deinit(allocator, io);

    var exes: [row_counts.len]zml.Exe = undefined;
    var inputs: [row_counts.len]zml.Buffer = undefined;
    for (row_counts, &exes, &inputs) |row_count, *exe, *input| {
        const x = zml.Tensor.init(.{ .m = row_count, .k = feature_count }, .f32);

        exe.* = try platform.compileFn(allocator, io, churn, .{x}, .{});
        input.* = try zml.Buffer.uninitialized(io, platform, x.shape(), .{});

        const executable = try exe.exe.executable(platform.pjrt_api);
        defer executable.deinit(platform.pjrt_api);

        const stats = try executable.getCompiledMemoryStats(platform.pjrt_api);
        log.info("m={d}: temp={d} MiB", .{ row_count, stats.temp_size_in_bytes / mib });
    }
    defer for (&exes, &inputs) |*exe, *input| {
        exe.deinit();
        input.deinit();
    };

    // The "cache": a long-lived buffer leaving only `headroom_mib` of the allocator budget free.
    // bytes_in_use is ignored on purpose: under VMM it still counts stale autotuning buffers that
    // the cache allocation itself reclaims.
    const maybe_bytes_limit = platform.devices[0].memoryStats().bytes_limit;
    const bytes_limit = maybe_bytes_limit orelse return error.MissingBytesLimit;
    const cache_bytes = bytes_limit - cli_args.headroom_mib * mib;
    var cache = try zml.Buffer.uninitialized(io, platform, .init(.{ .bytes = cache_bytes }, .u8), .{});
    defer cache.deinit();
    log.info("allocator={t} bytes_limit={d} MiB cache={d} MiB headroom={d} MiB", .{ cli_args.allocator, bytes_limit / mib, cache_bytes / mib, cli_args.headroom_mib });

    var total_ns: [row_counts.len]u64 = @splat(0);
    for (0..cli_args.iterations * row_counts.len) |step| {
        const exe_index = step % row_counts.len;

        var exe_args = try exes[exe_index].args(allocator);
        defer exe_args.deinit(allocator);
        var exe_results = try exes[exe_index].results(allocator);
        defer exe_results.deinit(allocator);
        exe_args.set(.{inputs[exe_index]});

        const start: std.Io.Timestamp = .now(io, .awake);
        exes[exe_index].call(io, exe_args, &exe_results, .{});
        var result = exe_results.get(zml.Buffer);
        defer result.deinit();
        try result.await(io);
        const elapsed_ns: u64 = @intCast(start.untilNow(io, .awake).toNanoseconds());

        // The first pass maps each executable's temp for the first time; only steady state is measured.
        if (step >= row_counts.len) {
            total_ns[exe_index] += elapsed_ns;
        }
    }

    for (row_counts, total_ns) |row_count, exe_total_ns| {
        const avg_ms = @as(f64, @floatFromInt(exe_total_ns)) / @as(f64, @floatFromInt(cli_args.iterations - 1)) / std.time.ns_per_ms;
        log.info("m={d}: avg {d:.2} ms/iteration", .{ row_count, avg_ms });
    }
}
