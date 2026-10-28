//! Minimal repro for the Metal multi-output reduce fusion miscompile.
//!
//! Computes `(h @ w_gate) * (h @ w_up)` with `h = rmsNorm(x)` over `seq` rows and checks it against a
//! host reference. With `seq=1`, XLA rewrites each dot into a reduce and `multi_output_fusion` merges
//! the two sibling reduces into one kernel; the Metal codegen only writes its first output, so `up`
//! reads back as zeros. See README.md.
//!
//!   bazel run //examples/metal_mof_repro --@zml//platforms:metal=true -- --seq=1 --dump=/tmp/mof_dump
const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;
const bf16 = zml.floats.BFloat16;

const eps = 1e-6;

/// Same structure as a Gemma MLP up-projection: both dots share a normalized `x`, and the weights use
/// the HF `{out, in}` layout so each output element is a contiguous row reduction.
fn gateUp(x: zml.Tensor, w_gate: zml.Tensor, w_up: zml.Tensor) zml.Tensor {
    const h = zml.nn.rmsNorm(x, .d, eps);
    return h.dot(w_gate, .d).mul(h.dot(w_up, .d));
}

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\ metal_mof_repro --seq=1 --dim=2816 --ffn=2112 [--dump=/abs/dir]
        ;
        seq: usize = 1,
        dim: usize = 2816,
        ffn: usize = 2112,
        dump: ?[]const u8 = null,
    };

    const allocator = init.gpa;
    const io = init.io;
    const cli_args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("platform: {s}", .{@tagName(platform.target)});

    const x: zml.Tensor = .init(.{ .s = cli_args.seq, .d = cli_args.dim }, .bf16);
    const w_gate: zml.Tensor = .init(.{ .f = cli_args.ffn, .d = cli_args.dim }, .bf16);
    const w_up: zml.Tensor = .init(.{ .f = cli_args.ffn, .d = cli_args.dim }, .bf16);

    var exe = try platform.compileFn(allocator, io, gateUp, .{ x, w_gate, w_up }, .{ .xla_dump_to = cli_args.dump });
    defer exe.deinit();
    if (cli_args.dump) |dir| log.info("HLO dumped to {s}", .{dir});

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();
    const x_host = try randomSlice(allocator, x.shape(), random);
    defer x_host.free(allocator);
    const wg_host = try randomSlice(allocator, w_gate.shape(), random);
    defer wg_host.free(allocator);
    const wu_host = try randomSlice(allocator, w_up.shape(), random);
    defer wu_host.free(allocator);

    var x_buf: zml.Buffer = try .fromSlice(io, platform, x_host);
    defer x_buf.deinit();
    var wg_buf: zml.Buffer = try .fromSlice(io, platform, wg_host);
    defer wg_buf.deinit();
    var wu_buf: zml.Buffer = try .fromSlice(io, platform, wu_host);
    defer wu_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ x_buf, wg_buf, wu_buf });
    exe.call(io, exe_args, &exe_results, .{});
    var out = exe_results.get(zml.Buffer);
    defer out.deinit();

    const got = try out.toSliceAlloc(allocator, io);
    defer got.free(allocator);
    const err = maxAbsError(got.items(bf16), x_host.items(bf16), wg_host.items(bf16), wu_host.items(bf16), cli_args);
    log.info("gate*up max|got|={d:.4} max|ref|={d:.4} max|got-ref|={d:.4} -> {s}", .{
        err.max_abs_got,
        err.max_abs_ref,
        err.max_abs_diff,
        if (err.max_abs_diff < 0.02 * err.max_abs_ref) "OK" else "WRONG",
    });
}

const Errors = struct { max_abs_got: f32 = 0, max_abs_ref: f32 = 0, max_abs_diff: f32 = 0 };

fn maxAbsError(got: []const bf16, x: []const bf16, w_gate: []const bf16, w_up: []const bf16, args: anytype) Errors {
    const dim = args.dim;
    const ffn = args.ffn;
    var e: Errors = .{};
    for (0..args.seq) |s| {
        const row = x[s * dim ..][0..dim];
        var sum_sq: f32 = 0;
        for (row) |v| sum_sq += v.toF32() * v.toF32();
        const inv_rms = 1 / @sqrt(sum_sq / @as(f32, @floatFromInt(dim)) + eps);
        for (0..ffn) |f| {
            var gate: f32 = 0;
            var up: f32 = 0;
            for (row, 0..) |v, d| {
                const h = v.toF32() * inv_rms;
                gate += h * w_gate[f * dim + d].toF32();
                up += h * w_up[f * dim + d].toF32();
            }
            const ref = gate * up;
            const g = got[s * ffn + f].toF32();
            e.max_abs_got = @max(e.max_abs_got, @abs(g));
            e.max_abs_ref = @max(e.max_abs_ref, @abs(ref));
            e.max_abs_diff = @max(e.max_abs_diff, @abs(g - ref));
        }
    }
    return e;
}

fn randomSlice(allocator: std.mem.Allocator, shape: zml.Shape, random: std.Random) !zml.Slice {
    const slice: zml.Slice = try .alloc(allocator, shape);
    for (slice.items(bf16)) |*v| v.* = .fromF32(random.float(f32) * 2 - 1);
    return slice;
}
