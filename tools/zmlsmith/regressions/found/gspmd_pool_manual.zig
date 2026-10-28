const std = @import("std");
const zml = @import("zml");

// Upstream XLA GSPMD partitioner: a strided maxPool1d over a size 1 axis, then a
// manualComputation partitioned on the other axis, aborts the compiler with
// "spmd_partitioner.cc Check failed: !target.IsReplicatedOrSingleDevice()". Shardy
// compiles it. Found by zmlsmith with --gspmd (seed 171195).
test "gspmd_pool_manual" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Local = struct {
        fn body(x: zml.Tensor, out: zml.Shape) zml.Tensor {
            return x.negate().reshape(out);
        }
        fn forward(x: zml.Tensor) zml.Tensor {
            const p = x.maxPool1d(.{ .window_dimensions = 1, .window_strides = 3 }).values;
            const ps = p.withPartitioning(.{ .e = .data });
            return zml.ops.manualComputation(body, ps, ps.shape()).withPartitioning(.{ .e = .replicated });
        }
    };
    const mesh = try @constCast(platform).registerSharding("gspmd_pool_manual", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }));
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .e = 4, .n = 1 }, .f64));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{ .shardings = &.{mesh}, .partitioner = .gspmd });
    exe.deinit();
}
