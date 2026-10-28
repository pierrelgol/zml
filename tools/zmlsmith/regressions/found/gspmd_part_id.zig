const std = @import("std");
const zml = @import("zml");

// With Compiler.Options.partitioner = .gspmd, a manualComputation whose body calls
// ops.partitionId doesn't compile (PJRT returns error.Unimplemented; XLA's message is
// "PartitionId instruction is not supported for SPMD partitioning since the meaning is
// ambiguous"). The emitted module has no SPMDFullToShardShape before the partition_id, so
// the region is not marked manual in the GSPMD lowering. The same program compiles with
// the default Shardy partitioner. Found by zmlsmith with --gspmd (seed 150209).
test "gspmd_part_id" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Local = struct {
        fn body(x: zml.Tensor, out: zml.Shape) zml.Tensor {
            _ = x;
            return zml.ops.partitionId().convert(.i32).broad(out);
        }
        fn forward(x: zml.Tensor) zml.Tensor {
            const xs = x.withPartitioning(.{ .a = .data });
            return zml.ops.manualComputation(body, xs, xs.shape().withDtype(.i32)).withPartitioning(.{ .a = .replicated });
        }
    };
    const mesh = try @constCast(platform).registerSharding("gspmd_part_id", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }));
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .a = 4 }, .f32));
    // Shardy compiles it.
    var shardy = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{ .shardings = &.{mesh} });
    shardy.deinit();
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{ .shardings = &.{mesh}, .partitioner = .gspmd });
    exe.deinit();
}
