const std = @import("std");
const zml = @import("zml");

// Upstream XLA SPMD partitioner: a pad with interior padding and negative edge padding
// on a partitioned axis aborts the compiler, CHECK failure in spmd_partitioner.cc
// "get_first_valid_element_offset_on_dilated_shard(0) == ...". Found by zmlsmith
// (seeds 121725 and 141095), where the axis was partitioned by manualComputation.
test "xla_spmd_pad" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Local = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return x.pad(0, .{ .p = zml.Tensor.Pad{ .low = -1, .high = -45, .interior = 2 } }).withPartitioning(.{ .p = .data });
        }
    };
    const mesh = try @constCast(platform).registerSharding("xla_spmd_pad", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }));
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .p = 40 }, .i32).withPartitioning(.{ .p = .data }));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{ .shardings = &.{mesh} });
    exe.deinit();
}
