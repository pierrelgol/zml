const std = @import("std");
const zml = @import("zml");

// Upstream XLA GSPMD partitioner, on a two level mesh (`data` on the bus, `model` on the
// rings): a program returning nn.causalAttnMask next to an unrelated manualComputation
// result aborts the compiler with "hlo_sharding.cc Check failed: !IsManualLeaf() &&
// !IsUnknownLeaf()". Shardy compiles it, and so does GSPMD on the default one level mesh.
// Found by zmlsmith with --gspmd --mesh2d (seed 190135).
fn twoLevelMesh(allocator: std.mem.Allocator, target: zml.Target, devices: []const zml.platform.Device) anyerror!zml.Sharding.PhysicalMesh {
    const half = devices.len / 2;
    const groups = try allocator.alloc(zml.Sharding.PhysicalNode, 2);
    for (groups, 0..) |*grp, gi| {
        const leaves = try allocator.alloc(zml.Sharding.PhysicalNode, half);
        for (leaves, 0..) |*l, j| l.* = .device(devices[gi * half + j]);
        grp.* = .axis(.link, .{ .ring = .linear }, leaves);
    }
    return zml.Sharding.PhysicalMesh.fromTree(allocator, target, .axis(.bus, .tree, groups));
}

test "gspmd_mask_manual" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform: *zml.Platform = try .auto(allocator, io, .{ .physical_mesh = .{ .custom = twoLevelMesh } });
    defer platform.deinit(allocator, io);
    const Local = struct {
        fn body(x: zml.Tensor, out: zml.Shape) zml.Tensor {
            return x.relu().reshape(out);
        }
        fn forward(x: zml.Tensor) [2]zml.Tensor {
            const xs = x.withPartitioning(.{ .e = .data });
            const m = zml.ops.manualComputation(body, xs, xs.shape()).withPartitioning(.{ .e = .replicated });
            return .{ zml.nn.causalAttnMask(.{ .q = 5, .k = 2 }, .f32, null), m };
        }
    };
    const mesh = try platform.registerShardingWithStrategy("m", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }), .parseBindings(.{ .data = .bus, .model = .link }));
    const x: zml.Tensor = .fromShape(zml.Shape.init(.{ .m = 5, .e = 4 }, .f32));
    var exe = try platform.compileFn(allocator, io, Local.forward, .{x}, .{ .shardings = &.{mesh}, .partitioner = .gspmd });
    exe.deinit();
}
