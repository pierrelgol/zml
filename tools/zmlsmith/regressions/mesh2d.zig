const std = @import("std");
const zml = @import("zml");

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

// Sharding on a two level physical mesh with `data` and `model` bound to distinct axes.
test "mesh2d: data x model partitioned reduction" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform: *zml.Platform = try .auto(allocator, io, .{ .physical_mesh = .{ .custom = twoLevelMesh } });
    defer platform.deinit(allocator, io);
    std.debug.print("physical mesh: {f}\n", .{platform.physical_mesh});
    const sharding = try platform.registerShardingWithStrategy("m", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }), .parseBindings(.{ .data = .bus, .model = .link }));
    std.debug.print("sharding: {f}\n", .{sharding});

    const Local = struct {
        fn f(x: zml.Tensor) zml.Tensor {
            return x.sum(.m).withPartitioning(.{ .d = .data });
        }
    };
    const sh = zml.Shape.init(.{ .d = 4, .m = 6 }, .f32).withPartitioning(.{ .d = .data, .m = .model });
    var data: [24]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @floatFromInt(i);
    var exe = try platform.compileFn(allocator, io, Local.f, .{zml.Tensor.fromShape(sh)}, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    var b = try zml.Buffer.fromBytes(io, platform, sh, sharding, std.mem.sliceAsBytes(&data));
    defer b.deinit();
    std.debug.print("input shards: {d}\n", .{b.numShards()});
    const back = try b.toSliceAlloc(allocator, io);
    defer back.free(allocator);
    try std.testing.expectEqualSlices(f32, &data, back.constItems(f32));
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.f, .{b});
    defer res.deinit();
    const s = try res.toSliceAlloc(allocator, io);
    defer s.free(allocator);
    for (0..4) |r| {
        var want: f32 = 0;
        for (0..6) |c| want += data[r * 6 + c];
        try std.testing.expectEqual(want, s.constItems(f32)[r]);
    }
}
