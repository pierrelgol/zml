const std = @import("std");
const zml = @import("zml");

const Config = @import("Config.zig");

const KvCache = @This();

sliding: Pool,
full: Pool,

pub fn init(config: Config, seqlen: u32, dtype: zml.DataType, mesh: *const zml.Mesh) KvCache {
    const text = config.text_config;
    const sliding: zml.Shape = .init(.{
        .layer = config.countLayers(.sliding_attention),
        .k = seqlen,
        .h = text.num_key_value_heads,
        .hd = text.head_dim,
    }, dtype);
    const full: zml.Shape = .init(.{
        .layer = config.countLayers(.full_attention),
        .k = seqlen,
        .h = text.num_global_key_value_heads,
        .hd = text.global_head_dim,
    }, dtype);
    return .{ .sliding = .init(sliding, mesh), .full = .init(full, mesh) };
}

pub fn pool(self: KvCache, kind: Config.LayerType) Pool {
    switch (kind) {
        .sliding_attention => {
            return self.sliding;
        },
        .full_attention => {
            return self.full;
        },
    }
}

pub fn withPool(self: KvCache, kind: Config.LayerType, updated: Pool) KvCache {
    var cache = self;
    switch (kind) {
        .sliding_attention => {
            cache.sliding = updated;
        },
        .full_attention => {
            cache.full = updated;
        },
    }
    return cache;
}

pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
    var sliding = try self.sliding.initBuffer(io, platform);
    errdefer Pool.deinitBuffer(&sliding);
    return .{ .sliding = sliding, .full = try self.full.initBuffer(io, platform) };
}

pub fn deinitBuffer(buffers: *zml.Bufferized(KvCache)) void {
    Pool.deinitBuffer(&buffers.sliding);
    Pool.deinitBuffer(&buffers.full);
}

pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
    return .{ .sliding = self.sliding.reuseBuffer(other.sliding), .full = self.full.reuseBuffer(other.full) };
}

pub const Pool = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub fn init(shape: zml.Shape, mesh: *const zml.Mesh) Pool {
        var sharded = shape.withPartitioning(mesh, .{ .h = .replicated });
        if (@rem(shape.dim(.h), mesh.numPartitionsForLogicalAxis(.model)) == 0) {
            sharded = shape.withPartitioning(mesh, .{ .h = .model });
        }
        return .{ .k = .fromShape(sharded), .v = .fromShape(sharded) };
    }

    pub fn write(self: Pool, layer_index: u32, k: zml.Tensor, v: zml.Tensor, start: zml.Tensor) Pool {
        const offset = .{ .layer = zml.Tensor.scalar(layer_index, .u32), .k = start.convert(.u32) };
        return .{
            .k = self.k.dynamicUpdateSlice(offset, k.rename(.{ .s = .k }).convert(self.k.dtype())),
            .v = self.v.dynamicUpdateSlice(offset, v.rename(.{ .s = .k }).convert(self.v.dtype())),
        };
    }

    pub fn read(self: Pool, layer_index: u32, written: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const k = self.k.slice(.layer, .single(layer_index));
        const v = self.v.slice(.layer, .single(layer_index));
        const slot = zml.Tensor.iota(k.shape().withDtype(.i32), .k);
        const valid = slot.cmp(.LT, written.convert(.i32).broad(slot.shape()));
        return .{ valid.select(k, .zeroes(k.shape())), valid.select(v, .zeroes(v.shape())) };
    }

    pub fn initBuffer(self: Pool, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Pool) {
        var k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), .{});
        errdefer k.deinit();
        return .{ .k = k, .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), .{}) };
    }

    pub fn deinitBuffer(buffers: *zml.Bufferized(Pool)) void {
        buffers.k.deinit();
        buffers.v.deinit();
    }

    pub fn reuseBuffer(self: Pool, other: Pool) Pool {
        return .{ .k = self.k.reuseBuffer(other.k), .v = self.v.reuseBuffer(other.v) };
    }
};
