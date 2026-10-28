//! Regression tests produced by zmlsmith. Each reduced failure saved by `zmlsmith fuzz`
//! (`<out>/<signature>/reduced_test.zig`) can be copied here and imported below.

test {
    _ = @import("generated.zig");
    _ = @import("generated_sharding.zig");
    _ = @import("generated_quant.zig");
    _ = @import("generated_composite.zig");
    _ = @import("generated_rng.zig");
    _ = @import("generated_ieee.zig");
    _ = @import("generated_attention.zig");
    _ = @import("generated_options.zig");
    _ = @import("loader.zig");
    _ = @import("runtime.zig");
    _ = @import("mesh2d.zig");
    _ = @import("vfs_http.zig");
    _ = @import("meta.zig");
    _ = @import("dtype_props.zig");
    _ = @import("tok_props.zig");
    _ = @import("store_view.zig");
    _ = @import("paged_props.zig");
    _ = @import("kv_update.zig");
    _ = @import("sampling.zig");
    _ = @import("iree_props.zig");
    _ = @import("fixed_pool.zig");
    _ = @import("print.zig");
    _ = @import("conv_layouts.zig");
    _ = @import("loader_shard.zig");
    _ = @import("hf_vfs.zig");
}
