//! Robustness audit: deliberately invalid uses of the API. ZML is expected to reject them
//! with a clear assertion. Zig safety panics (overflow, out of bounds, unreachable),
//! invalid MLIR or backend aborts mean a missing validation.
//!
//! `zmlsmith robust` runs every case in its own process and classifies the outcome.

const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

const child = @import("child.zig");

fn t(comptime dims: anytype, dt: zml.DataType) Shape {
    @setEvalBranchQuota(100_000);
    return .init(dims, dt);
}

/// Each case builds a program misusing one API; `x`, `y` are generic arguments.
pub const cases = blk: {
    @setEvalBranchQuota(1_000_000);
    break :blk cases_;
};
const cases_ = .{
    .{ "axis_equal_rank", struct {
        fn f(x: Tensor) Tensor {
            return x.sum(@as(i8, 2));
        }
    }.f, .{t(.{ .a = 2, .b = 3 }, .f32)} },
    .{ "axis_below_minus_rank", struct {
        fn f(x: Tensor) Tensor {
            return x.sum(@as(i8, -3));
        }
    }.f, .{t(.{ .a = 2, .b = 3 }, .f32)} },
    .{ "unknown_tag", struct {
        fn f(x: Tensor) Tensor {
            return x.sum(.z);
        }
    }.f, .{t(.{ .a = 2, .b = 3 }, .f32)} },
    .{ "add_mismatched_dims", struct {
        fn f(x: Tensor, y: Tensor) Tensor {
            return x.add(y);
        }
    }.f, .{ t(.{ .a = 2, .b = 3 }, .f32), t(.{ .a = 3, .b = 2 }, .f32) } },
    .{ "add_mismatched_dtype", struct {
        fn f(x: Tensor, y: Tensor) Tensor {
            return x.add(y);
        }
    }.f, .{ t(.{ .a = 2 }, .f32), t(.{ .a = 2 }, .i32) } },
    .{ "dot_mismatched_contracting", struct {
        fn f(x: Tensor, y: Tensor) Tensor {
            return x.dot(y, .k);
        }
    }.f, .{ t(.{ .m = 2, .k = 3 }, .f32), t(.{ .k = 4, .n = 2 }, .f32) } },
    .{ "dot_missing_tag", struct {
        fn f(x: Tensor, y: Tensor) Tensor {
            return x.dot(y, .k);
        }
    }.f, .{ t(.{ .m = 2, .k = 3 }, .f32), t(.{ .n = 3 }, .f32) } },
    .{ "concat_mismatched_other_dim", struct {
        fn f(x: Tensor, y: Tensor) Tensor {
            return Tensor.concatenate(&.{ x, y }, .a);
        }
    }.f, .{ t(.{ .a = 2, .b = 3 }, .f32), t(.{ .a = 2, .b = 4 }, .f32) } },
    .{ "concat_mismatched_dtype", struct {
        fn f(x: Tensor, y: Tensor) Tensor {
            return Tensor.concatenate(&.{ x, y }, .a);
        }
    }.f, .{ t(.{ .a = 2 }, .f32), t(.{ .a = 2 }, .f16) } },
    .{ "slice_start_after_end", struct {
        fn f(x: Tensor) Tensor {
            return x.slice(.a, .{ .start = 3, .end = 1 });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "slice_end_past_dim", struct {
        fn f(x: Tensor) Tensor {
            return x.slice(.a, .{ .start = 0, .end = 9 });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "slice_zero_step", struct {
        fn f(x: Tensor) Tensor {
            return x.slice(.a, .{ .start = 0, .end = 2, .step = 0 });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "slice_negative_step", struct {
        fn f(x: Tensor) Tensor {
            return x.slice(.a, .{ .start = 3, .end = 0, .step = -1 });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "slice_start_below_minus_dim", struct {
        fn f(x: Tensor) Tensor {
            return x.slice(.a, .{ .start = -9 });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "reshape_wrong_count", struct {
        fn f(x: Tensor) Tensor {
            return x.reshape(.{ 3, 3 });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "reshape_two_auto", struct {
        fn f(x: Tensor) Tensor {
            return x.reshape(.{ -1, -1 });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "transpose_wrong_len", struct {
        fn f(x: Tensor) Tensor {
            return x.transpose(.{ 1, 0, 2 });
        }
    }.f, .{t(.{ .a = 2, .b = 3 }, .f32)} },
    .{ "transpose_repeated_axis", struct {
        fn f(x: Tensor) Tensor {
            return x.transpose(.{ 0, 0 });
        }
    }.f, .{t(.{ .a = 2, .b = 3 }, .f32)} },
    .{ "squeeze_non_one", struct {
        fn f(x: Tensor) Tensor {
            return x.squeeze(.a);
        }
    }.f, .{t(.{ .a = 2, .b = 3 }, .f32)} },
    .{ "insert_axes_full_rank", struct {
        fn f(x: Tensor) Tensor {
            return x.insertAxes(0, .{.z});
        }
    }.f, .{t(.{ 1, 1, 1, 1, 1, 1, 1, 1 }, .f32)} },
    .{ "with_tags_wrong_count", struct {
        fn f(x: Tensor) Tensor {
            return x.withTags(.{.z});
        }
    }.f, .{t(.{ 2, 3 }, .f32)} },
    .{ "with_tags_duplicate", struct {
        fn f(x: Tensor) Tensor {
            return x.withTags(.{ .z, .z });
        }
    }.f, .{t(.{ 2, 3 }, .f32)} },
    .{ "broad_incompatible", struct {
        fn f(x: Tensor) Tensor {
            return x.broad(Shape.init(.{ .a = 3 }, .f32));
        }
    }.f, .{t(.{ .a = 2 }, .f32)} },
    .{ "broadcast_axes_out_of_range", struct {
        fn f(x: Tensor) Tensor {
            return x.broadcast(Shape.init(.{ 2, 3 }, .f32), &.{5});
        }
    }.f, .{t(.{2}, .f32)} },
    .{ "gather_float_indices", struct {
        fn f(x: Tensor, i: Tensor) Tensor {
            return x.gather(.{ .a = i }, .{});
        }
    }.f, .{ t(.{ .a = 4 }, .f32), t(.{ .n = 2 }, .f32) } },
    .{ "split_wrong_sum", struct {
        fn f(x: Tensor) Tensor {
            return x.split(.a, &.{ 1, 1 })[0];
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "chunk_exact_not_divisible", struct {
        fn f(x: Tensor) Tensor {
            return x.chunkExact(.a, 3)[0];
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "chunk_trailing_more_than_dim", struct {
        fn f(x: Tensor) Tensor {
            return x.chunkAllowTrailing(0, 5)[0];
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "topk_k_zero", struct {
        fn f(x: Tensor) Tensor {
            return x.topK(0, 0, .{}).values;
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "topk_k_too_large", struct {
        fn f(x: Tensor) Tensor {
            return x.topK(0, 9, .{}).values;
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "pad_negative_beyond_dim", struct {
        fn f(x: Tensor) Tensor {
            return x.pad(0, .{ .a = Tensor.Pad{ .low = -5 } });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "repeat_wrong_len", struct {
        fn f(x: Tensor) Tensor {
            return x.repeat(&.{ 2, 2 });
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "stutter_zero", struct {
        fn f(x: Tensor) Tensor {
            return x.stutter1d(0, 0);
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "select_non_bool", struct {
        fn f(p: Tensor, x: Tensor) Tensor {
            return p.select(x, x);
        }
    }.f, .{ t(.{ .a = 4 }, .i32), t(.{ .a = 4 }, .f32) } },
    .{ "cmp_mismatched_dims", struct {
        fn f(x: Tensor, y: Tensor) Tensor {
            return x.cmp(.LT, y);
        }
    }.f, .{ t(.{ .a = 4 }, .f32), t(.{ .a = 3 }, .f32) } },
    .{ "convert_complex", struct {
        fn f(x: Tensor) Tensor {
            return x.convert(.c64).abs();
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "bitcast_incompatible_last_dim", struct {
        fn f(x: Tensor) Tensor {
            return x.bitCast(.u32);
        }
    }.f, .{t(.{ .a = 3 }, .u8)} },
    .{ "unflatten_not_divisible", struct {
        fn f(x: Tensor) Tensor {
            return x.unflatten(0, 3);
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "unflatten_last_negative", struct {
        fn f(x: Tensor) Tensor {
            return x.unflatten(-1, 2);
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "dynamic_slice_float_offset", struct {
        fn f(x: Tensor, o: Tensor) Tensor {
            return x.slice(.a, .dyn(o, 2));
        }
    }.f, .{ t(.{ .a = 4 }, .f32), t(.{}, .f32) } },
    .{ "dynamic_slice_too_long", struct {
        fn f(x: Tensor, o: Tensor) Tensor {
            return x.slice(.a, .dyn(o, 9));
        }
    }.f, .{ t(.{ .a = 4 }, .f32), t(.{}, .i32) } },
    .{ "dynamic_update_larger_update", struct {
        fn f(x: Tensor, u: Tensor, o: Tensor) Tensor {
            return x.dynamicUpdateSlice1d(u, 0, o);
        }
    }.f, .{ t(.{ .a = 4 }, .f32), t(.{ .a = 6 }, .f32), t(.{}, .i32) } },
    .{ "triangular_same_axis", struct {
        fn f(x: Tensor) Tensor {
            return x.triangular(.{ 0, 0 }, 0);
        }
    }.f, .{t(.{ .a = 4, .b = 4 }, .f32)} },
    .{ "conv_mismatched_features", struct {
        fn f(x: Tensor, k: Tensor) Tensor {
            return x.conv1d(k, .{});
        }
    }.f, .{ t(.{ 1, 2, 5 }, .f32), t(.{ 1, 3, 2 }, .f32) } },
    .{ "cholesky_rank3", struct {
        fn f(x: Tensor) Tensor {
            return x.cholesky(true);
        }
    }.f, .{t(.{ 2, 3, 3 }, .f32)} },
    .{ "argmax_empty_axis", struct {
        fn f(x: Tensor) Tensor {
            return x.argMax(0).indices;
        }
    }.f, .{t(.{ .a = 0 }, .f32)} },
    .{ "mean_empty_axis", struct {
        fn f(x: Tensor) Tensor {
            return x.mean(0);
        }
    }.f, .{t(.{ .a = 0, .b = 2 }, .i32)} },
    .{ "sum_empty_tensor", struct {
        fn f(x: Tensor) Tensor {
            return x.sum(0);
        }
    }.f, .{t(.{ .a = 0 }, .f32)} },
    .{ "softmax_int", struct {
        fn f(x: Tensor) Tensor {
            return x.softmax(0);
        }
    }.f, .{t(.{ .a = 4 }, .i32)} },
    .{ "exp_int", struct {
        fn f(x: Tensor) Tensor {
            return x.exp();
        }
    }.f, .{t(.{ .a = 4 }, .i32)} },
    .{ "pow_int", struct {
        fn f(x: Tensor) Tensor {
            return x.pow(x);
        }
    }.f, .{t(.{ .a = 4 }, .i32)} },
    .{ "shift_float", struct {
        fn f(x: Tensor) Tensor {
            return x.shiftLeft(x);
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "not_float", struct {
        fn f(x: Tensor) Tensor {
            return x.not();
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "iota_tag_missing", struct {
        fn f() Tensor {
            return Tensor.iota(Shape.init(.{ .a = 3 }, .i32), .b);
        }
    }.f, .{} },
    .{ "arange_empty", struct {
        fn f() Tensor {
            return Tensor.arange(.{ .start = 3, .end = 3 }, .i32);
        }
    }.f, .{} },
    .{ "linspace_one_step", struct {
        fn f() Tensor {
            return Tensor.linspace(.{ .start = 0, .end = 1, .steps = 1 }, .f32);
        }
    }.f, .{} },
    .{ "scalar_nan", struct {
        fn f() Tensor {
            return Tensor.scalar(std.math.nan(f32), .f32);
        }
    }.f, .{} },
    .{ "scalar_out_of_range", struct {
        fn f() Tensor {
            return Tensor.scalar(300, .u8);
        }
    }.f, .{} },
    .{ "subbyte_odd_last_dim", struct {
        fn f(x: Tensor) Tensor {
            return x;
        }
    }.f, .{t(.{3}, .i4)} },
    .{ "subbyte_scalar", struct {
        fn f(x: Tensor) Tensor {
            return x;
        }
    }.f, .{t(.{}, .u4)} },
    .{ "reuse_buffer_twice", struct {
        fn f(x: Tensor) [2]Tensor {
            return .{ x.addConstant(1).reuseBuffer(x), x.addConstant(2).reuseBuffer(x) };
        }
    }.f, .{t(.{ .a = 4 }, .f32)} },
    .{ "while_cond_not_bool", struct {
        const W = struct {
            pub const State = struct { i: Tensor };
            pub fn cond(_: @This(), s: State) Tensor {
                return s.i;
            }
            pub fn body(_: @This(), s: State) State {
                return .{ .i = s.i.addConstant(1) };
            }
        };
        fn f(x: Tensor) Tensor {
            return zml.ops.@"while"(W, .{}, .{ .i = x }).i;
        }
    }.f, .{t(.{}, .i32)} },
    .{ "while_body_changes_shape", struct {
        const W = struct {
            pub const State = struct { i: Tensor, x: Tensor };
            pub fn cond(_: @This(), s: State) Tensor {
                return s.i.cmp(.LT, Tensor.scalar(2, .i32));
            }
            pub fn body(_: @This(), s: State) State {
                return .{ .i = s.i.addConstant(1), .x = s.x.slice(0, .{ .end = 1 }) };
            }
        };
        fn f(i: Tensor, x: Tensor) Tensor {
            return zml.ops.@"while"(W, .{}, .{ .i = i, .x = x }).x;
        }
    }.f, .{ t(.{}, .i32), t(.{ .a = 4 }, .f32) } },
    .{ "if_pred_not_scalar", struct {
        const If = struct {
            x: Tensor,
            pub fn onTrue(c: @This()) Tensor {
                return c.x;
            }
            pub fn onFalse(c: @This()) Tensor {
                return c.x.negate();
            }
        };
        fn f(p: Tensor, x: Tensor) Tensor {
            return zml.ops.@"if"(If, .{ .x = x }, p);
        }
    }.f, .{ t(.{ .a = 2 }, .bool), t(.{ .a = 4 }, .f32) } },
    .{ "if_branch_shapes_differ", struct {
        const If = struct {
            x: Tensor,
            pub fn onTrue(c: @This()) Tensor {
                return c.x;
            }
            pub fn onFalse(c: @This()) Tensor {
                return c.x.slice(0, .{ .end = 1 });
            }
        };
        fn f(p: Tensor, x: Tensor) Tensor {
            return zml.ops.@"if"(If, .{ .x = x }, p);
        }
    }.f, .{ t(.{}, .bool), t(.{ .a = 4 }, .f32) } },
    // unpackFp4 merges the packed axis with the trailing bitcast axis: undocumented
    // requirement that it is the last one.
    .{ "unpack_fp4_packed_not_last", struct {
        fn f(x: Tensor) Tensor {
            return zml.nn.unpackFp4(x, .stored, .k);
        }
    }.f, .{t(.{ .stored = 4, .out = 2 }, .u8)} },
    .{ "uniform_f8e4m3", struct {
        fn f(x: Tensor) Tensor {
            _, const y = (Tensor.Rng{ ._state = x }).uniform(t(.{ .a = 4 }, .f8e4m3), .{});
            return y;
        }
    }.f, .{t(.{2}, .u64)} },
    .{ "quantize_block_fp8_e5m2", struct {
        fn f(x: Tensor) Tensor {
            return zml.quantization.quantizeBlockFp8(x, .k, 2, .f8e5m2, .f32).values;
        }
    }.f, .{t(.{ .k = 4 }, .f32)} },
    .{ "quantize_nvfp4_block_remainder", struct {
        fn f(x: Tensor) Tensor {
            return zml.quantization.quantizeNvfp4(x, null, .k).values;
        }
    }.f, .{t(.{ .k = 24 }, .f32)} },
    // The name must be a namespaced op name, like "ns.op": ZML passes it through to MLIR.
    .{ "composite_unnamespaced_name", struct {
        fn decomposition(in: []const Tensor, _: void) Tensor {
            return in[0];
        }
        fn f(x: Tensor) Tensor {
            return zml.ops.composite("op", &.{x}, &.{x.shape()}, decomposition, {}, .{})[0];
        }
    }.f, .{t(.{ .a = 2 }, .f32)} },
    // The uniqueness guards on spatial dims only check the last element.
    .{ "conv2d_repeated_spatial_dim", struct {
        fn f(x: Tensor, w: Tensor) Tensor {
            return x.conv2d(w, .{ .input_spatial_dimensions = &.{ 2, 2 } });
        }
    }.f, .{ t(.{ 1, 1, 4, 4 }, .f32), t(.{ 1, 1, 2, 2 }, .f32) } },
    .{ "scaled_dot_narrow_output", struct {
        fn f(x: Tensor, y: Tensor) Tensor {
            return zml.nn.scaledDot(x, y, null, y, .f8e4m3fn, .k, .{});
        }
    }.f, .{ t(.{ .m = 2, .k = 32 }, .f8e4m3fn), t(.{ .n = 2, .k = 32 }, .f8e4m3fn) } },
};

/// Invalid uses of executables, arguments and results at run time, after a valid compile.
const Rt = struct {
    fn neg(x: Tensor) Tensor {
        return x.negate();
    }
    fn add(x: Tensor, y: Tensor) Tensor {
        return x.add(y);
    }
    fn incDonated(x: Tensor) Tensor {
        return x.addConstant(1).reuseBuffer(x);
    }

    const sh = t(.{ .a = 4 }, .f32);
    const data = [_]f32{ 1, 2, 3, 4 } ** 2;

    fn compile1(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, comptime f: anytype) !zml.Exe {
        return platform.compileFn(allocator, io, f, .{Tensor.fromShape(sh)}, .{});
    }
    fn buf(io: std.Io, platform: *const zml.Platform, s: Shape) !zml.Buffer {
        return zml.Buffer.fromBytes(io, platform, s, .replicated, std.mem.sliceAsBytes(&data)[0..s.byteSize()]);
    }
    fn call(allocator: std.mem.Allocator, io: std.Io, exe: *const zml.Exe, in: anytype, n_out: comptime_int) !void {
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(in);
        exe.callOpts(io, args, &results, .{ .wait = true });
        var outs: [n_out]zml.Buffer = undefined;
        var ptrs: [n_out]*zml.Buffer = undefined;
        for (&ptrs, &outs) |*p, *o| p.* = o;
        results.fill(ptrs);
        for (&outs) |*o| _ = try o.toSliceAlloc(allocator, io);
    }

    fn wrongShape(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        const exe = try compile1(allocator, io, platform, neg);
        try call(allocator, io, &exe, .{try buf(io, platform, t(.{ .a = 8 }, .f32))}, 1);
    }
    fn wrongDtype(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        const exe = try compile1(allocator, io, platform, neg);
        try call(allocator, io, &exe, .{try buf(io, platform, t(.{ .a = 4 }, .i32))}, 1);
    }
    fn tooManyArgs(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        const exe = try compile1(allocator, io, platform, neg);
        const b = try buf(io, platform, sh);
        try call(allocator, io, &exe, .{ b, b }, 1);
    }
    fn tooFewArgs(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        const exe = try platform.compileFn(allocator, io, add, .{ Tensor.fromShape(sh), Tensor.fromShape(sh) }, .{});
        try call(allocator, io, &exe, .{try buf(io, platform, sh)}, 1);
    }
    fn tooManyResults(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        const exe = try compile1(allocator, io, platform, neg);
        try call(allocator, io, &exe, .{try buf(io, platform, sh)}, 2);
    }
    fn readDonated(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        const exe = try compile1(allocator, io, platform, incDonated);
        const b = try buf(io, platform, sh);
        try call(allocator, io, &exe, .{b}, 1);
        // The argument buffer was donated to the result: an error is the expected answer.
        _ = b.toSliceAlloc(allocator, io) catch |err| {
            std.debug.print("ZMLSMITH_REJECTED {t}\n", .{err});
            std.process.exit(1);
        };
    }
    fn bakeTooMany(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        const exe = try compile1(allocator, io, platform, neg);
        const b = try buf(io, platform, sh);
        var args = try exe.args(allocator);
        args.bake(.{ b, b });
    }
    fn poolZeroBlocks(allocator: std.mem.Allocator, _: std.Io, _: *const zml.Platform) !void {
        var mem: [16]u8 = undefined;
        var p = try zml.mem.FixedBufferPool.init(allocator, &mem, 0);
        p.deinit(allocator);
    }
    fn poolEmptyBuffer(allocator: std.mem.Allocator, io: std.Io, _: *const zml.Platform) !void {
        var p = try zml.mem.FixedBufferPool.init(allocator, &.{}, 2);
        defer p.deinit(allocator);
        const b = try p.get(io);
        p.put(io, b);
    }
    fn squeezeNonUnit(_: std.mem.Allocator, _: std.Io, _: *const zml.Platform) !void {
        var bytes = [_]u8{0} ** 24;
        const s: zml.Slice = .init(t(.{ .a = 2, .b = 3 }, .f32), &bytes);
        // Drops an axis of 3 elements: the view silently loses data.
        const q = s.squeeze(.b);
        if (q.shape.count() != 2) return error.Unexpected;
    }
    fn poolNotDivisible(allocator: std.mem.Allocator, _: std.Io, _: *const zml.Platform) !void {
        var mem: [10]u8 = undefined;
        _ = try zml.mem.FixedBufferPool.init(allocator, &mem, 3);
    }
};

pub const runtime_cases = .{
    .{ "call_wrong_arg_shape", Rt.wrongShape },
    .{ "call_wrong_arg_dtype", Rt.wrongDtype },
    .{ "call_too_many_args", Rt.tooManyArgs },
    .{ "call_too_few_args", Rt.tooFewArgs },
    .{ "fill_too_many_results", Rt.tooManyResults },
    .{ "read_donated_argument", Rt.readDonated },
    .{ "bake_too_many", Rt.bakeTooMany },
    .{ "fixed_pool_not_divisible", Rt.poolNotDivisible },
    .{ "slice_squeeze_non_unit", Rt.squeezeNonUnit },
    .{ "fixed_pool_zero_blocks", Rt.poolZeroBlocks },
    .{ "fixed_pool_empty_buffer", Rt.poolEmptyBuffer },
};

/// Runs case `index` in process. It is expected to panic.
pub fn runCase(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, index: usize) !void {
    inline for (runtime_cases, 0..) |c, i| {
        if (index == cases.len + i) {
            try c[1](allocator, io, platform);
            std.debug.print("ZMLSMITH_ACCEPTED {s}\n", .{c[0]});
            return;
        }
    }
    inline for (cases, 0..) |c, i| {
        if (i == index) {
            var args: std.meta.ArgsTuple(@TypeOf(c[1])) = undefined;
            inline for (0..args.len) |j| args[j] = .fromShape(c[2][j]);
            var exe = try platform.compileFn(allocator, io, c[1], args, .{});
            exe.deinit();
            std.debug.print("ZMLSMITH_ACCEPTED {s}\n", .{c[0]});
            return;
        }
    }
    return error.NoSuchCase;
}

/// Zig safety checks and backend failures: the API didn't validate its inputs.
const unvalidated = [_][]const u8{
    "integer overflow",
    "index out of bounds",
    "reached unreachable",
    "attempt to unwrap error",
    "cast truncated bits",
    "attempt to cast negative",
    "division by zero",
    "exact division produced remainder",
    "incorrect alignment",
    "Failed to create MLIR operation",
    "ZML generated invalid mlir",
    "crash:xla",
    "access of union field",
    "start index",
    "out of bounds",
    // Memory errors, from undefined or freed memory reaching PJRT.
    "Segmentation fault",
    "General protection",
    "Illegal instruction",
    "Bus error",
};

pub fn audit(allocator: std.mem.Allocator, io: std.Io, exe_path: []const u8, timeout_s: i64) !usize {
    var bad: usize = 0;
    const all = cases ++ runtime_cases;
    inline for (all, 0..) |c, i| {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const idx = try std.fmt.allocPrint(a, "{d}", .{i});
        const out = try child.run(a, io, &.{ exe_path, "robust-case", "--seed", idx }, timeout_s);
        var verdict: []const u8 = "rejected";
        if (!out.crashed) {
            // Compiling and running without complaint: an invalid program was accepted.
            verdict = if (std.mem.indexOf(u8, out.log, "ZMLSMITH_ACCEPTED") != null) "ACCEPTED" else "rejected (error)";
        } else {
            for (unvalidated) |u| if (std.mem.indexOf(u8, out.signature, u) != null or std.mem.indexOf(u8, out.log, u) != null) {
                verdict = "UNVALIDATED";
                break;
            };
        }
        if (!std.mem.startsWith(u8, verdict, "rejected")) bad += 1;
        std.debug.print("{s: <34} {s: <12} {s}\n", .{ c[0], verdict, out.signature[0..@min(out.signature.len, 110)] });
    }
    return bad;
}
