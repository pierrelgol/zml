//! The registry of op families. Each family is a namespace with:
//!
//! - `Attrs`: the serializable attributes of a node,
//! - `weight`: default generation weight (0 for families that are never proposed),
//! - `propose(g)`: builds a valid node from the current program, or returns null,
//! - `eval(e, attrs, node, ins)`: the reference semantics, written from the docs,
//! - `lower(attrs, node, ins)`: the ZML API calls under test,
//! - `emit(attrs, node, w, ins)`: the same calls as Zig source, for regression tests.

const std = @import("std");

const zml = @import("zml");

const gen = @import("gen.zig");
const ir = @import("ir.zig");
const Value = @import("Value.zig");

const source = @import("ops/source.zig");
const elementwise = @import("ops/elementwise.zig");
const layout = @import("ops/layout.zig");
const reduce_ops = @import("ops/reduce.zig");
const linalg = @import("ops/linalg.zig");
const indexing = @import("ops/indexing.zig");
const misc = @import("ops/misc.zig");
const window = @import("ops/window.zig");
const scatter_ops = @import("ops/scatter.zig");
const nn = @import("ops/nn.zig");
const control = @import("ops/control.zig");
const dsl = @import("ops/dsl.zig");
const special = @import("ops/special.zig");
const sharding = @import("ops/sharding.zig");
const quant_ops = @import("ops/quant.zig");
const ieee_ops = @import("ops/ieee.zig");
const attention_ops = @import("ops/attention.zig");
const gdn_ops = @import("ops/gdn.zig");

pub const Input = source.input.Attrs;

pub const impls = struct {
    pub const input = source.input;
    pub const literal = source.literal;
    pub const constant = source.constant;
    pub const scalar = source.scalar;
    pub const iota = source.iota;
    pub const arange = source.arange;
    pub const unary = elementwise.unary;
    pub const binary = elementwise.binary;
    pub const binary_const = elementwise.binary_const;
    pub const compare = elementwise.compare;
    pub const select = elementwise.select;
    pub const clamp = elementwise.clamp;
    pub const convert = elementwise.convert;
    pub const transpose = layout.transpose;
    pub const reshape = layout.reshape;
    pub const axes = layout.axes;
    pub const broadcast = layout.broadcast;
    pub const slice = layout.slice;
    pub const concat = layout.concat;
    pub const reverse = layout.reverse;
    pub const pad = layout.pad;
    pub const repeat = layout.repeat;
    pub const reduce = reduce_ops.reduce;
    pub const cumsum = reduce_ops.cumsum;
    pub const argmax = reduce_ops.argmax;
    pub const softmax = reduce_ops.softmax;
    pub const dot = linalg.dot;
    pub const outer = linalg.outer;
    pub const gather = indexing.gather;
    pub const dynamic_slice = indexing.dynamic_slice;
    pub const dynamic_update = indexing.dynamic_update;
    pub const sort = indexing.sort;
    pub const chunk = misc.chunk;
    pub const roll = misc.roll;
    pub const triangular = misc.triangular;
    pub const to_diagonal = misc.to_diagonal;
    pub const cartesian = misc.cartesian;
    pub const group = misc.group;
    pub const reinterpret = misc.reinterpret;
    pub const helpers = misc.helpers;
    pub const float_ops = misc.float_ops;
    pub const softmax_biased = misc.softmax_biased;
    pub const swiglu = misc.swiglu;
    pub const memory = misc.memory;
    pub const reduce_window = window.reduce_window;
    pub const pool = window.pool;
    pub const conv = window.conv;
    pub const scatter = scatter_ops.scatter;
    pub const gather_slices = scatter_ops.gather_slices;
    pub const activation = nn.activation;
    pub const norm = nn.norm;
    pub const rope = nn.rope;
    pub const causal_mask = nn.causal_mask;
    pub const sdpa = nn.sdpa;
    pub const layers = nn.layers;
    pub const resize = nn.resize;
    pub const realimg = nn.realimg;
    pub const param = control.param;
    pub const while_loop = control.while_loop;
    pub const cond = control.cond;
    pub const reduce_n = dsl.reduce_n;
    pub const sort_n = dsl.sort_n;
    pub const rng = special.rng;
    pub const decomp = special.decomp;
    pub const complex = special.complex;
    pub const partition = sharding.partition;
    pub const manual = sharding.manual;
    pub const quant = quant_ops.quant;
    pub const composite = quant_ops.composite;
    pub const ieee = ieee_ops.ieee;
    pub const attention = attention_ops.attention;
    pub const gdn = gdn_ops.gdn;
};

pub const Op = blk: {
    const decls = @typeInfo(impls).@"struct".decls;
    var names: [decls.len][]const u8 = undefined;
    var types: [decls.len]type = undefined;
    var attrs: [decls.len]std.builtin.Type.UnionField.Attributes = undefined;
    for (decls, 0..) |d, i| {
        names[i] = d.name;
        types[i] = @field(impls, d.name).Attrs;
        attrs[i] = .{};
    }
    const Tag = @Enum(u8, .exhaustive, &names, &std.simd.iota(u8, decls.len));
    break :blk @Union(.auto, Tag, &names, &types, &attrs);
};

pub const Family = std.meta.Tag(Op);
pub const family_count = @typeInfo(Family).@"enum".fields.len;

pub const default_weights: [family_count]u32 = blk: {
    var res: [family_count]u32 = undefined;
    for (@typeInfo(Family).@"enum".fields, 0..) |f, i| res[i] = @field(impls, f.name).weight;
    break :blk res;
};

pub fn isArgument(op: Op) bool {
    return op == .input or op == .literal;
}

/// Deliberate bugs injected in the lowering (`--mutate`), to measure what the oracle catches.
pub const Mutation = enum {
    none,
    /// f32 -> bf16 converts truncate instead of rounding to nearest even.
    truncating_convert,
    /// negate returns +0 for both zeros.
    drop_neg_zero,
    /// sqrt returns one ulp above the correctly rounded result.
    sqrt_ulp,
};
pub var mutation: Mutation = .none;

/// The target the program runs on: f32 dots on GPUs use TF32 (`dot_precision = .fast`).
pub var target: zml.Target = .cpu;

pub const Eval = struct {
    allocator: std.mem.Allocator,
    /// During generation, ops reject results the oracle could not compare reliably.
    /// When replaying a program, nothing is rejected.
    strict: bool,
    /// The program being evaluated, for control flow ops that evaluate their regions.
    prog: ?*const ir.Program = null,
    /// Values of the `param` nodes of the region being evaluated.
    bindings: []const Value = &.{},

    pub const Error = error{ OutOfMemory, Reject };

    pub fn alloc(e: *Eval, sh: zml.Shape) error{OutOfMemory}!Value {
        return Value.alloc(e.allocator, sh);
    }

    /// Rejects only during generation.
    pub fn reject(e: *Eval) Error!void {
        if (e.strict) return error.Reject;
    }
};

pub fn propose(fam: Family, g: *gen.Gen) gen.Error!?gen.Proposal {
    switch (fam) {
        inline else => |t| {
            const I = @field(impls, @tagName(t));
            if (comptime !@hasDecl(I, "propose")) return null;
            const saved = g.empty_ok;
            g.empty_ok = @hasDecl(I, "empty_ok");
            defer g.empty_ok = saved;
            return I.propose(g);
        },
    }
}

pub fn eval(e: *Eval, node: *const ir.Node, ins: []const Value) Eval.Error!Value {
    switch (node.op) {
        inline else => |attrs, t| return @field(impls, @tagName(t)).eval(e, attrs, node, ins),
    }
}

pub fn lower(node: *const ir.Node, ins: []const zml.Tensor) zml.Tensor {
    switch (node.op) {
        inline else => |attrs, t| return @field(impls, @tagName(t)).lower(attrs, node, ins),
    }
}

pub fn emit(node: *const ir.Node, w: *std.Io.Writer, ins: []const []const u8) std.Io.Writer.Error!void {
    switch (node.op) {
        inline else => |attrs, t| return @field(impls, @tagName(t)).emit(attrs, node, w, ins),
    }
}
