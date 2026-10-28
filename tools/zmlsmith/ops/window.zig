//! Windowed ops: ops.reduceWindow, maxPool1d/2d and conv1d/2d, interpreted from the
//! stablehlo reduce_window and convolution specs.

const std = @import("std");

const zml = @import("zml");

const k = @import("../kit.zig");
const Shape = k.Shape;
const DataType = k.DataType;
const Tensor = k.Tensor;
const Value = k.Value;
const Gen = k.Gen;
const Eval = k.Eval;
const Node = k.Node;
const Writer = k.Writer;
const List = k.List;

fn printI64s(w: *Writer, xs: []const i64) Writer.Error!void {
    try w.writeAll("&[_]i64{ ");
    for (xs, 0..) |x, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", x });
    try w.writeAll(" }");
}

fn printPairs(w: *Writer, xs: []const [2]i64) Writer.Error!void {
    try w.writeAll("&[_][2]i64{ ");
    for (xs, 0..) |x, i| try w.print("{s}.{{ {d}, {d} }}", .{ if (i > 0) ", " else "", x[0], x[1] });
    try w.writeAll(" }");
}

/// Per-axis window parameters, for every axis of the operand.
pub const Window = struct {
    dims: List(i64, 8) = .{},
    strides: List(i64, 8) = .{},
    base_dil: List(i64, 8) = .{},
    win_dil: List(i64, 8) = .{},
    pad: List([2]i64, 8) = .{},

    fn outDim(w: Window, ax: usize, d: i64) i64 {
        const dilated = if (d == 0) 0 else (d - 1) * w.base_dil.slice()[ax] + 1;
        const padded = dilated + w.pad.slice()[ax][0] + w.pad.slice()[ax][1];
        const win = (w.dims.slice()[ax] - 1) * w.win_dil.slice()[ax] + 1;
        if (padded < win) return 0;
        return @divFloor(padded - win, w.strides.slice()[ax]) + 1;
    }

    fn outShape(w: Window, sh: Shape) Shape {
        var res = sh;
        for (0..sh.rank()) |ax| res = res.setDim(ax, w.outDim(ax, sh.dim(ax)));
        return res;
    }

    /// Identity window on `rank` axes.
    fn identity(rank: usize) Window {
        var w: Window = .{};
        for (0..rank) |_| {
            w.dims.append(1);
            w.strides.append(1);
            w.base_dil.append(1);
            w.win_dil.append(1);
            w.pad.append(.{ 0, 0 });
        }
        return w;
    }

    /// Randomizes the window on axis `ax` of size `d`. Keeps at least one output element.
    fn randomAxis(w: *Window, g: *Gen, ax: usize, d: i64, allow_base_dil: bool) void {
        w.dims.mut()[ax] = g.intIn(i64, 1, @min(d, 4));
        w.strides.mut()[ax] = g.intIn(i64, 1, 3);
        if (allow_base_dil and g.chance(0.2)) w.base_dil.mut()[ax] = 2;
        if (g.chance(0.25)) w.win_dil.mut()[ax] = 2;
        if (g.chance(0.4)) w.pad.mut()[ax] = .{ g.intIn(i64, 0, 2), g.intIn(i64, 0, 2) };
    }

    /// For an output index, iterates the input positions of the window. `f` is called
    /// with the flat input offset, or null when the position falls in padding or a hole.
    fn forEach(w: Window, in: Shape, out_idx: []const i64, ctx: anytype, comptime f: anytype) void {
        const rank = in.rank();
        const st = Value.strides(in);
        var wsh: Shape = .init(.{}, .i32);
        for (0..rank) |ax| wsh = wsh.appendDim(w.dims.slice()[ax], null);
        var it: Value.Index = .init(wsh);
        while (!it.done) {
            var off: usize = 0;
            var valid = true;
            for (0..rank) |ax| {
                const p = out_idx[ax] * w.strides.slice()[ax] + it.idx[ax] * w.win_dil.slice()[ax] - w.pad.slice()[ax][0];
                const bd = w.base_dil.slice()[ax];
                if (p < 0 or @mod(p, bd) != 0 or @divExact(p, bd) >= in.dim(ax)) {
                    valid = false;
                    break;
                }
                off += @as(usize, @intCast(@divExact(p, bd))) * st[ax];
            }
            f(ctx, if (valid) off else null);
            _ = it.next();
        }
    }
};

pub const reduce_window = struct {
    pub const Kind = enum { sum, max, min };
    pub const Attrs = struct { kind: Kind, win: Window };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const kind = g.rng.enumValue(Kind);
        const x = try g.operand(.{ .class = .numeric, .min_rank = 1, .max_rank = 4 });
        const sh = g.shapeOf(x);
        var w: Window = .identity(sh.rank());
        const n = g.intIn(usize, 1, @min(2, sh.rank()));
        for (0..n) |_| {
            const ax = g.rng.uintLessThan(usize, sh.rank());
            if (sh.dim(ax) == 0) return error.Reject;
            w.randomAxis(g, ax, sh.dim(ax), !(g.prog.sharded and g.cfg.avoid.contains(.sharded_reduce_window)));
        }
        const out = w.outShape(sh);
        if (out.count() == 0) return error.Reject;
        const node: k.Node = .{ .op = .{ .reduce_window = .{ .kind = kind, .win = w } }, .shape = out };
        try g.avoid(.xla_reduce_window_chain, k.gen.paddedWindow(node) and (k.gen.paddedWindow(g.prog.nodes.items[x]) or g.prog.nodes.items[x].op == .cumsum));
        return .{ .op = .{ .reduce_window = .{ .kind = kind, .win = w } }, .inputs = try k.dupeIds(g, &.{x}), .shape = out };
    }

    const Acc = struct { x: Value, kind: Kind, i: i128 = 0, f: f64 = 0, abs: f64 = 0, n: usize = 0, integral: bool = true };

    fn step(acc: *Acc, off: ?usize) void {
        // Padding and holes contribute the init value: 0 for sum, the dtype min/max otherwise.
        const o = off orelse return;
        acc.n += 1;
        if (Value.isFloat(acc.x.dtype())) {
            const a = acc.x.getF(o);
            acc.abs += @abs(a);
            acc.integral = acc.integral and a == @round(a);
            acc.f = switch (acc.kind) {
                .sum => acc.f + a,
                .max => @max(acc.f, a),
                .min => @min(acc.f, a),
            };
        } else {
            const a = acc.x.getI(o);
            acc.i = switch (acc.kind) {
                .sum => Value.wrapInt(acc.x.dtype(), acc.i + a),
                .max => @max(acc.i, a),
                .min => @min(acc.i, a),
            };
        }
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const dt = x.dtype();
        var v = try e.alloc(node.shape);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        var err: f64 = 0;
        const errs = try v.trackErrs(e.allocator);
        while (!it.done) : (i += 1) {
            var acc: Acc = .{ .x = x, .kind = attrs.kind };
            if (Value.isFloat(dt)) {
                acc.f = switch (attrs.kind) {
                    .sum => 0,
                    .max => -std.math.inf(f64),
                    .min => std.math.inf(f64),
                };
            } else {
                acc.i = switch (attrs.kind) {
                    .sum => 0,
                    .max => Value.minInt(dt),
                    .min => Value.maxInt(dt),
                };
            }
            attrs.win.forEach(x.shape, it.idx[0..node.shape.rank()], &acc, step);
            if (Value.isFloat(dt)) {
                var lane_err = x.err * @as(f64, @floatFromInt(@max(acc.n, 1)));
                if (attrs.kind == .sum and !(x.err == 0 and acc.integral and acc.abs < std.math.ldexp(@as(f64, 1), Value.precisionBits(dt))))
                    lane_err += @as(f64, @floatFromInt(acc.n + 1)) * Value.eps(dt) * acc.abs;
                if (!k.storeF(v, i, acc.f)) lane_err += k.roundErr(dt, @abs(acc.f));
                if (acc.f == 0) lane_err = @max(lane_err, Value.any_zero_sign);
                errs[i] = lane_err;
                err = @max(err, lane_err);
            } else v.setI(i, acc.i);
            _ = it.next();
        }
        v.err = err;
        v.rounded = Value.isFloat(dt);
        return v;
    }

    fn sumFn(a: zml.ops.ReduceArgs) [1]Tensor {
        return .{a.left.add(a.right)};
    }
    fn maxFn(a: zml.ops.ReduceArgs) [1]Tensor {
        return .{a.left.maximum(a.right)};
    }
    fn minFn(a: zml.ops.ReduceArgs) [1]Tensor {
        return .{a.left.minimum(a.right)};
    }

    fn opts(w: *const Window) zml.ops.ReduceWindowOpts {
        return .{
            .window_dimensions = w.dims.slice(),
            .window_strides = w.strides.slice(),
            .base_dilations = w.base_dil.slice(),
            .window_dilations = w.win_dil.slice(),
            .padding = w.pad.slice(),
        };
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const dt = x.dtype();
        const init: Tensor = switch (attrs.kind) {
            .sum => .constant(dt.zero()),
            .max => .constant(dt.minValue()),
            .min => .constant(dt.maxValue()),
        };
        const f: *const zml.ops.ReduceWindowFn(1) = switch (attrs.kind) {
            .sum => &sumFn,
            .max => &maxFn,
            .min => &minFn,
        };
        return zml.ops.reduceWindow(1, .{x}, .{init}, opts(&attrs.win), f)[0];
    }

    pub fn emit(attrs: Attrs, node: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const dt = @tagName(node.shape.dtype());
        const init = switch (attrs.kind) {
            .sum => "zero",
            .max => "minValue",
            .min => "maxValue",
        };
        const op = switch (attrs.kind) {
            .sum => "add",
            .max => "maximum",
            .min => "minimum",
        };
        try w.print("zml.ops.reduceWindow(1, .{{{s}}}, .{{zml.Tensor.constant(zml.DataType.{s}(.{s}))}}, .{{ .window_dimensions = ", .{ ins[0], init, dt });
        try printI64s(w, attrs.win.dims.slice());
        try w.writeAll(", .window_strides = ");
        try printI64s(w, attrs.win.strides.slice());
        try w.writeAll(", .base_dilations = ");
        try printI64s(w, attrs.win.base_dil.slice());
        try w.writeAll(", .window_dilations = ");
        try printI64s(w, attrs.win.win_dil.slice());
        try w.writeAll(", .padding = ");
        try printPairs(w, attrs.win.pad.slice());
        try w.print(" }}, &struct {{\n                fn f(a: zml.ops.ReduceArgs) [1]zml.Tensor {{\n                    return .{{a.left.{s}(a.right)}};\n                }}\n            }}.f)[0]", .{op});
    }
};

// -----------------------------------------------------------------------------

pub const pool = struct {
    pub const Attrs = struct { two_d: bool, indices: bool, win: Window, default_strides: bool };
    pub const weight = 3;

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const two_d = g.chance(0.4);
        const x = try g.operand(.{ .class = .numeric, .min_rank = if (two_d) 3 else 1, .max_rank = if (two_d) 4 else 4 });
        const sh = g.shapeOf(x);
        var w: Window = .identity(sh.rank());
        const last = sh.rank() - 1;
        const axes_: []const usize = if (two_d) &.{ last - 1, last } else &.{last};
        for (axes_) |ax| {
            if (sh.dim(ax) == 0) return error.Reject;
            w.randomAxis(g, ax, sh.dim(ax), !(g.prog.sharded and g.cfg.avoid.contains(.sharded_reduce_window)));
        }
        const default_strides = g.chance(0.3);
        if (default_strides) for (axes_) |ax| {
            w.strides.mut()[ax] = w.dims.slice()[ax];
        };
        // maxPool2d indices are an iota along the last axis only: the column of the max.
        const indices = g.chance(0.5);
        // The bug needs the window max to equal the dtype minimum: unsigned zeros or bool.
        try g.avoid(.pool_min_index, indices and (k.Value.isInt(sh.dtype()) and !sh.dtype().isSignedInt()));
        const out = w.outShape(sh);
        if (out.count() == 0) return error.Reject;
        return .{ .op = .{ .pool = .{ .two_d = two_d, .indices = indices, .win = w, .default_strides = default_strides } }, .inputs = try k.dupeIds(g, &.{x}), .shape = if (indices) out.withDtype(.i32) else out };
    }

    const Acc = struct { x: Value, float: bool, best: ?usize = null, last_ax_stride: usize, any_pad: bool = false, indices: bool = false, d_last: usize = 1 };

    fn step(acc: *Acc, off: ?usize) void {
        const o = off orelse {
            acc.any_pad = true;
            return;
        };
        if (acc.best) |b| {
            const gt = if (acc.float) acc.x.getF(o) > acc.x.getF(b) else acc.x.getI(o) > acc.x.getI(b);
            const eq = if (acc.float) acc.x.getF(o) == acc.x.getF(b) else acc.x.getI(o) == acc.x.getI(b);
            // Documented: the first maximum wins, i.e. the smallest index; in 2d the index
            // is the column, so ties across rows keep the smallest column.
            if (gt or (acc.indices and eq and o % acc.d_last < b % acc.d_last)) acc.best = o;
        } else acc.best = o;
    }

    pub fn eval(e: *Eval, attrs: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        var v = try e.alloc(node.shape);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        const float = Value.isFloat(x.dtype());
        const last = x.shape.rank() - 1;
        const d_last: usize = @intCast(x.shape.dim(last));
        while (!it.done) : (i += 1) {
            var acc: Acc = .{ .x = x, .float = float, .last_ax_stride = 1, .indices = attrs.indices, .d_last = d_last };
            attrs.win.forEach(x.shape, it.idx[0..node.shape.rank()], &acc, step);
            const b = acc.best orelse {
                // An all padding window: the init values, the dtype minimum and index 0.
                try e.reject();
                _ = it.next();
                continue;
            };
            if (acc.any_pad) {
                // Padding holds the dtype minimum, which only matters if it is the max.
                const m = if (float) x.getF(b) else @as(f64, @floatFromInt(x.getI(b)));
                const lo = if (float) -std.math.inf(f64) else @as(f64, @floatFromInt(Value.minInt(x.dtype())));
                if (m == lo) try e.reject();
            }
            if (attrs.indices) {
                if (float and x.err > 0) try e.reject();
                v.setI(i, @intCast(b % d_last));
            } else v.copyElem(i, x, b);
            _ = it.next();
        }
        if (!attrs.indices) v.err = x.err;
        return v;
    }

    pub fn lower(attrs: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const x = ins[0];
        const w = attrs.win;
        const last = x.rank() - 1;
        if (!attrs.two_d) {
            const r = x.maxPool1d(.{
                .window_dimensions = w.dims.slice()[last],
                .window_strides = if (attrs.default_strides) null else w.strides.slice()[last],
                .base_dilations = w.base_dil.slice()[last],
                .window_dilations = w.win_dil.slice()[last],
                .padding = w.pad.slice()[last],
            });
            return if (attrs.indices) r.indices else r.values;
        }
        const r = x.maxPool2d(.{
            .window_dimensions = .{ w.dims.slice()[last - 1], w.dims.slice()[last] },
            .window_strides = if (attrs.default_strides) null else .{ w.strides.slice()[last - 1], w.strides.slice()[last] },
            .base_dilations = .{ w.base_dil.slice()[last - 1], w.base_dil.slice()[last] },
            .window_dilations = .{ w.win_dil.slice()[last - 1], w.win_dil.slice()[last] },
            .padding = .{ w.pad.slice()[last - 1], w.pad.slice()[last] },
        });
        return if (attrs.indices) r.indices else r.values;
    }

    pub fn emit(attrs: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const win = attrs.win;
        const last = win.dims.len - 1;
        if (!attrs.two_d) {
            try w.print("{s}.maxPool1d(.{{ .window_dimensions = {d}, .window_strides = ", .{ ins[0], win.dims.slice()[last] });
            if (attrs.default_strides) try w.writeAll("null") else try w.print("{d}", .{win.strides.slice()[last]});
            return w.print(", .base_dilations = {d}, .window_dilations = {d}, .padding = .{{ {d}, {d} }} }}).{s}", .{ win.base_dil.slice()[last], win.win_dil.slice()[last], win.pad.slice()[last][0], win.pad.slice()[last][1], if (attrs.indices) "indices" else "values" });
        }
        const l = last - 1;
        try w.print("{s}.maxPool2d(.{{ .window_dimensions = .{{ {d}, {d} }}, .window_strides = ", .{ ins[0], win.dims.slice()[l], win.dims.slice()[last] });
        if (attrs.default_strides) try w.writeAll("null") else try w.print(".{{ {d}, {d} }}", .{ win.strides.slice()[l], win.strides.slice()[last] });
        try w.print(", .base_dilations = .{{ {d}, {d} }}, .window_dilations = .{{ {d}, {d} }}, .padding = .{{ .{{ {d}, {d} }}, .{{ {d}, {d} }} }} }}).{s}", .{
            win.base_dil.slice()[l],                    win.base_dil.slice()[last], win.win_dil.slice()[l],   win.win_dil.slice()[last],
            win.pad.slice()[l][0],                      win.pad.slice()[l][1],      win.pad.slice()[last][0], win.pad.slice()[last][1],
            if (attrs.indices) "indices" else "values",
        });
    }
};

// -----------------------------------------------------------------------------

pub const conv = struct {
    /// Axis roles for one of input, kernel or output: [batch-or-out-feature, feature, spatial...].
    pub const Layout = List(u8, 4);
    pub const Attrs = struct {
        spatial: u8,
        in_layout: Layout,
        k_layout: Layout,
        out_layout: Layout,
        strides: List(i64, 2),
        pad: List(i64, 4),
        lhs_dil: List(i64, 2),
        rhs_dil: List(i64, 2),
        reversal: List(bool, 2),
        groups: i64,
        /// batch_group_count: the input batch is split in contiguous groups, output feature
        /// o reads group o / (cout / batch_groups). Exclusive with groups > 1.
        batch_groups: i64 = 1,
    };
    pub const weight = 3;

    fn randLayout(g: *Gen, rank: usize, default: bool) Layout {
        var l: Layout = .{};
        for (0..rank) |i| l.append(@intCast(i));
        if (!default) g.rng.shuffle(u8, l.mut());
        return l;
    }

    pub fn propose(g: *Gen) k.GenError!?k.Proposal {
        const ns: u8 = if (g.chance(0.5)) 1 else 2;
        const rank = ns + 2;
        const dt = g.randDtype(.numeric) orelse return null;
        const default = g.chance(0.5);
        var a: Attrs = .{
            .spatial = ns,
            .in_layout = randLayout(g, rank, default),
            .k_layout = randLayout(g, rank, default),
            .out_layout = randLayout(g, rank, default),
            .strides = .{},
            .pad = .{},
            .lhs_dil = .{},
            .rhs_dil = .{},
            .reversal = .{},
            .groups = if (g.chance(0.3)) g.intIn(i64, 2, 3) else 1,
        };
        if (a.groups == 1 and g.chance(0.2)) a.batch_groups = g.intIn(i64, 2, 3);
        const batch = a.batch_groups * g.intIn(i64, 1, 2);
        const cin = a.groups * g.intIn(i64, 1, 2);
        const cout = a.groups * a.batch_groups * g.intIn(i64, 1, 2);
        // Logical dims [b, c, s...] then placed by the layouts.
        var in_dims: [4]i64 = .{ batch, cin, 0, 0 };
        var k_dims: [4]i64 = .{ cout, @divExact(cin, a.groups), 0, 0 };
        for (0..ns) |s| {
            in_dims[2 + s] = g.intIn(i64, 1, 6);
            k_dims[2 + s] = g.intIn(i64, 1, @min(3, in_dims[2 + s]));
            a.strides.append(g.intIn(i64, 1, 2));
            // Negative padding crops the input.
            const neg_ok = default or !g.cfg.avoid.contains(.xla_conv_neg_pad_layout);
            a.pad.append(if (neg_ok and g.chance(0.15)) -1 else g.intIn(i64, 0, 1));
            a.pad.append(if (neg_ok and g.chance(0.15)) -1 else g.intIn(i64, 0, 1));
            a.lhs_dil.append(if (g.chance(0.2)) 2 else 1);
            a.rhs_dil.append(if (g.chance(0.2)) 2 else 1);
            a.reversal.append(g.chance(0.2));
        }
        {
            var rev = false;
            var strided = false;
            var padded = false;
            for (a.reversal.slice()) |r| rev = rev or r;
            for (a.strides.slice()) |st| strided = strided or st > 1;
            for (a.pad.slice()) |pd| padded = padded or pd > 0;
            try g.avoid(.xla_conv_reversal, rev and strided and padded);
        }
        var in_sh: Shape = .init(.{}, dt);
        var k_sh: Shape = .init(.{}, dt);
        var in_phys: [4]i64 = undefined;
        var k_phys: [4]i64 = undefined;
        for (0..rank) |r| {
            in_phys[a.in_layout.slice()[r]] = in_dims[r];
            k_phys[a.k_layout.slice()[r]] = k_dims[r];
        }
        var tags: [4]Shape.Tag = undefined;
        g.randTags(&tags);
        const tagged = g.chance(0.6);
        for (0..rank) |p| {
            in_sh = in_sh.appendDim(in_phys[p], if (tagged) tags[p] else null);
            k_sh = k_sh.appendDim(k_phys[p], null);
        }
        const x = try g.operandLike(in_sh, true);
        const kern = try g.freshInput(k_sh);
        // Output: as computed by ZML, the input shape with each axis replaced by the dim of
        // the output role at that position. Tags stay those of the input.
        var out = g.shapeOf(x);
        var out_dims: [4]i64 = .{ @divExact(batch, a.batch_groups), cout, 0, 0 };
        for (0..ns) |s| {
            const d = in_dims[2 + s];
            const dil = if (d == 0) 0 else (d - 1) * a.lhs_dil.slice()[s] + 1;
            const padded = dil + a.pad.slice()[2 * s] + a.pad.slice()[2 * s + 1];
            const wd = (k_dims[2 + s] - 1) * a.rhs_dil.slice()[s] + 1;
            if (padded < wd or padded <= 0) return error.Reject;
            out_dims[2 + s] = @divFloor(padded - wd, a.strides.slice()[s]) + 1;
        }
        for (0..rank) |r| out = out.setDim(a.out_layout.slice()[r], out_dims[r]);
        return .{ .op = .{ .conv = a }, .inputs = try k.dupeIds(g, &.{ x, kern }), .shape = out };
    }

    pub fn eval(e: *Eval, a: Attrs, node: *const Node, ins: []const Value) Eval.Error!Value {
        const x = ins[0];
        const kern = ins[1];
        const ns = a.spatial;
        const rank = ns + 2;
        const dt = node.shape.dtype();
        const float = Value.isFloat(dt);
        var v = try e.alloc(node.shape);
        const xs = Value.strides(x.shape);
        const ks = Value.strides(kern.shape);
        const il = a.in_layout.slice();
        const kl = a.k_layout.slice();
        const ol = a.out_layout.slice();
        const cin_g: usize = @intCast(kern.shape.dim(kl[1]));
        const cout: usize = @intCast(kern.shape.dim(kl[0]));
        const cout_g = cout / @as(usize, @intCast(a.groups));
        const cout_bg = cout / @as(usize, @intCast(a.batch_groups));
        const batch_per_group = @divExact(x.shape.dim(il[0]), a.batch_groups);
        var it: Value.Index = .init(node.shape);
        var i: usize = 0;
        var err: f64 = 0;
        const conv_errs = try v.trackErrs(e.allocator);
        while (!it.done) : (i += 1) {
            const idx = it.idx[0..rank];
            const b = idx[ol[0]];
            const o: usize = @intCast(idx[ol[1]]);
            const grp = o / cout_g;
            var acc_i: i128 = 0;
            var acc_f: f64 = 0;
            var abs: f64 = 0;
            var in_term: f64 = 0;
            var integral = true;
            var n: usize = 0;
            var wsh: Shape = .init(.{}, .i32);
            for (0..ns) |s| wsh = wsh.appendDim(kern.shape.dim(kl[2 + s]), null);
            for (0..cin_g) |ci| {
                var wit: Value.Index = .init(wsh);
                while (!wit.done) {
                    var xi: [4]i64 = undefined;
                    var ki: [4]i64 = undefined;
                    xi[il[0]] = @as(i64, @intCast(o / cout_bg)) * batch_per_group + b;
                    xi[il[1]] = @intCast(grp * cin_g + ci);
                    ki[kl[0]] = @intCast(o);
                    ki[kl[1]] = @intCast(ci);
                    var valid = true;
                    for (0..ns) |s| {
                        const kd = kern.shape.dim(kl[2 + s]);
                        const kw = wit.idx[s];
                        ki[kl[2 + s]] = if (a.reversal.slice()[s]) kd - 1 - kw else kw;
                        const p = idx[ol[2 + s]] * a.strides.slice()[s] + kw * a.rhs_dil.slice()[s] - a.pad.slice()[2 * s];
                        const ld = a.lhs_dil.slice()[s];
                        if (p < 0 or @mod(p, ld) != 0 or @divExact(p, ld) >= x.shape.dim(il[2 + s])) {
                            valid = false;
                        } else xi[il[2 + s]] = @divExact(p, ld);
                    }
                    if (valid) {
                        const xo = Value.offset(xs, xi[0..rank]);
                        const ko = Value.offset(ks, ki[0..rank]);
                        n += 1;
                        if (float) {
                            const xv = x.getF(xo);
                            const kv = kern.getF(ko);
                            const p = xv * kv;
                            in_term += @abs(xv) * kern.inErr(ko) + @abs(kv) * x.inErr(xo) + x.inErr(xo) * kern.inErr(ko);
                            acc_f += p;
                            abs += @abs(p);
                            integral = integral and p == @round(p);
                        } else acc_i = Value.wrapInt(dt, acc_i + x.getI(xo) *% kern.getI(ko));
                    }
                    _ = wit.next();
                }
            }
            if (float) {
                var lane_err = in_term;
                if (!(in_term == 0 and integral and abs < std.math.ldexp(@as(f64, 1), Value.precisionBits(dt))))
                    lane_err += @as(f64, @floatFromInt(n + 1)) * Value.eps(dt) * abs;
                if (!k.storeF(v, i, acc_f)) lane_err += k.roundErr(dt, @abs(acc_f));
                if (acc_f == 0) lane_err = @max(lane_err, Value.any_zero_sign);
                conv_errs[i] = lane_err;
                err = @max(err, lane_err);
            } else v.setI(i, acc_i);
            _ = it.next();
        }
        v.err = err;
        v.rounded = float;
        return v;
    }

    pub fn lower(a: Attrs, _: *const Node, ins: []const Tensor) Tensor {
        const il = a.in_layout.slice();
        const kl = a.k_layout.slice();
        const ol = a.out_layout.slice();
        if (a.spatial == 1) {
            return ins[0].conv1d(ins[1], .{
                .window_strides = a.strides.slice()[0],
                .padding = a.pad.slice(),
                .lhs_dilation = a.lhs_dil.slice()[0],
                .rhs_dilation = a.rhs_dil.slice()[0],
                .window_reversal = a.reversal.slice()[0],
                .input_batch_dimension = il[0],
                .input_feature_dimension = il[1],
                .input_spatial_dimensions = il[2],
                .kernel_output_feature_dimension = kl[0],
                .kernel_input_feature_dimension = kl[1],
                .kernel_spatial_dimensions = kl[2],
                .output_batch_dimension = ol[0],
                .output_feature_dimension = ol[1],
                .output_spatial_dimensions = ol[2],
                .feature_group_count = a.groups,
                .batch_group_count = a.batch_groups,
            });
        }
        const S = struct {
            var in_sp: [2]i64 = undefined;
            var k_sp: [2]i64 = undefined;
            var o_sp: [2]i64 = undefined;
        };
        for (0..2) |s| {
            S.in_sp[s] = il[2 + s];
            S.k_sp[s] = kl[2 + s];
            S.o_sp[s] = ol[2 + s];
        }
        return ins[0].conv2d(ins[1], .{
            .window_strides = a.strides.slice(),
            .padding = a.pad.slice(),
            .lhs_dilation = a.lhs_dil.slice(),
            .rhs_dilation = a.rhs_dil.slice(),
            .window_reversal = a.reversal.slice(),
            .input_batch_dimension = il[0],
            .input_feature_dimension = il[1],
            .input_spatial_dimensions = &S.in_sp,
            .kernel_output_feature_dimension = kl[0],
            .kernel_input_feature_dimension = kl[1],
            .kernel_spatial_dimensions = &S.k_sp,
            .output_batch_dimension = ol[0],
            .output_feature_dimension = ol[1],
            .output_spatial_dimensions = &S.o_sp,
            .feature_group_count = a.groups,
            .batch_group_count = a.batch_groups,
        });
    }

    pub fn emit(a: Attrs, _: *const Node, w: *Writer, ins: []const []const u8) Writer.Error!void {
        const il = a.in_layout.slice();
        const kl = a.k_layout.slice();
        const ol = a.out_layout.slice();
        const one = a.spatial == 1;
        try w.print("{s}.conv{d}d({s}, .{{ .window_strides = ", .{ ins[0], a.spatial, ins[1] });
        if (one) try w.print("{d}", .{a.strides.slice()[0]}) else try printI64s(w, a.strides.slice());
        try w.writeAll(", .padding = ");
        try printI64s(w, a.pad.slice());
        try w.writeAll(", .lhs_dilation = ");
        if (one) try w.print("{d}", .{a.lhs_dil.slice()[0]}) else try printI64s(w, a.lhs_dil.slice());
        try w.writeAll(", .rhs_dilation = ");
        if (one) try w.print("{d}", .{a.rhs_dil.slice()[0]}) else try printI64s(w, a.rhs_dil.slice());
        try w.writeAll(", .window_reversal = ");
        if (one) try w.print("{}", .{a.reversal.slice()[0]}) else try w.print("&.{{ {}, {} }}", .{ a.reversal.slice()[0], a.reversal.slice()[1] });
        try w.print(", .input_batch_dimension = {d}, .input_feature_dimension = {d}, .input_spatial_dimensions = ", .{ il[0], il[1] });
        try printSpatial(w, one, il[2..]);
        try w.print(", .kernel_output_feature_dimension = {d}, .kernel_input_feature_dimension = {d}, .kernel_spatial_dimensions = ", .{ kl[0], kl[1] });
        try printSpatial(w, one, kl[2..]);
        try w.print(", .output_batch_dimension = {d}, .output_feature_dimension = {d}, .output_spatial_dimensions = ", .{ ol[0], ol[1] });
        try printSpatial(w, one, ol[2..]);
        try w.print(", .feature_group_count = {d}, .batch_group_count = {d} }})", .{ a.groups, a.batch_groups });
    }

    fn printSpatial(w: *Writer, one: bool, xs: []const u8) Writer.Error!void {
        if (one) return w.print("{d}", .{xs[0]});
        try w.print("&.{{ {d}, {d} }}", .{ xs[0], xs[1] });
    }
};
