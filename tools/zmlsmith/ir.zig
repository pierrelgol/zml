//! The program representation shared by the generator, the ZML lowering, the reference
//! interpreter, the reducer and the regression test emitter.
//!
//! A program is a list of nodes in topological order: node `i` only reads nodes `< i`.
//! Every node records the shape (dims, tags and dtype) the generator predicts ZML returns.

const std = @import("std");

const zml = @import("zml");
const Shape = zml.Shape;
const DataType = zml.DataType;

const ops = @import("ops.zig");

pub const max_operands = 8;

/// Tags are compared by pointer at runtime, so every tag the fuzzer uses comes from this
/// pool of interned literals. Names ZML treats specially (`_0`..`_7`, `_`, `last`, ...)
/// are kept out.
/// The last entries are tags ZML itself creates (`stack(.., .coord)`, `bitCast`, `bytes`):
/// they are never picked at random, but the shape oracle must be able to name them.
pub const tag_names = [_][:0]const u8{ "a", "b", "c", "d", "e", "f", "g", "h", "k", "m", "n", "p", "coord", "bitcast", "bytes", "q", "s", "hd", "voc", "sc", "v" };
pub const random_tag_count = 12;
pub const all_tags = [tag_names.len]Shape.Tag{
    Shape.toTag(.a),     Shape.toTag(.b),       Shape.toTag(.c),     Shape.toTag(.d),
    Shape.toTag(.e),     Shape.toTag(.f),       Shape.toTag(.g),     Shape.toTag(.h),
    Shape.toTag(.k),     Shape.toTag(.m),       Shape.toTag(.n),     Shape.toTag(.p),
    Shape.toTag(.coord), Shape.toTag(.bitcast), Shape.toTag(.bytes), Shape.toTag(.q),
    Shape.toTag(.s),     Shape.toTag(.hd),      Shape.toTag(.voc),   Shape.toTag(.sc),
    Shape.toTag(.v),
};
/// Tags the generator picks from.
pub const tag_pool = all_tags[0..random_tag_count].*;

/// A tag reference that can be stored in attributes and serialized.
pub const TagId = enum(u8) {
    a,
    b,
    c,
    d,
    e,
    f,
    g,
    h,
    k,
    m,
    n,
    p,
    coord,
    bitcast,
    bytes,
    q,
    s,
    hd,
    voc,
    sc,
    v,
    none = 255,
    _,

    pub fn tag(self: TagId) Shape.Tag {
        if (self == .none) return Shape.TagUnknown;
        return all_tags[@intFromEnum(self)];
    }

    pub fn of(t: Shape.Tag) TagId {
        for (all_tags, 0..) |p, i| {
            if (p == t) return @enumFromInt(i);
        }
        return .none;
    }

    pub fn name(self: TagId) []const u8 {
        if (self == .none) return "_";
        return tag_names[@intFromEnum(self)];
    }
};

/// A small inline list usable in attributes.
pub fn List(comptime T: type, comptime N: usize) type {
    return struct {
        len: u8 = 0,
        buf: [N]T = undefined,

        const Self = @This();
        pub const Item = T;
        pub const capacity = N;

        pub fn init(items: []const T) Self {
            var res: Self = .{};
            for (items) |it| res.append(it);
            return res;
        }

        pub fn append(self: *Self, item: T) void {
            std.debug.assert(self.len < N);
            self.buf[self.len] = item;
            self.len += 1;
        }

        pub fn slice(self: *const Self) []const T {
            return self.buf[0..self.len];
        }

        pub fn mut(self: *Self) []T {
            return self.buf[0..self.len];
        }
    };
}

pub const Node = struct {
    op: ops.Op,
    inputs: List(u32, max_operands) = .{},
    shape: Shape,
    /// 0 for top-level nodes. Nodes of a control flow region (a while body, an if branch)
    /// carry the id of their region; they are lowered and evaluated by the op owning it.
    scope: u32 = 0,
};

pub const Program = struct {
    seed: u64 = 0,
    /// Compile with a `data` x `model` mesh over all devices, and transfer arguments sharded.
    sharded: bool = false,
    /// In sharded programs, also transfer arguments sharded (see ops/sharding.zig).
    shard_args: bool = false,
    nodes: std.ArrayList(Node) = .empty,
    outputs: std.ArrayList(u32) = .empty,

    pub fn add(self: *Program, allocator: std.mem.Allocator, node: Node) !u32 {
        try self.nodes.append(allocator, node);
        return @intCast(self.nodes.items.len - 1);
    }

    /// Indices of the nodes that become executable arguments, in argument order.
    pub fn argumentNodes(self: *const Program, allocator: std.mem.Allocator) ![]u32 {
        var res: std.ArrayList(u32) = .empty;
        for (self.nodes.items, 0..) |n, i| {
            if (ops.isArgument(n.op)) try res.append(allocator, @intCast(i));
        }
        return res.toOwnedSlice(allocator);
    }

    pub fn clone(self: *const Program, allocator: std.mem.Allocator) !Program {
        var res: Program = .{ .seed = self.seed, .sharded = self.sharded, .shard_args = self.shard_args };
        try res.nodes.appendSlice(allocator, self.nodes.items);
        try res.outputs.appendSlice(allocator, self.outputs.items);
        return res;
    }

    pub fn format(self: *const Program, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("seed {d}\n", .{self.seed});
        if (self.sharded) try w.writeAll("sharded\n");
        if (self.shard_args) try w.writeAll("shard_args\n");
        for (self.nodes.items, 0..) |n, i| {
            try w.print("%{d} = ", .{i});
            try printNode(w, &n);
            try w.writeByte('\n');
        }
        try w.writeAll("return");
        for (self.outputs.items, 0..) |o, i| {
            try w.print("{s} %{d}", .{ if (i == 0) "" else ",", o });
        }
        try w.writeByte('\n');
    }
};

pub fn printShape(w: *std.Io.Writer, sh: Shape) std.Io.Writer.Error!void {
    try w.writeByte('{');
    for (0..sh.rank()) |ax| {
        if (ax > 0) try w.writeByte(',');
        try w.print("{s}={d}", .{ TagId.of(sh.tag(ax)).name(), sh.dim(ax) });
    }
    try w.print("}}:{s}", .{@tagName(sh.dtype())});
}

pub fn printNode(w: *std.Io.Writer, n: *const Node) std.Io.Writer.Error!void {
    switch (n.op) {
        inline else => |attrs, tag| {
            try w.writeAll(@tagName(tag));
            try w.writeByte('(');
            for (n.inputs.slice(), 0..) |in, j| {
                if (j > 0) try w.writeAll(", ");
                try w.print("%{d}", .{in});
            }
            try w.writeByte(')');
            if (@TypeOf(attrs) != void and @typeInfo(@TypeOf(attrs)).@"struct".fields.len > 0) {
                try w.writeAll(" ");
                try printAttr(w, attrs);
            }
        },
    }
    try w.writeAll(" -> ");
    try printShape(w, n.shape);
    if (n.scope != 0) try w.print(" @{d}", .{n.scope});
}

fn printAttr(w: *std.Io.Writer, v: anytype) std.Io.Writer.Error!void {
    const T = @TypeOf(v);
    if (T == []const u8) {
        try w.writeByte('x');
        for (v) |b| try w.print("{x:0>2}", .{b});
        return;
    }
    if (T == TagId) return w.print(".{s}", .{v.name()});
    switch (@typeInfo(T)) {
        .int => try w.print("{d}", .{v}),
        .float => try w.print("{e}", .{v}),
        .bool => try w.writeAll(if (v) "true" else "false"),
        .@"enum" => try w.print(".{s}", .{@tagName(v)}),
        .array => {
            try w.writeByte('[');
            for (v, 0..) |e, i| {
                if (i > 0) try w.writeByte(',');
                try printAttr(w, e);
            }
            try w.writeByte(']');
        },
        .@"struct" => |s| {
            if (@hasDecl(T, "Item") and @hasField(T, "buf")) {
                try w.writeByte('[');
                for (v.slice(), 0..) |e, i| {
                    if (i > 0) try w.writeByte(',');
                    try printAttr(w, e);
                }
                try w.writeByte(']');
                return;
            }
            try w.writeByte('{');
            inline for (s.fields, 0..) |f, i| {
                if (i > 0) try w.writeAll(", ");
                try w.print("{s}=", .{f.name});
                try printAttr(w, @field(v, f.name));
            }
            try w.writeByte('}');
        },
        else => @compileError("unsupported attribute type " ++ @typeName(T)),
    }
}

// ---------------------------------------------------------------------------------------
// Parsing, the exact inverse of `Program.format`.

pub const ParseError = error{ InvalidProgram, OutOfMemory };

pub const Parser = struct {
    src: []const u8,
    pos: usize = 0,
    allocator: std.mem.Allocator,

    fn skipWs(p: *Parser) void {
        while (p.pos < p.src.len and (p.src[p.pos] == ' ' or p.src[p.pos] == '\t' or p.src[p.pos] == '\r')) p.pos += 1;
    }

    fn peek(p: *Parser) u8 {
        p.skipWs();
        return if (p.pos < p.src.len) p.src[p.pos] else 0;
    }

    fn eat(p: *Parser, c: u8) bool {
        if (p.peek() == c) {
            p.pos += 1;
            return true;
        }
        return false;
    }

    fn expect(p: *Parser, c: u8) ParseError!void {
        if (!p.eat(c)) return error.InvalidProgram;
    }

    fn expectStr(p: *Parser, s: []const u8) ParseError!void {
        p.skipWs();
        if (!std.mem.startsWith(u8, p.src[p.pos..], s)) return error.InvalidProgram;
        p.pos += s.len;
    }

    fn ident(p: *Parser) ParseError![]const u8 {
        p.skipWs();
        const start = p.pos;
        while (p.pos < p.src.len and (std.ascii.isAlphanumeric(p.src[p.pos]) or p.src[p.pos] == '_')) p.pos += 1;
        if (start == p.pos) return error.InvalidProgram;
        return p.src[start..p.pos];
    }

    fn number(p: *Parser) ParseError![]const u8 {
        p.skipWs();
        const start = p.pos;
        while (p.pos < p.src.len) : (p.pos += 1) {
            const c = p.src[p.pos];
            if (!(std.ascii.isAlphanumeric(c) or c == '-' or c == '+' or c == '.')) break;
        }
        if (start == p.pos) return error.InvalidProgram;
        return p.src[start..p.pos];
    }

    fn int(p: *Parser, comptime T: type) ParseError!T {
        return std.fmt.parseInt(T, try p.number(), 10) catch error.InvalidProgram;
    }

    fn nodeRef(p: *Parser) ParseError!u32 {
        try p.expect('%');
        return p.int(u32);
    }

    fn endLine(p: *Parser) void {
        while (p.pos < p.src.len and p.src[p.pos] != '\n') p.pos += 1;
        if (p.pos < p.src.len) p.pos += 1;
    }

    fn shape(p: *Parser) ParseError!Shape {
        try p.expect('{');
        var dims: [Shape.MAX_RANK]i64 = undefined;
        var tags: [Shape.MAX_RANK]Shape.Tag = undefined;
        var rank: usize = 0;
        while (!p.eat('}')) {
            if (rank > 0) try p.expect(',');
            const t = try p.ident();
            try p.expect('=');
            dims[rank] = try p.int(i64);
            tags[rank] = if (std.mem.eql(u8, t, "_")) Shape.TagUnknown else (std.meta.stringToEnum(TagId, t) orelse return error.InvalidProgram).tag();
            rank += 1;
        }
        try p.expect(':');
        const dt = std.meta.stringToEnum(DataType, try p.ident()) orelse return error.InvalidProgram;
        return makeShape(dt, dims[0..rank], tags[0..rank]);
    }

    fn attr(p: *Parser, comptime T: type) ParseError!T {
        if (T == []const u8) {
            p.skipWs();
            try p.expect('x');
            const start = p.pos;
            while (p.pos < p.src.len and std.ascii.isHex(p.src[p.pos])) p.pos += 1;
            const hex = p.src[start..p.pos];
            const out = try p.allocator.alloc(u8, hex.len / 2);
            _ = std.fmt.hexToBytes(out, hex) catch return error.InvalidProgram;
            return out;
        }
        if (T == TagId) {
            try p.expect('.');
            const n = try p.ident();
            if (std.mem.eql(u8, n, "_")) return .none;
            return std.meta.stringToEnum(TagId, n) orelse error.InvalidProgram;
        }
        switch (@typeInfo(T)) {
            .int => return p.int(T),
            .float => return std.fmt.parseFloat(T, try p.number()) catch error.InvalidProgram,
            .bool => {
                const n = try p.ident();
                return std.mem.eql(u8, n, "true");
            },
            .@"enum" => {
                try p.expect('.');
                return std.meta.stringToEnum(T, try p.ident()) orelse error.InvalidProgram;
            },
            .array => |a| {
                var res: T = undefined;
                try p.expect('[');
                for (0..a.len) |i| {
                    if (i > 0) try p.expect(',');
                    res[i] = try p.attr(a.child);
                }
                try p.expect(']');
                return res;
            },
            .@"struct" => |s| {
                if (@hasDecl(T, "Item") and @hasField(T, "buf")) {
                    var res: T = .{};
                    try p.expect('[');
                    while (!p.eat(']')) {
                        if (res.len > 0) try p.expect(',');
                        if (res.len >= T.capacity) return error.InvalidProgram;
                        res.append(try p.attr(T.Item));
                    }
                    return res;
                }
                var res: T = undefined;
                try p.expect('{');
                inline for (s.fields, 0..) |f, i| {
                    if (i > 0) try p.expect(',');
                    const n = try p.ident();
                    if (!std.mem.eql(u8, n, f.name)) return error.InvalidProgram;
                    try p.expect('=');
                    @field(res, f.name) = try p.attr(f.type);
                }
                try p.expect('}');
                return res;
            },
            else => @compileError("unsupported attribute type " ++ @typeName(T)),
        }
    }

    fn node(p: *Parser) ParseError!Node {
        const name = try p.ident();
        var n: Node = .{ .op = undefined, .shape = undefined };
        try p.expect('(');
        while (!p.eat(')')) {
            if (n.inputs.len > 0) try p.expect(',');
            if (n.inputs.len >= max_operands) return error.InvalidProgram;
            n.inputs.append(try p.nodeRef());
        }
        const tag = std.meta.stringToEnum(std.meta.Tag(ops.Op), name) orelse return error.InvalidProgram;
        switch (tag) {
            inline else => |t| {
                const A = @FieldType(ops.Op, @tagName(t));
                const attrs: A = if (A == void) {} else if (@typeInfo(A).@"struct".fields.len == 0) .{} else try p.attr(A);
                n.op = @unionInit(ops.Op, @tagName(t), attrs);
            },
        }
        try p.expectStr("->");
        n.shape = try p.shape();
        if (p.eat('@')) n.scope = try p.int(u32);
        return n;
    }
};

pub fn parse(allocator: std.mem.Allocator, src: []const u8) ParseError!Program {
    var p: Parser = .{ .src = src, .allocator = allocator };
    var prog: Program = .{};
    while (p.pos < p.src.len) {
        const c = p.peek();
        if (c == '\n') {
            p.pos += 1;
            continue;
        }
        if (c == '#' or c == 0) {
            p.endLine();
            continue;
        }
        if (c == '%') {
            const id = try p.nodeRef();
            if (id != prog.nodes.items.len) return error.InvalidProgram;
            try p.expect('=');
            const n = try p.node();
            for (n.inputs.slice()) |in| if (in >= id) return error.InvalidProgram;
            try prog.nodes.append(allocator, n);
            p.endLine();
            continue;
        }
        const word = try p.ident();
        if (std.mem.eql(u8, word, "seed")) {
            prog.seed = try p.int(u64);
        } else if (std.mem.eql(u8, word, "sharded")) {
            prog.sharded = true;
        } else if (std.mem.eql(u8, word, "shard_args")) {
            prog.shard_args = true;
        } else if (std.mem.eql(u8, word, "return")) {
            while (p.peek() == '%' or p.peek() == ',') {
                _ = p.eat(',');
                const o = try p.nodeRef();
                if (o >= prog.nodes.items.len) return error.InvalidProgram;
                try prog.outputs.append(allocator, o);
            }
        } else return error.InvalidProgram;
        p.endLine();
    }
    return prog;
}

pub fn makeShape(dt: DataType, dims: []const i64, tags: []const Shape.Tag) Shape {
    var sh: Shape = .init(.{}, dt);
    for (dims, tags) |d, t| sh = sh.appendDim(d, t);
    return sh;
}

test "program roundtrip" {
    const allocator = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var prog: Program = .{ .seed = 42 };
    const sh = makeShape(.f32, &.{ 3, 4 }, &.{ TagId.a.tag(), Shape.TagUnknown });
    _ = try prog.add(a, .{ .op = .{ .input = .{ .seed = 7, .dist = .int_range } }, .shape = sh });
    try prog.outputs.append(a, 0);

    var out: std.Io.Writer.Allocating = .init(a);
    try prog.format(&out.writer);
    const parsed = try parse(a, out.written());
    var out2: std.Io.Writer.Allocating = .init(a);
    try parsed.format(&out2.writer);
    try std.testing.expectEqualStrings(out.written(), out2.written());
}
