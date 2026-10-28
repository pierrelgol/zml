//! A host tensor used by the reference interpreter: contiguous row-major bytes in the
//! same layout PJRT uses for its buffers, plus an error bound for float results.

const std = @import("std");

const zml = @import("zml");
const Shape = zml.Shape;
const DataType = zml.DataType;

const Value = @This();

shape: Shape,
bytes: []align(16) u8,
/// Absolute error bound against an ideal device. Zero means the device must produce
/// exactly these values: integers, or floats that never went through a rounding step.
err: f64 = 0,
/// Holds infinities on purpose (attention masks): compared exactly, and only consumed by
/// ops that ask for it.
allow_inf: bool = false,
/// Per element bounds on the error of the returned values, when the producer tracks them;
/// `err` is then their max. Zero means bit exact, sign of zero included. Producers that
/// don't track them leave null and the comparator falls back to `err` with some slack.
errs: ?[]f64 = null,
/// The producer rounded some results to the dtype. On CPU, bf16 and f16 ops run in f32
/// and XLA may skip the intermediate rounding, so consumers allow half an ulp more.
rounded: bool = false,
/// Some results are rounded products. XLA CPU contracts a product feeding an add or a
/// subtract into a fused multiply-add, which then sees the unrounded product: half an ulp
/// more for those consumers (binary add and sub).
contractible: bool = false,

pub fn alloc(allocator: std.mem.Allocator, shape: Shape) !Value {
    const bytes = try allocator.alignedAlloc(u8, .@"16", shape.byteSize());
    @memset(bytes, 0);
    return .{ .shape = shape, .bytes = bytes };
}

pub fn dupe(self: Value, allocator: std.mem.Allocator) !Value {
    const bytes = try allocator.alignedAlloc(u8, .@"16", self.bytes.len);
    @memcpy(bytes, self.bytes);
    // Per element bounds aren't copied: callers usually change the data. Producers that
    // keep values unchanged and want the tight bound set it explicitly.
    return .{ .shape = self.shape, .bytes = bytes, .err = self.err, .allow_inf = self.allow_inf };
}

/// A per element bound that accepts both signs of an exact zero result, and nothing else:
/// for ops whose zero sign isn't specified (maximum and minimum of +0 and -0).
pub const any_zero_sign: f64 = std.math.floatTrueMin(f64);

/// Allocates per element error bounds, all zero.
pub fn trackErrs(self: *Value, allocator: std.mem.Allocator) ![]f64 {
    const e = try allocator.alloc(f64, self.count());
    @memset(e, 0);
    self.errs = e;
    return e;
}

/// Sets `err` to the max of the per element bounds.
pub fn finishErrs(self: *Value) void {
    var m: f64 = 0;
    for (self.errs.?) |x| m = @max(m, x);
    self.err = m;
}

/// The error bound of element `i` as returned.
pub fn errAt(self: Value, i: usize) f64 {
    if (self.errs) |e| return e[i];
    return self.err;
}

/// The error bound of element `i` as seen by a consumer: with excess precision, the value
/// before rounding to bf16/f16.
pub fn inErr(self: Value, i: usize) f64 {
    var e = self.errAt(i);
    if (self.errs != null and self.rounded and (self.dtype() == .bf16 or self.dtype() == .f16)) e += 0.5 * ulp(self.dtype(), self.getF(i));
    return e;
}

/// The spacing of the dtype's values at `x` (subnormal spacing below the normal range).
pub fn ulp(dt: DataType, x: f64) f64 {
    const p: i32 = precisionBits(dt);
    const min_exp: i32 = switch (dt) {
        .f16 => -14,
        .bf16, .f32 => -126,
        .f64 => -1022,
        .f4e2m1, .f8e3m4, .f8e4m3, .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .f8e8m0 => 1 - MiniFloat.of(dt).bias,
        else => -126,
    };
    const a = @abs(x);
    const e: i32 = if (a == 0 or !std.math.isFinite(a)) min_exp else @max(min_exp, std.math.frexp(a).exponent - 1);
    return std.math.ldexp(@as(f64, 1), e - (p - 1));
}

pub fn count(self: Value) usize {
    return self.shape.count();
}

pub fn dtype(self: Value) DataType {
    return self.shape.dtype();
}

pub fn slice(self: Value) zml.Slice {
    return .init(self.shape, self.bytes);
}

/// The dtypes the interpreter knows how to compute with.
pub fn supported(dt: DataType) bool {
    return switch (dt) {
        .c64, .c128 => false,
        else => true,
    };
}

/// Sub-byte integers, f8 and f4 floats: only converted and moved around, ops don't
/// compute with them.
pub fn isStorage(dt: DataType) bool {
    return switch (dt) {
        .i2, .i4, .u2, .u4, .f4e2m1, .f8e3m4, .f8e4m3, .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .f8e8m0 => true,
        else => false,
    };
}

pub fn isFloat(dt: DataType) bool {
    return switch (dt) {
        .f16, .bf16, .f32, .f64, .f4e2m1, .f8e3m4, .f8e4m3, .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .f8e8m0 => true,
        else => false,
    };
}

pub fn isInt(dt: DataType) bool {
    return switch (dt) {
        .i2, .i4, .i8, .i16, .i32, .i64, .u2, .u4, .u8, .u16, .u32, .u64 => true,
        else => false,
    };
}

/// Bits per element for packed sub-byte dtypes, 0 otherwise.
fn subByteBits(dt: DataType) u4 {
    return switch (dt) {
        .i2, .u2 => 2,
        .i4, .u4, .f4e2m1 => 4,
        else => 0,
    };
}

/// Number of explicit mantissa bits plus one: integers up to 2^p are exact.
pub fn precisionBits(dt: DataType) u7 {
    return switch (dt) {
        .f8e8m0 => 1,
        .f4e2m1, .f8e5m2, .f8e5m2fnuz => 2 + @as(u7, @intFromBool(dt != .f4e2m1)),
        .f8e4m3, .f8e4m3fn, .f8e4m3fnuz, .f8e4m3b11fnuz => 4,
        .f8e3m4 => 5,
        .bf16 => 8,
        .f16 => 11,
        .f32 => 24,
        .f64 => 53,
        else => 0,
    };
}

/// Machine epsilon of a float dtype.
pub fn eps(dt: DataType) f64 {
    return std.math.ldexp(@as(f64, 1), -@as(i32, precisionBits(dt)) + 1);
}

/// Largest magnitude the generator lets float values reach, well below overflow.
pub fn floatLimit(dt: DataType) f64 {
    return switch (dt) {
        .f4e2m1 => 6,
        .f8e3m4 => 15,
        .f8e4m3b11fnuz => 30,
        .f8e4m3, .f8e4m3fn, .f8e4m3fnuz => 240,
        .f8e5m2, .f8e5m2fnuz => 5.0e4,
        .f8e8m0 => 1.0e6,
        .f16 => 1.0e4,
        .bf16, .f32, .f64 => 1.0e6,
        else => unreachable,
    };
}

pub fn bitWidth(dt: DataType) u7 {
    return switch (dt) {
        .bool => 1,
        .i2, .u2 => 2,
        .i4, .u4 => 4,
        .i8, .u8 => 8,
        .i16, .u16 => 16,
        .i32, .u32 => 32,
        .i64, .u64 => 64,
        else => unreachable,
    };
}

pub fn minInt(dt: DataType) i128 {
    return switch (dt) {
        .bool, .u2, .u4 => 0,
        .i2 => -2,
        .i4 => -8,
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |t| std.math.minInt(t.toZigType()),
        else => unreachable,
    };
}

pub fn maxInt(dt: DataType) i128 {
    return switch (dt) {
        .bool => 1,
        .i2 => 1,
        .u2 => 3,
        .i4 => 7,
        .u4 => 15,
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |t| std.math.maxInt(t.toZigType()),
        else => unreachable,
    };
}

/// Wraps an integer into the two's complement range of `dt`.
pub fn wrapInt(dt: DataType, x: i128) i128 {
    return switch (dt) {
        .bool => @intFromBool(x != 0),
        .i2, .i4, .u2, .u4 => {
            const bits = bitWidth(dt);
            const mask = (@as(i128, 1) << bits) - 1;
            const u = x & mask;
            if (dt.isSignedInt() and u >= (@as(i128, 1) << (bits - 1))) return u - (@as(i128, 1) << bits);
            return u;
        },
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |t| {
            const T = t.toZigType();
            const U = std.meta.Int(.unsigned, @bitSizeOf(T));
            const bits: U = @truncate(@as(u128, @bitCast(x)));
            return @as(T, @bitCast(bits));
        },
        else => unreachable,
    };
}

/// Raw bits of packed element `i` (LSB first in each byte, like PJRT and bitcast_convert).
fn getBits(self: Value, i: usize) u8 {
    const bits = subByteBits(self.dtype());
    const bit = i * bits;
    return @truncate((self.bytes[bit / 8] >> @intCast(bit % 8)) & ((@as(u16, 1) << bits) - 1));
}

fn setBits(self: Value, i: usize, v: u8) void {
    const bits = subByteBits(self.dtype());
    const bit = i * bits;
    const mask: u8 = @intCast(((@as(u16, 1) << bits) - 1) << @intCast(bit % 8));
    self.bytes[bit / 8] = (self.bytes[bit / 8] & ~mask) | ((v << @intCast(bit % 8)) & mask);
}

pub fn getF(self: Value, i: usize) f64 {
    return switch (self.dtype()) {
        .f4e2m1 => MiniFloat.of(.f4e2m1).decode(self.getBits(i)),
        .f8e3m4, .f8e4m3, .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .f8e8m0 => MiniFloat.of(self.dtype()).decode(self.bytes[i]),
        .f16 => @floatCast(self.items(f16)[i]),
        .bf16 => self.items(zml.floats.BFloat16)[i].toF32(),
        .f32 => self.items(f32)[i],
        .f64 => self.items(f64)[i],
        .bool => if (self.bytes[i] != 0) 1 else 0,
        else => @floatFromInt(self.getI(i)),
    };
}

/// f32 to bf16 bits, round to nearest even (zml.floats.BFloat16.fromF32 rounds half up).
/// Going through f32 first is fine: 24 bits is enough to round twice without error.
pub fn bf16Rne(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((bits >> 16) | 0x40);
    const lsb = (bits >> 16) & 1;
    return @intCast((bits + 0x7fff + lsb) >> 16);
}

pub fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// Stores `x`, rounding it to the nearest value representable in the dtype.
pub fn setF(self: Value, i: usize, x: f64) void {
    switch (self.dtype()) {
        .f4e2m1 => self.setBits(i, MiniFloat.of(.f4e2m1).encode(x)),
        .f8e3m4, .f8e4m3, .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .f8e8m0 => self.bytes[i] = MiniFloat.of(self.dtype()).encode(x),
        .f16 => self.items(f16)[i] = @floatCast(x),
        .bf16 => self.items(u16)[i] = bf16Rne(@floatCast(x)),
        .f32 => self.items(f32)[i] = @floatCast(x),
        .f64 => self.items(f64)[i] = x,
        else => unreachable,
    }
}

pub fn getI(self: Value, i: usize) i128 {
    return switch (self.dtype()) {
        .bool => self.bytes[i],
        .i2, .i4, .u2, .u4 => wrapInt(self.dtype(), self.getBits(i)),
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |t| self.items(t.toZigType())[i],
        else => unreachable,
    };
}

/// Stores `x` with two's complement wrapping.
pub fn setI(self: Value, i: usize, x: i128) void {
    switch (self.dtype()) {
        .bool => self.bytes[i] = @intFromBool(x != 0),
        .i2, .i4, .u2, .u4 => self.setBits(i, @truncate(@as(u128, @bitCast(x)))),
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |t| {
            self.items(t.toZigType())[i] = @intCast(wrapInt(t, x));
        },
        else => unreachable,
    }
}

/// Generic element read for value-agnostic code: floats as f64, integers converted.
pub fn getAsF(self: Value, i: usize) f64 {
    return self.getF(i);
}

pub fn items(self: Value, comptime T: type) []T {
    return @as([*]T, @ptrCast(@alignCast(self.bytes.ptr)))[0..self.count()];
}

pub fn elemSize(self: Value) usize {
    return self.dtype().sizeOf();
}

/// Copies element `src_i` of `src` into element `dst_i` of `self`. Dtypes must match.
pub fn copyElem(self: Value, dst_i: usize, src: Value, src_i: usize) void {
    if (subByteBits(self.dtype()) != 0) return self.setBits(dst_i, src.getBits(src_i));
    const n = self.elemSize();
    @memcpy(self.bytes[dst_i * n ..][0..n], src.bytes[src_i * n ..][0..n]);
}

pub fn maxAbs(self: Value) f64 {
    var m: f64 = 0;
    for (0..self.count()) |i| m = @max(m, @abs(self.getF(i)));
    return m;
}

/// True when every element is finite (always true for integers).
pub fn allFinite(self: Value) bool {
    if (!isFloat(self.dtype())) return true;
    for (0..self.count()) |i| {
        if (!std.math.isFinite(self.getF(i))) return false;
    }
    return true;
}

/// Row-major strides in elements.
pub fn strides(shape: Shape) [Shape.MAX_RANK]usize {
    var res: [Shape.MAX_RANK]usize = @splat(0);
    var acc: usize = 1;
    var ax: usize = shape.rank();
    while (ax > 0) {
        ax -= 1;
        res[ax] = acc;
        acc *= @intCast(shape.dim(ax));
    }
    return res;
}

/// Multi-dimensional index iterator over a shape, in row-major order.
pub const Index = struct {
    dims: [Shape.MAX_RANK]i64,
    rank: usize,
    idx: [Shape.MAX_RANK]i64 = @splat(0),
    done: bool,

    pub fn init(shape: Shape) Index {
        var dims: [Shape.MAX_RANK]i64 = @splat(0);
        @memcpy(dims[0..shape.rank()], shape.dims());
        return .{ .dims = dims, .rank = shape.rank(), .done = shape.count() == 0 };
    }

    /// Advances to the next index. Returns false once all indices were visited.
    pub fn next(self: *Index) bool {
        var ax = self.rank;
        while (ax > 0) {
            ax -= 1;
            self.idx[ax] += 1;
            if (self.idx[ax] < self.dims[ax]) return true;
            self.idx[ax] = 0;
        }
        self.done = true;
        return false;
    }
};

pub fn offset(st: [Shape.MAX_RANK]usize, idx: []const i64) usize {
    var off: usize = 0;
    for (idx, 0..) |v, ax| off += @as(usize, @intCast(v)) * st[ax];
    return off;
}

/// f8 and f4 formats, from their specs (the MLIR / OCP definitions), independently of
/// zml.floats.
pub const MiniFloat = struct {
    exp_bits: u4,
    man_bits: u4,
    bias: i32,
    /// IEEE like: the max exponent encodes inf and NaN.
    ieee: bool,
    /// FNUZ: no negative zero, 0x80 is the only NaN.
    fnuz: bool,
    /// e4m3fn: only S.1111.111 is NaN, no infinity.
    fn_nan: bool,
    signed: bool = true,

    pub fn of(dt: DataType) MiniFloat {
        return switch (dt) {
            .f8e4m3fn => .{ .exp_bits = 4, .man_bits = 3, .bias = 7, .ieee = false, .fnuz = false, .fn_nan = true },
            .f8e4m3fnuz => .{ .exp_bits = 4, .man_bits = 3, .bias = 8, .ieee = false, .fnuz = true, .fn_nan = false },
            .f8e4m3b11fnuz => .{ .exp_bits = 4, .man_bits = 3, .bias = 11, .ieee = false, .fnuz = true, .fn_nan = false },
            .f8e4m3 => .{ .exp_bits = 4, .man_bits = 3, .bias = 7, .ieee = true, .fnuz = false, .fn_nan = false },
            .f8e5m2 => .{ .exp_bits = 5, .man_bits = 2, .bias = 15, .ieee = true, .fnuz = false, .fn_nan = false },
            .f8e5m2fnuz => .{ .exp_bits = 5, .man_bits = 2, .bias = 16, .ieee = false, .fnuz = true, .fn_nan = false },
            .f8e3m4 => .{ .exp_bits = 3, .man_bits = 4, .bias = 3, .ieee = true, .fnuz = false, .fn_nan = false },
            .f8e8m0 => .{ .exp_bits = 8, .man_bits = 0, .bias = 127, .ieee = false, .fnuz = false, .fn_nan = false, .signed = false },
            .f4e2m1 => .{ .exp_bits = 2, .man_bits = 1, .bias = 1, .ieee = false, .fnuz = false, .fn_nan = false },
            else => unreachable,
        };
    }

    pub fn decode(f: MiniFloat, code: u8) f64 {
        const man_mask: u8 = @intCast((@as(u16, 1) << f.man_bits) - 1);
        const exp_mask: u8 = @intCast((@as(u16, 1) << f.exp_bits) - 1);
        const man = code & man_mask;
        const e = (code >> @as(u3, @intCast(f.man_bits))) & exp_mask;
        const sign_bit: u8 = if (f.signed) @as(u8, 1) << @intCast(@as(u8, f.exp_bits) + f.man_bits) else 0;
        const neg = f.signed and code & sign_bit != 0;
        if (!f.signed) {
            if (code == 0xff) return std.math.nan(f64);
            return std.math.ldexp(@as(f64, 1), @as(i32, e) - f.bias);
        }
        if (f.fnuz and code == sign_bit) return std.math.nan(f64);
        if (f.fn_nan and e == exp_mask and man == man_mask) return std.math.nan(f64);
        if (f.ieee and e == exp_mask) {
            if (man != 0) return std.math.nan(f64);
            return if (neg) -std.math.inf(f64) else std.math.inf(f64);
        }
        const m: f64 = @floatFromInt(man);
        const scale = std.math.ldexp(@as(f64, 1), f.man_bits);
        const mag = if (e == 0)
            std.math.ldexp(m / scale, 1 - f.bias)
        else
            std.math.ldexp(1 + m / scale, @as(i32, e) - f.bias);
        return if (neg) -mag else mag;
    }

    /// Round to nearest, ties to even, by searching all finite codes.
    pub fn encode(f: MiniFloat, x: f64) u8 {
        const n: u16 = @as(u16, 1) << @intCast(@as(u8, f.exp_bits) + f.man_bits + @intFromBool(f.signed));
        var best: u8 = 0;
        var best_d: f64 = std.math.inf(f64);
        for (0..n) |c| {
            const code: u8 = @intCast(c);
            const v = f.decode(code);
            if (!std.math.isFinite(v)) continue;
            // The positive zero only: -0.0 doesn't exist in FNUZ and is NaN-encoded.
            if (v == 0 and std.math.signbit(v) and f.fnuz) continue;
            const d = @abs(v - x);
            const better = d < best_d or (d == best_d and code & 1 == 0 and best & 1 == 1);
            if (better) {
                best = code;
                best_d = d;
            }
        }
        // Negative values that round to zero keep their sign (none in FNUZ formats).
        if (f.decode(best) == 0 and std.math.signbit(x) and f.signed and !f.fnuz) {
            return @as(u8, 1) << @intCast(@as(u8, f.exp_bits) + f.man_bits);
        }
        return best;
    }
};

test MiniFloat {
    const e4m3fn = MiniFloat.of(.f8e4m3fn);
    try std.testing.expectEqual(@as(f64, 448), e4m3fn.decode(0x7e));
    try std.testing.expectEqual(std.math.ldexp(@as(f64, 1), -9), e4m3fn.decode(0x01));
    try std.testing.expect(std.math.isNan(e4m3fn.decode(0x7f)));
    try std.testing.expectEqual(@as(f64, 1), MiniFloat.of(.f8e4m3fnuz).decode(0x40));
    try std.testing.expect(std.math.isNan(MiniFloat.of(.f8e4m3fnuz).decode(0x80)));
    try std.testing.expectEqual(@as(f64, 57344), MiniFloat.of(.f8e5m2).decode(0x7b));
    try std.testing.expect(std.math.isInf(MiniFloat.of(.f8e5m2).decode(0x7c)));
    try std.testing.expectEqual(@as(f64, 1), MiniFloat.of(.f8e8m0).decode(0x7f));
    try std.testing.expectEqual(@as(f64, 6), MiniFloat.of(.f4e2m1).decode(0x7));
    // Round trips through the encoder, ties to even.
    inline for (.{ .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .f8e4m3b11fnuz, .f8e3m4, .f8e4m3, .f4e2m1 }) |dt| {
        const f = MiniFloat.of(dt);
        for (0..@as(usize, 1) << @intCast(@as(u8, f.exp_bits) + f.man_bits + 1)) |c| {
            const v = f.decode(@intCast(c));
            if (!std.math.isFinite(v)) continue;
            try std.testing.expectEqual(v, f.decode(f.encode(v)));
        }
    }
    try std.testing.expectEqual(@as(u8, 0x38), MiniFloat.of(.f8e4m3fn).encode(1.0));
    try std.testing.expectEqual(@as(u8, 0x38), MiniFloat.of(.f8e4m3fn).encode(1.0 + 1.0 / 16.0));
}

test wrapInt {
    try std.testing.expectEqual(@as(i128, -128), wrapInt(.i8, 128));
    try std.testing.expectEqual(@as(i128, 255), wrapInt(.u8, -1));
    try std.testing.expectEqual(@as(i128, -8), wrapInt(.i4, 8));
    try std.testing.expectEqual(@as(i128, 3), wrapInt(.u2, 7));
    try std.testing.expectEqual(@as(i128, std.math.maxInt(u64)), wrapInt(.u64, -1));
}
