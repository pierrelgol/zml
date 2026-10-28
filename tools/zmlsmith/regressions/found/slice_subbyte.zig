const std = @import("std");
const zml = @import("zml");

// An i4 {2, 4} tensor is 8 nibbles, 4 bytes, and each row is 4 nibbles = 2 bytes, so the
// byte stride of dim 0 must be 2. Shape.computeByteStrides builds strides from
// dtype().sizeOf(), which is 1 byte for i4 (zml/shape.zig:1219-1220 at 74c590b0), so it
// returns 4 and subSlice moves to the wrong offset.
test "slice_subbyte" {
    const sh = zml.Shape.init(.{ 2, 4 }, .i4);
    try std.testing.expectEqual(@as(usize, 4), sh.byteSize());
    try std.testing.expectEqual(@as(i64, 2), sh.computeByteStrides().get(0));

    var bytes: [4]u8 = .{ 0x21, 0x43, 0x65, 0x87 };
    const s: zml.Slice = .init(sh, &bytes);
    try std.testing.expectEqual(@as(usize, 2), s.subSlice(0, 1, 1).offset_bytes);
}
