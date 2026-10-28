const std = @import("std");
const zml = @import("zml");

// DataType.resolvePeerType returns the first argument when both have the same size: it
// isn't commutative (f16 with bf16 gives f16, bf16 with f16 gives bf16, likewise for
// the f8 formats) and it resolves u8 with i8 to u8, or u32 with i16 to u32, types that
// can't hold the other's values (zml/dtype.zig:179-190 at 74c590b0). Found by reading the
// code while adding dtype coverage to zmlsmith.
test "peer_type" {
    const D = zml.DataType;
    try std.testing.expectEqual(D.f16.resolvePeerType(.bf16), D.bf16.resolvePeerType(.f16));
    try std.testing.expect(D.u8.resolvePeerType(.i8) != D.u8);
    try std.testing.expect(D.u32.resolvePeerType(.i16) != D.u32);
}
