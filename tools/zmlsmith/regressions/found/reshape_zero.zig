const std = @import("std");
const zml = @import("zml");

test "Shape.reshape(-1) with a zero dim" {
    const sh: zml.Shape = .init(.{ 0, 3 }, .f32);
    try zml.testing.expectEqualShapes(zml.Shape.init(.{0}, .f32), sh.reshape(.{-1}));
}
