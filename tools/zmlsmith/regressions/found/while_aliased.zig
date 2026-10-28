const std = @import("std");
const zml = @import("zml");

// The same tensor in two loop state slots: ops.while registers block arguments by tensor
// id, so both slots read the last registered argument.
test "while_aliased_state" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Loop = struct {
        pub const State = struct { i: zml.Tensor, a: zml.Tensor, b: zml.Tensor };
        pub fn cond(_: @This(), st: State) zml.Tensor {
            return st.i.cmp(.LT, zml.Tensor.scalar(3, .i32));
        }
        pub fn body(_: @This(), st: State) State {
            return .{ .i = st.i.addConstant(1), .a = st.a.addConstant(1), .b = st.b.scale(2) };
        }
    };
    const Local = struct {
        fn forward(x: zml.Tensor) [2]zml.Tensor {
            const r = zml.ops.@"while"(Loop, .{}, .{ .i = zml.Tensor.scalar(0, .i32), .a = x, .b = x });
            return .{ r.a, r.b };
        }
    };

    const x: zml.Tensor = .fromShape(zml.Shape.init(.{}, .f32));
    var exe = try zml.module.compile(allocator, io, Local.forward, .{x}, platform, .{});
    defer exe.deinit();

    var bx: zml.Buffer = try .scalar(io, platform, 1.0, .f32);
    defer bx.deinit();
    var res = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{bx});
    defer for (&res) |*r| r.deinit();

    // a: 1 + 3 = 4, b: 1 * 2^3 = 8.
    try std.testing.expectEqual(@as(f32, 4), try res[0].getValue(f32, io));
    try std.testing.expectEqual(@as(f32, 8), try res[1].getValue(f32, io));
}
