const std = @import("std");
const zml = @import("zml");

// Exe.Arguments allocates its per device buffer table undefined (FlatBuffers.init,
// zml/exe.zig:87 at 74c590b0) and Exe.call never checks that every argument was set:
// calling a two argument executable after setting one passes an undefined pointer to
// PJRT_LoadedExecutable_Execute, which segfaults instead of failing with "argument 1 not
// set" (zml/exe.zig:306-316). Found by zmlsmith robust (call_too_few_args).
test "exe_args_unset" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Local = struct {
        fn add(x: zml.Tensor, y: zml.Tensor) zml.Tensor {
            return x.add(y);
        }
    };
    const sh: zml.Shape = .init(.{4}, .f32);
    var exe = try zml.module.compile(allocator, io, Local.add, .{ zml.Tensor.fromShape(sh), zml.Tensor.fromShape(sh) }, platform, .{});
    defer exe.deinit();
    var b = try zml.Buffer.fromBytes(io, platform, sh, .replicated, std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3, 4 }));
    defer b.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{b});
    exe.callOpts(io, args, &results, .{ .wait = true });
}
