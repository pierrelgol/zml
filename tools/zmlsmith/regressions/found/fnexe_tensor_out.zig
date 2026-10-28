const std = @import("std");
const zml = @import("zml");

// FnExe(f).Output is built by typedOutputDestinations, which takes one pointer per field
// of the Bufferized result. For a function returning a bare Tensor the Bufferized result
// is a Buffer, so Output becomes a struct of pointers to Buffer's own fields and the
// result can't be received: `r.run(.{ .outputs = &out })` doesn't compile
// (zml/exe.zig:534 at 74c590b0). A single tensor is the common model output. Found while
// adding FnExe coverage to zmlsmith.
const Fns = struct {
    fn forward(in: struct { x: zml.Tensor }) zml.Tensor {
        return in.x.negate();
    }
};

test "fnexe_tensor_out" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const T = zml.FnExe(Fns.forward);
    const x: zml.Tensor = .init(.{4}, .f32);
    var exe = try T.compile(allocator, io, platform, .{}, .{.{ .x = x }});
    defer exe.deinit();
    var b = try zml.Buffer.fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3, 4 }));
    defer b.deinit();
    var r = try exe.runner(allocator);
    defer r.deinit(allocator);
    var out: zml.Buffer = undefined;
    r.run(io, .{ .inputs = .{ .x = b }, .outputs = &out, .opts = .{ .wait = true } });
    out.deinit();
}
