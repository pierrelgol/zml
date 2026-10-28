//! Parser fuzzing: the ZML entry points that read untrusted bytes. Inputs are valid seeds
//! mutated structurally (JSON values replaced, deleted, retyped, extreme numbers and
//! strings) and at the byte level, then fed to the parser. Returning an error is fine; a
//! panic, abort or hang is a finding. When a tokenizer loads, it also encodes and decodes
//! a few texts and random ids.
//!
//!   zmlsmith parse-fuzz --target T [--seed S] [--count N] [--jobs J] [--out DIR]
//!   zmlsmith parse-run --target T FILE      run one input in process
//!   zmlsmith parse-gen --target T --seed S  print the input generated from a seed

const std = @import("std");

const zml = @import("zml");

const child = @import("child.zig");

pub const Target = enum { safetensors, tokenizer, homemade, rope, vfs, spm, s3, gcs };

const allocator = std.heap.c_allocator;

// -----------------------------------------------------------------------------
// Seeds.

const st_header =
    \\{"__metadata__":{"format":"pt","name":"m"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"BF16","shape":[3],"data_offsets":[16,22]},"i":{"dtype":"I8","shape":[],"data_offsets":[22,23]}}
;

const tok_bpe =
    \\{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":5,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
    \\"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
    \\"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
    \\"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,
    \\"vocab":{"a":0,"b":1,"ab":2,"Ġ":3,"Ġa":4,"<s>":5},"merges":["a b","Ġ a"]}}
;

const tok_wordpiece =
    \\{"version":"1.0","added_tokens":[],"normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
    \\"pre_tokenizer":{"type":"BertPreTokenizer"},"post_processor":null,"decoder":{"type":"WordPiece","prefix":"##","cleanup":true},
    \\"model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"[UNK]":0,"a":1,"##b":2,"hello":3}}}
;

const tok_llama =
    \\{"version":"1.0","added_tokens":[],"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"\u2581"},{"type":"Replace","pattern":{"String":" "},"content":"\u2581"}]},
    \\"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"\u2581"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]},
    \\"model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":true,"byte_fallback":true,
    \\"vocab":{"<unk>":0,"<s>":1,"</s>":2,"\u2581":3,"a":4,"\u2581a":5,"<0x41>":6},"merges":["\u2581 a"]}}
;

const tok_seeds = [_][]const u8{ tok_bpe, tok_wordpiece, tok_llama };

const rope_seeds = [_][]const u8{
    \\{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":4096,"beta_fast":32,"beta_slow":1,"mscale":1.0,"mscale_all_dim":0.5,"rope_theta":10000}
    ,
    \\{"rope_type":"llama3","factor":8.0,"high_freq_factor":4.0,"low_freq_factor":1.0,"original_max_position_embeddings":8192}
    ,
    \\{"rope_type":"linear","factor":2.0}
    ,
    \\{"rope_type":"default","rope_theta":500000}
    ,
    \\{"rope_type":"proportional","partial_rotary_factor":0.5}
    ,
};

/// Server answers for the vfs target: the HEAD answer, the separator, the GET answer.
const http_sep = "\n--GET--\n";
const http_seeds = [_][]const u8{
    "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 16\r\n\r\n" ++ http_sep ++
        "HTTP/1.1 206 Partial Content\r\nConnection: close\r\nContent-Range: bytes 0-15/16\r\nContent-Length: 16\r\n\r\n0123456789abcdef",
    "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 16\r\n\r\n" ++ http_sep ++
        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 16\r\n\r\n0123456789abcdef",
    "HTTP/1.1 302 Found\r\nConnection: close\r\nLocation: /g.bin\r\nContent-Length: 0\r\n\r\n" ++ http_sep ++
        "HTTP/1.1 206 Partial Content\r\nConnection: close\r\nContent-Range: bytes 0-7/8\r\nContent-Length: 8\r\n\r\n01234567",
    "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 32\r\n\r\n" ++ http_sep ++
        "HTTP/1.1 206 Partial Content\r\nConnection: close\r\nTransfer-Encoding: chunked\r\nContent-Range: bytes 0-31/32\r\n\r\n10\r\n0123456789abcdef\r\n10\r\n0123456789abcdef\r\n0\r\n\r\n",
};

/// Replaces a header line value or number by an odd one, keeping the HTTP framing.
fn mutateHttp(a: std.mem.Allocator, r: std.Random, src: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitSequence(u8, src, "\r\n");
    const odd = [_][]const u8{ "", "-", "/", " ", "-1", "18446744073709551615", "99999999999999999999", "bytes */16", "0-15 bytes/16", "bytes 15-0/16", "bytes 0-/16", "bytes 0-15/", "bytes 0-99/16", "chunked", "/", "http://", "0x10" };
    while (lines.next()) |line| {
        var l = line;
        if (std.mem.indexOfScalar(u8, line, ':') != null and r.uintLessThan(u8, 4) == 0) {
            const colon = std.mem.indexOfScalar(u8, line, ':').?;
            switch (r.uintLessThan(u8, 3)) {
                0 => l = try std.fmt.allocPrint(a, "{s}: {s}", .{ line[0..colon], pick([]const u8, r, &odd) }),
                1 => continue, // drop the header
                else => l = try std.fmt.allocPrint(a, "{s}\r\n{s}", .{ line, line }), // duplicate it
            }
        } else if (std.mem.startsWith(u8, line, "HTTP/1.1 ") and r.uintLessThan(u8, 6) == 0) {
            l = try std.fmt.allocPrint(a, "HTTP/1.1 {s} X", .{pick([]const u8, r, &.{ "100", "204", "301", "304", "404", "500", "206", "200", "999", "2000" })});
        }
        if (out.items.len > 0) try out.appendSlice(a, "\r\n");
        try out.appendSlice(a, l);
    }
    return out.items;
}

fn varint(a: std.mem.Allocator, out: *std.ArrayList(u8), x: u64) !void {
    var v = x;
    while (v >= 0x80) : (v >>= 7) try out.append(a, @as(u8, @truncate(v)) | 0x80);
    try out.append(a, @truncate(v));
}

fn lenField(a: std.mem.Allocator, out: *std.ArrayList(u8), field: u8, bytes: []const u8) !void {
    try out.append(a, field << 3 | 2);
    try varint(a, out, bytes.len);
    try out.appendSlice(a, bytes);
}

/// A sentencepiece ModelProto: pieces (field 1: piece, score, type) and maybe a
/// trainer_spec (field 2: model_type, byte_fallback). `mutate` alters the pieces.
fn spmModel(a: std.mem.Allocator, r: std.Random, mutate: bool) ![]u8 {
    const Piece = struct { s: []const u8, score: f32, type: u8 };
    var pieces: std.ArrayList(Piece) = .empty;
    try pieces.appendSlice(a, &.{
        .{ .s = "<unk>", .score = 0, .type = 2 },
        .{ .s = "<s>", .score = 0, .type = 3 },
        .{ .s = "</s>", .score = 0, .type = 3 },
        .{ .s = "\u{2581}", .score = -1, .type = 1 },
        .{ .s = "a", .score = -2, .type = 1 },
        .{ .s = "b", .score = -2, .type = 1 },
        .{ .s = "\u{2581}a", .score = -1.5, .type = 1 },
        .{ .s = "ab", .score = -1, .type = 1 },
        .{ .s = "<0x41>", .score = 0, .type = 6 },
    });
    if (mutate) for (0..1 + r.uintLessThan(usize, 4)) |_| {
        const i = r.uintLessThan(usize, pieces.items.len);
        switch (r.uintLessThan(u8, 6)) {
            0 => pieces.items[i].type = r.uintLessThan(u8, 8),
            1 => pieces.items[i].score = pick(f32, r, &.{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 1e30, 0 }),
            2 => pieces.items[i].s = pick([]const u8, r, &.{ "", "\xff", "\u{2581}\u{2581}", "<0xZZ>", "<0x100>", "a", "<unk>" }),
            3 => _ = pieces.orderedRemove(i),
            4 => try pieces.append(a, pieces.items[i]),
            else => try pieces.insert(a, 0, pieces.items[i]),
        }
    };
    var out: std.ArrayList(u8) = .empty;
    for (pieces.items) |p| {
        var inner: std.ArrayList(u8) = .empty;
        try lenField(a, &inner, 1, p.s);
        try inner.append(a, 2 << 3 | 5);
        try inner.appendSlice(a, std.mem.asBytes(&p.score));
        try inner.append(a, 3 << 3);
        try varint(a, &inner, p.type);
        try lenField(a, &out, 1, inner.items);
    }
    if (r.boolean()) {
        var spec: std.ArrayList(u8) = .empty;
        try spec.append(a, 3 << 3); // model_type: unigram 1, bpe 2, word 3, char 4
        try varint(a, &spec, r.uintLessThan(u8, 6));
        try varint(a, &spec, 35 << 3); // byte_fallback
        try varint(a, &spec, r.uintLessThan(u8, 2));
        try lenField(a, &out, 2, spec.items);
    }
    return out.items;
}

// -----------------------------------------------------------------------------
// Mutators.

fn pick(comptime T: type, r: std.Random, items: []const T) T {
    return items[r.uintLessThan(usize, items.len)];
}

fn randomValue(a: std.mem.Allocator, r: std.Random, depth: u32) !std.json.Value {
    return switch (r.uintLessThan(u8, if (depth > 2) 6 else 8)) {
        0 => .null,
        1 => .{ .bool = r.boolean() },
        2 => .{ .integer = @as(i64, r.uintLessThan(u8, 8)) - 2 },
        3 => .{ .integer = pick(i64, r, &.{ -1, 0, 1, 255, 256, 65535, 1 << 31, (1 << 31) - 1, -(1 << 31), 1 << 40, std.math.maxInt(i64), std.math.minInt(i64) }) },
        4 => .{ .float = pick(f64, r, &.{ 0.0, -0.0, 0.5, -1.5, 1e308, -1e308, 1e-320, 3.0 }) },
        5 => .{ .string = switch (r.uintLessThan(u8, 6)) {
            0 => "",
            1 => "a",
            2 => "\xc3\xa9\xe2\x84\xb3\xf0\x9f\x98\x80",
            3 => try a.dupe(u8, "x" ** 1500),
            4 => "\xff\xfe\x00",
            else => pick([]const u8, r, &.{ "F32", "BF16", "F64", "I64", "U8", "BOOL", "F8_E4M3", "BPE", "WordPiece", "Unigram", "ByteLevel", "yarn", "llama3", "##", "<unk>" }),
        } },
        6 => blk: {
            var arr: std.json.Array = .init(a);
            for (0..r.uintLessThan(usize, 4)) |_| try arr.append(try randomValue(a, r, depth + 1));
            break :blk .{ .array = arr };
        },
        else => blk: {
            var obj: std.json.ObjectMap = .empty;
            for (0..r.uintLessThan(usize, 3)) |i| try obj.put(a, try std.fmt.allocPrint(a, "k{d}", .{i}), try randomValue(a, r, depth + 1));
            break :blk .{ .object = obj };
        },
    };
}

fn collect(list: *std.ArrayList(*std.json.Value), a: std.mem.Allocator, v: *std.json.Value) !void {
    try list.append(a, v);
    switch (v.*) {
        .array => |*arr| for (arr.items) |*x| try collect(list, a, x),
        .object => |*obj| for (obj.values()) |*x| try collect(list, a, x),
        else => {},
    }
}

/// Deep copy: a shallow copy of an array or object shares its storage, which later
/// removals would corrupt.
fn clone(a: std.mem.Allocator, v: std.json.Value) !std.json.Value {
    switch (v) {
        .array => |arr| {
            var res: std.json.Array = .init(a);
            for (arr.items) |x| try res.append(try clone(a, x));
            return .{ .array = res };
        },
        .object => |obj| {
            var res: std.json.ObjectMap = .empty;
            var it = obj.iterator();
            while (it.next()) |e| try res.put(a, e.key_ptr.*, try clone(a, e.value_ptr.*));
            return .{ .object = res };
        },
        else => return v,
    }
}

/// Applies a few structural mutations to a JSON document.
fn mutateJson(a: std.mem.Allocator, r: std.Random, src: []const u8) ![]u8 {
    var root = try std.json.parseFromSliceLeaky(std.json.Value, a, src, .{});
    const n = 1 + r.uintLessThan(u32, 4);
    for (0..n) |_| {
        var nodes: std.ArrayList(*std.json.Value) = .empty;
        try collect(&nodes, a, &root);
        const v = nodes.items[r.uintLessThan(usize, nodes.items.len)];
        switch (r.uintLessThan(u8, 8)) {
            // Replace by a random value.
            0, 1, 2 => v.* = try randomValue(a, r, 0),
            // Small numeric changes: off by one, negation, scaling.
            3 => switch (v.*) {
                .integer => |x| v.* = .{ .integer = pick(i64, r, &.{ x +% 1, x -% 1, -%x, x *% 1000, 0 }) },
                .float => |x| v.* = .{ .float = pick(f64, r, &.{ -x, x * 1e6, 0, x + 0.5 }) },
                else => v.* = try randomValue(a, r, 0),
            },
            // Delete a key or an element.
            4, 5 => switch (v.*) {
                .object => |*obj| if (obj.count() > 0) {
                    _ = obj.orderedRemove(obj.keys()[r.uintLessThan(usize, obj.count())]);
                },
                .array => |*arr| if (arr.items.len > 0) {
                    _ = arr.orderedRemove(r.uintLessThan(usize, arr.items.len));
                },
                else => {},
            },
            // Grow: add a key, append an element, duplicate an element.
            6 => switch (v.*) {
                .object => |*obj| try obj.put(a, pick([]const u8, r, &.{ "shape", "dtype", "data_offsets", "type", "vocab", "merges", "factor", "rope_type", "unk_token", "content", "id", "__metadata__", "x", "k" ** 1100 }), try randomValue(a, r, 0)),
                .array => |*arr| if (arr.items.len > 0 and r.boolean()) {
                    const dup = try clone(a, arr.items[r.uintLessThan(usize, arr.items.len)]);
                    try arr.append(dup);
                } else {
                    try arr.append(try randomValue(a, r, 0));
                },
                else => {},
            },
            // Wrap in an array or object.
            else => {
                var arr: std.json.Array = .init(a);
                try arr.append(v.*);
                v.* = .{ .array = arr };
            },
        }
    }
    return std.json.Stringify.valueAlloc(a, root, .{});
}

fn mutateBytes(a: std.mem.Allocator, r: std.Random, src: []const u8) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    try b.appendSlice(a, src);
    for (0..1 + r.uintLessThan(u32, 3)) |_| {
        if (b.items.len == 0) break;
        const i = r.uintLessThan(usize, b.items.len);
        switch (r.uintLessThan(u8, 5)) {
            0 => b.items[i] ^= @as(u8, 1) << r.int(u3),
            1 => b.items[i] = r.int(u8),
            2 => b.shrinkRetainingCapacity(i),
            3 => _ = b.orderedRemove(i),
            else => try b.insert(a, i, pick(u8, r, &.{ '"', '{', '}', '[', ']', ',', ':', '0', '-', '\\', 0, 0xff })),
        }
    }
    return b.items;
}

fn safetensorsFile(a: std.mem.Allocator, r: std.Random, header: []const u8, data_len: usize) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var len: u64 = header.len;
    if (r.uintLessThan(u8, 8) == 0) len = pick(u64, r, &.{ 0, len -| 1, len + 1, len + 64, 1 << 40, std.math.maxInt(u64) });
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, len, .little);
    try out.appendSlice(a, &len_bytes);
    try out.appendSlice(a, header);
    const n = if (r.uintLessThan(u8, 4) == 0) r.uintLessThan(usize, data_len + 1) else data_len;
    for (0..n) |i| try out.append(a, @truncate(i *% 37));
    return out.items;
}

/// The input of a seed.
pub fn generate(a: std.mem.Allocator, target: Target, seed: u64) ![]u8 {
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    const structural = r.uintLessThan(u8, 10) < 8;
    switch (target) {
        .safetensors => {
            var header: []const u8 = st_header;
            if (structural) header = try mutateJson(a, r, header);
            const file = try safetensorsFile(a, r, header, 23);
            return if (!structural or r.uintLessThan(u8, 5) == 0) mutateBytes(a, r, file) else file;
        },
        .tokenizer, .homemade => {
            const base = tok_seeds[r.uintLessThan(usize, tok_seeds.len)];
            const doc = if (structural) try mutateJson(a, r, base) else try a.dupe(u8, base);
            return if (!structural or r.uintLessThan(u8, 5) == 0) mutateBytes(a, r, doc) else doc;
        },
        .spm => {
            const doc = try spmModel(a, r, structural);
            return if (!structural or r.uintLessThan(u8, 4) == 0) mutateBytes(a, r, doc) else doc;
        },
        .vfs, .s3, .gcs => {
            const base = http_seeds[r.uintLessThan(usize, http_seeds.len)];
            const doc = if (structural) try mutateHttp(a, r, base) else try a.dupe(u8, base);
            return if (!structural or r.uintLessThan(u8, 5) == 0) mutateBytes(a, r, doc) else doc;
        },
        .rope => {
            const base = rope_seeds[r.uintLessThan(usize, rope_seeds.len)];
            const doc = if (structural) try mutateJson(a, r, base) else try a.dupe(u8, base);
            return if (!structural or r.uintLessThan(u8, 5) == 0) mutateBytes(a, r, doc) else doc;
        },
    }
}

// -----------------------------------------------------------------------------
// Targets.

const texts = [_][]const u8{ "", " ", "a", "ab ab", "hello world", "\xc3\xa9\xe2\x84\xb3", "\xff\xfe", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "<s> a<s>b", "\n\t  " };

/// Runs one input. Errors are expected outcomes; only panics and aborts are findings.
pub fn run(io: std.Io, target: Target, bytes: []const u8, tmp_dir: []const u8) void {
    switch (target) {
        .safetensors => {
            // A header length larger than the file is allocated before being checked
            // (found/st_header_len.zig): gigabytes filled with 0xaa in Debug builds, seconds
            // per input. Short reads and overflowing lengths are kept.
            if (bytes.len >= 8) {
                const len = std.mem.readInt(u64, bytes[0..8], .little);
                if (len > bytes.len + (1 << 20) and len < std.math.maxInt(u64) - 64) return;
            }
            const path = std.fmt.allocPrint(allocator, "{s}/in.safetensors", .{tmp_dir}) catch return;
            std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes }) catch return;
            const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return;
            defer file.close(io);
            var registry: zml.safetensors.TensorRegistry = .init(allocator);
            defer registry.deinit();
            zml.safetensors.parseSafetensors(allocator, io, &registry, file) catch return;
            _ = registry.totalBytes();
            var it = registry.iterator();
            while (it.next()) |e| {
                var buf: [64]u8 = undefined;
                var reader = registry.reader(io, e.key_ptr.*, &buf) catch continue;
                var sink: [256]u8 = undefined;
                _ = reader.interface.readSliceShort(&sink) catch {};
            }
        },
        .tokenizer, .spm => {
            // A protobuf starting with '{' would go to the json parser.
            if (target == .spm and bytes.len > 0 and bytes[0] == '{') return;
            var tok = zml.tokenizer.Tokenizer.fromBytes(allocator, bytes) catch |err| {
                std.debug.print("ZMLSMITH_PARSE_ERROR {t}\n", .{err});
                return;
            };
            defer tok.deinit();
            var enc = tok.encoder() catch return;
            defer enc.deinit();
            var dec = tok.decoder() catch return;
            defer dec.deinit();
            for (texts) |t| {
                // Encoding to no tokens panics in sentencepiece (found/spm_encode_empty.zig).
                if (target == .spm and t.len == 0) continue;
                const ids = enc.encodeAlloc(allocator, t) catch continue;
                defer allocator.free(ids);
                var out = dec.decodeAlloc(allocator, ids) catch continue;
                out.deinit(allocator);
            }
            // Ids the vocabulary may not have, decoded in one call and streamed.
            const ids = [_]u32{ 0, 1, 2, 5, 6, 7, 100, 65535, 1 << 20, std.math.maxInt(u32) };
            for (ids) |id| {
                var out = dec.decodeAlloc(allocator, &.{id}) catch continue;
                out.deinit(allocator);
            }
            var buf: [64]u8 = undefined;
            for (ids) |id| _ = dec.feedOne(id, &buf) catch {};
            _ = dec.finalize(&buf) catch {};
        },
        // The pure Zig normalizer config loader. homemade.fromHfJson, the full loader,
        // does not compile (found/tok_homemade_load.zig).
        .homemade => {
            const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return;
            defer parsed.deinit();
            if (parsed.value != .object) return;
            const config = switch (parsed.value.object.get("normalizer") orelse parsed.value) {
                .object => |o| o,
                else => return,
            };
            const n = zml.tokenizer.homemade.Normalizer.fromHfJson(config) catch return;
            for (texts) |t| {
                const res = n.normalizeWithMapping(allocator, t) catch continue;
                res.deinit(allocator);
            }
        },
        .vfs => runVfs(io, bytes, .http) catch {},
        .s3 => runVfs(io, bytes, .s3) catch {},
        .gcs => runVfs(io, bytes, .gcs) catch {},
        .rope => {
            // The jsonParseFromValue path, through a parsed std.json.Value, must agree.
            if (std.json.parseFromSlice(std.json.Value, allocator, bytes, .{})) |v| {
                defer v.deinit();
                if (std.json.parseFromValue(zml.nn.RopeOpts.Scaling, allocator, v.value, .{})) |pv| {
                    defer pv.deinit();
                    var s = pv.value;
                    _ = s.attentionScaling();
                    _ = s.getRopeTheta();
                    _ = s.partialRotaryDim(64);
                    if (std.json.parseFromSlice(zml.nn.RopeOpts.Scaling, allocator, bytes, .{})) |ps| {
                        defer ps.deinit();
                        if (std.meta.activeTag(ps.value) != std.meta.activeTag(pv.value)) @panic("rope Scaling: jsonParse and jsonParseFromValue disagree");
                    } else |_| @panic("rope Scaling: jsonParseFromValue accepts what jsonParse rejects");
                } else |_| {}
            } else |_| {}
            const parsed = std.json.parseFromSlice(zml.nn.RopeOpts.Scaling, allocator, bytes, .{}) catch return;
            defer parsed.deinit();
            var s = parsed.value;
            _ = s.attentionScaling();
            _ = s.getRopeTheta();
            s.setRopeTheta(1000);
            _ = s.partialRotaryDim(64);
        },
    }
}

/// Serves the HEAD and GET answers of `bytes` from a local socket, and reads the file
/// through VFS.HTTP: open (HEAD, redirects), then ranged reads to the end.
fn runVfs(io: std.Io, bytes: []const u8, backend: enum { http, s3, gcs }) !void {
    const sep = std.mem.indexOf(u8, bytes, http_sep);
    const head = if (sep) |i| bytes[0..i] else bytes;
    const get = if (sep) |i| bytes[i + http_sep.len ..] else bytes;
    // Known HEAD stage crashes, reported: a 2xx without Content-Length (http, s3, gcs
    // fetchSize) and any redirect (found/vfs_redirect_*.zig, vfs_bad_location.zig). Skip
    // them to reach the read path.
    if (head.len >= 12 and std.mem.startsWith(u8, head, "HTTP/1.1 ")) {
        const class = head[9];
        var has_len = false;
        var lines = std.mem.splitSequence(u8, head, "\r\n");
        while (lines.next()) |l| {
            if (std.ascii.startsWithIgnoreCase(l, "content-length:")) has_len = true;
        }
        if (class == '3' or (class == '2' and !has_len)) return;
    }
    const Server = struct {
        server: std.Io.net.Server,
        head: []const u8,
        get: []const u8,
        io: std.Io,
        done: std.atomic.Value(bool) = .init(false),
        fn serve(sv: *@This()) void {
            while (true) {
                const conn = sv.server.accept(sv.io) catch return;
                defer conn.close(sv.io);
                if (sv.done.load(.acquire)) return;
                var rbuf: [4096]u8 = undefined;
                var rd = conn.reader(sv.io, &rbuf);
                const first = rd.interface.takeDelimiterExclusive('\n') catch continue;
                const is_head = std.mem.startsWith(u8, first, "HEAD");
                rd.interface.toss(1);
                while (true) {
                    const line = rd.interface.takeDelimiterExclusive('\n') catch break;
                    rd.interface.toss(1);
                    if (line.len <= 1) break;
                }
                var wbuf: [4096]u8 = undefined;
                var w = conn.writer(sv.io, &wbuf);
                w.interface.writeAll(if (is_head) sv.head else sv.get) catch continue;
                w.interface.flush() catch {};
            }
        }
    };
    const addr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var sv: Server = .{ .server = try addr.listen(io, .{ .reuse_address = true }), .head = head, .get = get, .io = io };
    const t = try std.Thread.spawn(.{}, Server.serve, .{&sv});
    // Closing the socket doesn't wake a blocked accept: connect once more instead.
    defer {
        sv.done.store(true, .release);
        if (sv.server.socket.address.connect(io, .{ .mode = .stream })) |c| c.close(io) else |_| {}
        t.join();
        sv.server.deinit(io);
    }

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http: zml.io.VFS.HTTP = try .init(allocator, io, &client, .http);
    defer http.deinit();
    var vfs: zml.io.VFS = try .init(allocator, io);
    defer vfs.deinit();
    var url_buf: [64]u8 = undefined;
    const port = sv.server.socket.address.getPort();
    var s3_vfs: ?zml.io.VFS.S3 = null;
    defer if (s3_vfs) |*v| v.deinit();
    var gcs_vfs: ?zml.io.VFS.GCS = null;
    defer if (gcs_vfs) |*v| v.deinit();
    var ep_buf: [64]u8 = undefined;
    const endpoint = try std.fmt.bufPrint(&ep_buf, "http://127.0.0.1:{d}", .{port});
    const pool: @import("zml").io.VFS.S3.InitOpts = .{ .read_pool = .{ .chunk_size = 16, .num_workers = 2, .queue_capacity = 8, .max_retries = 0, .retry_initial_delay = .fromMilliseconds(1), .retry_max_delay = .fromMilliseconds(1) } };
    const url = switch (backend) {
        .s3 => blk: {
            s3_vfs = try .init(allocator, io, &client, .{ .access_key = "a", .secret_key = "b", .endpoint_url = endpoint, .region = "us-east-1" }, pool);
            try vfs.register("s3", s3_vfs.?.io());
            break :blk "s3://bucket/f.bin";
        },
        .gcs => blk: {
            gcs_vfs = try .init(allocator, io, &client, .{ .endpoint_url = endpoint, .read_pool = pool.read_pool });
            try vfs.register("gs", gcs_vfs.?.io());
            break :blk "gs://bucket/f.bin";
        },
        .http => blk: {
            try vfs.register("http", http.io());
            break :blk try std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/f.bin", .{port});
        },
    };
    const vio = vfs.io();
    var file = try std.Io.Dir.openFile(.cwd(), vio, url, .{ .mode = .read_only });
    defer file.close(vio);
    var buf: [64]u8 = undefined;
    var off: u64 = 0;
    for (0..8) |_| {
        const n = try file.readPositional(vio, &.{&buf}, off);
        if (n == 0) break;
        off += n;
    }
}

// -----------------------------------------------------------------------------
// Driver.

pub const Options = struct {
    target: Target,
    seed: u64,
    count: u64,
    jobs: u32,
    out: []const u8,
    timeout_s: i64,
    exe: []const u8,
    /// Time limit for reducing each new signature.
    reduce_s: i64 = 180,
};

/// Worker: runs seeds in process, printing START before each so the supervisor can tell
/// which one crashed.
pub fn batch(io: std.Io, target: Target, seed: u64, count: u64, tmp_dir: []const u8) void {
    for (0..count) |i| {
        const s = seed + i;
        std.debug.print("ZMLSMITH_START seed={d}\n", .{s});
        // Also in a file: the log is lost when the batch times out.
        var buf: [32]u8 = undefined;
        const status = std.fmt.allocPrint(allocator, "{s}/status", .{tmp_dir}) catch continue;
        defer allocator.free(status);
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = status, .data = std.fmt.bufPrint(&buf, "{d}", .{s}) catch continue }) catch {};
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const bytes = generate(arena.allocator(), target, s) catch continue;
        run(io, target, bytes, tmp_dir);
    }
}

fn lastStart(stderr: []const u8) ?u64 {
    const i = std.mem.lastIndexOf(u8, stderr, "ZMLSMITH_START seed=") orelse return null;
    const start = i + "ZMLSMITH_START seed=".len;
    var end = start;
    while (end < stderr.len and std.ascii.isDigit(stderr[end])) end += 1;
    return std.fmt.parseInt(u64, stderr[start..end], 10) catch null;
}

fn statusSeed(io: std.Io, tmp: []const u8) ?u64 {
    const path = std.fmt.allocPrint(allocator, "{s}/status", .{tmp}) catch return null;
    defer allocator.free(path);
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(32)) catch return null;
    defer allocator.free(data);
    return std.fmt.parseInt(u64, data, 10) catch null;
}

const Shared = struct {
    io: std.Io,
    opts: Options,
    mutex: std.Io.Mutex = .init,
    next: u64,
    end: u64,
    seen: std.StringHashMapUnmanaged(u64) = .empty,
    crashes: u64 = 0,
    done: bool = false,
};

fn runFile(io: std.Io, opts: Options, path: []const u8) !child.Outcome {
    const tgt = @tagName(opts.target);
    return child.run(allocator, io, &.{ opts.exe, "parse-run", "--target", tgt, path }, opts.timeout_s);
}

/// Shrinks a crashing input while it keeps its signature: removes chunks of halving
/// sizes, then simplifies bytes.
fn reduceInput(io: std.Io, opts: Options, input: []const u8, sig: []const u8, path: []const u8) ![]u8 {
    var best = try allocator.dupe(u8, input);
    var budget: u32 = 400;
    // Crashing children symbolize their stack trace, which takes seconds: also bound the time.
    const t0: std.Io.Timestamp = .now(io, .awake);
    var chunk = @max(1, best.len / 2);
    while (chunk >= 1 and budget > 0) {
        var i: usize = 0;
        var progress = false;
        while (i < best.len and budget > 0 and t0.untilNow(io, .awake).nanoseconds < opts.reduce_s * std.time.ns_per_s) {
            const end = @min(best.len, i + chunk);
            const cand = try std.mem.concat(allocator, u8, &.{ best[0..i], best[end..] });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = cand });
            budget -= 1;
            const o = try runFile(io, opts, path);
            if (std.mem.eql(u8, o.signature, sig)) {
                best = cand;
                progress = true;
            } else {
                i += chunk;
            }
        }
        if (!progress) {
            if (chunk == 1) break;
            chunk /= 2;
        }
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = best });
    return best;
}

fn report(sh: *Shared, seed: u64, out: child.Outcome) !void {
    sh.mutex.lockUncancelable(sh.io);
    const gop = try sh.seen.getOrPut(allocator, out.signature);
    sh.crashes += 1;
    if (gop.found_existing) {
        gop.value_ptr.* += 1;
        sh.mutex.unlock(sh.io);
        return;
    }
    gop.key_ptr.* = try allocator.dupe(u8, out.signature);
    gop.value_ptr.* = 1;
    _ = writeSummary(sh) catch "";
    sh.mutex.unlock(sh.io);

    const io = sh.io;
    const dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ sh.opts.out, try child.slug(allocator, out.signature) });
    try std.Io.Dir.cwd().createDirPath(io, dir);
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const input = try generate(arena.allocator(), sh.opts.target, seed);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(allocator, "{s}/seed-{d}.bin", .{ dir, seed }), .data = input });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(allocator, "{s}/seed-{d}.log", .{ dir, seed }), .data = out.log });
    const reduced_path = try std.fmt.allocPrint(allocator, "{s}/reduced.bin", .{dir});
    const reduced = try reduceInput(io, sh.opts, input, out.signature, reduced_path);
    std.debug.print("[parse-fuzz] {s} (seed {d}): {d} -> {d} bytes\n", .{ out.signature, seed, input.len, reduced.len });
}

fn worker(sh: *Shared, id: usize) void {
    const io = sh.io;
    const tmp = std.fmt.allocPrint(allocator, "{s}/tmp{d}", .{ sh.opts.out, id }) catch return;
    std.Io.Dir.cwd().createDirPath(io, tmp) catch return;
    const tgt = @tagName(sh.opts.target);
    while (true) {
        sh.mutex.lockUncancelable(sh.io);
        const start = sh.next;
        const n = @min(200, sh.end -| start);
        sh.next += n;
        sh.mutex.unlock(sh.io);
        if (n == 0) return;
        var s = start;
        while (s < start + n) {
            const seed_s = std.fmt.allocPrint(allocator, "{d}", .{s}) catch return;
            const count_s = std.fmt.allocPrint(allocator, "{d}", .{start + n - s}) catch return;
            const out = child.run(allocator, io, &.{ sh.opts.exe, "parse-batch", "--target", tgt, "--seed", seed_s, "--count", count_s, "--out", tmp }, sh.opts.timeout_s * 10) catch return;
            if (!out.crashed) break;
            // The crashing seed, rerun alone for a clean signature and log.
            // The crashing seed is the last one started; the batch log ends with its crash.
            const st = statusSeed(io, tmp) orelse s;
            const bad = lastStart(out.log) orelse if (st >= s) st else s;
            report(sh, bad, out) catch |err| std.debug.print("[parse-fuzz] report failed: {t}\n", .{err});
            s = bad + 1;
        }
    }
}

pub fn fuzz(io: std.Io, opts: Options) !void {
    try std.Io.Dir.cwd().createDirPath(io, opts.out);
    var sh: Shared = .{ .io = io, .opts = opts, .next = opts.seed, .end = opts.seed + opts.count };
    var threads: std.ArrayList(std.Thread) = .empty;
    for (0..opts.jobs) |i| try threads.append(allocator, try std.Thread.spawn(.{}, worker, .{ &sh, i }));
    for (threads.items) |t| t.join();
    sh.done = true;
    std.debug.print("{s}", .{try writeSummary(&sh)});
}

/// Writes summary.txt; called with the mutex held while fuzzing.
fn writeSummary(sh: *Shared) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(allocator);
    const seeds = @min(sh.next, sh.end) - sh.opts.seed;
    try w.writer.print("target: {t}, seeds: {d}{s}, crashes: {d}, distinct signatures: {d}\n", .{ sh.opts.target, seeds, if (sh.done) "" else " (running)", sh.crashes, sh.seen.count() });
    var it = sh.seen.iterator();
    while (it.next()) |e| try w.writer.print("  {d:>6}  {s}\n", .{ e.value_ptr.*, e.key_ptr.* });
    try std.Io.Dir.cwd().writeFile(sh.io, .{ .sub_path = try std.fmt.allocPrint(allocator, "{s}/summary.txt", .{sh.opts.out}), .data = w.written() });
    return w.written();
}
