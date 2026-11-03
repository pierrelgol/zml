const std = @import("std");
const json = std.json;
const mem = std.mem;
const Allocator = mem.Allocator;

const inference = @import("inference.zig");
const Session = @import("Session.zig");
const systemone = @import("systemone.zig");

const log = std.log.scoped(.diffusiongemma);

const max_samples = 32;
const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const digits = "123456789";
const scaffold_text = "<|channel>thought\n<channel|>";
const pad_token = 0;

const Format = enum {
    lines,
    indexed,

    fn join(format: Format) []const u8 {
        return switch (format) {
            .lines => "\n",
            .indexed => " ",
        };
    }

    fn reply(format: Format) []const u8 {
        return switch (format) {
            .lines => "Reply with one line per question, in this order, formatted as \"id: label\".",
            .indexed => "Reply on one line with each question's id immediately followed by its label, separated by single spaces.",
        };
    }
};

const Item = struct {
    id: []const u8,
    question: systemone.Question,
    /// What ask_if compares against and what the answer reports.
    names: []const []const u8,
    descriptions: []const ?[]const u8,
    /// The single token written in the answer slot for each alternative.
    labels: []const []const u8,

    fn init(arena: Allocator, id: []const u8, question: systemone.Question) !Item {
        var item: Item = .{ .id = id, .question = question, .names = &.{}, .descriptions = &.{}, .labels = &.{} };
        switch (question.kind) {
            .noul => |noul_criteria| {
                const criteria = noul_criteria orelse systemone.Question.NoulCriteria{};
                item.names = &.{ "true", "false" };
                item.descriptions = try arena.dupe(?[]const u8, &.{ criteria.true, criteria.false });
                item.labels = &.{ "yes", "no" };
            },
            .choice => |choice_criteria| {
                item.names = choice_criteria.map.keys();
                item.descriptions = choice_criteria.map.values();
                item.labels = try singleChars(arena, letters, item.names.len);
            },
            .score => |levels| {
                const descriptions = try arena.alloc(?[]const u8, levels.len);
                @memset(descriptions, null);
                item.names = levels;
                item.descriptions = descriptions;
                item.labels = try singleChars(arena, if (levels.len <= digits.len) digits else letters, levels.len);
            },
        }
        return item;
    }
};

fn singleChars(arena: Allocator, alphabet: []const u8, count: usize) ![]const []const u8 {
    if (count > alphabet.len) {
        return error.TooManyAlternatives;
    }
    const out = try arena.alloc([]const u8, count);
    for (out, 0..) |*label, index| {
        label.* = alphabet[index..][0..1];
    }
    return out;
}

const Answer = struct {
    probabilities: []const f64,
    top: usize,
};

const Slot = struct {
    pos: usize,
    label_ids: []const u32,
};

const Template = struct {
    tokens: []const u32,
    slots: []const Slot,
};

const Decision = struct {
    arena: Allocator,
    session: *Session,
    items: []const Item,
    answers: []?Answer,
    format: Format,
    samples: systemone.Samples,
    auto_max: u32,
    auto_threshold: f32,
    random: std.Random,
    chat: []const u32,
    scaffold: []const u32,
    output_tokens: usize = 0,

    fn encode(self: *Decision, text: []const u8) ![]u32 {
        var encoder = try self.session.text_tokenizer.encoder();
        defer encoder.deinit();
        return encoder.encodeAlloc(self.arena, text);
    }

    fn answerText(self: *Decision, group: []const usize, picks: []const usize) ![]const u8 {
        var out: std.Io.Writer.Allocating = .init(self.arena);
        for (group, picks, 0..) |item_index, pick, index| {
            const item = self.items[item_index];
            if (index != 0) {
                try out.writer.writeAll(self.format.join());
            }
            switch (self.format) {
                .lines => try out.writer.print("{s}: {s}", .{ item.id, item.labels[pick] }),
                .indexed => try out.writer.print("{s}{s}", .{ item.id, item.labels[pick] }),
            }
        }
        return out.written();
    }

    fn templateTokens(self: *Decision, group: []const usize, picks: []const usize, head: []const u32, lead: []const u8) ![]const u32 {
        const text = try mem.concat(self.arena, u8, &.{ lead, try self.answerText(group, picks) });
        return mem.concat(self.arena, u32, &.{ head, try self.encode(text) });
    }

    fn resolve(self: *Decision, group: []const usize, head: []const u32, lead: []const u8) !Template {
        const picks = try self.arena.alloc(usize, group.len);
        @memset(picks, 0);

        const base = try self.templateTokens(group, picks, head, lead);

        if (base.len + 1 > self.session.shapes.canvas_length) {
            log.err("Answer template is {d} tokens, the canvas holds {d}", .{ base.len, self.session.shapes.canvas_length - 1 });
            return error.TemplateTooLong;
        }

        const slots = try self.arena.alloc(Slot, group.len);
        for (group, slots, 0..) |item_index, *slot, question_index| {
            const labels = self.items[item_index].labels;
            const ids = try self.arena.alloc(u32, labels.len);
            var pos: ?usize = null;

            for (1..labels.len) |label_index| {
                picks[question_index] = label_index;
                defer picks[question_index] = 0;
                const tokens = try self.templateTokens(group, picks, head, lead);
                if (tokens.len != base.len) {
                    return error.LabelNotSingleToken;
                }
                var diff: ?usize = null;
                for (tokens, base, 0..) |a, b, index| {
                    if (a == b) continue;
                    if (diff != null) return error.LabelsDoNotShareSlot;
                    diff = index;
                }
                const changed = diff orelse return error.LabelsDoNotShareSlot;
                if (pos != null and pos.? != changed) {
                    return error.LabelsDoNotShareSlot;
                }
                pos = changed;
                ids[label_index] = tokens[changed];
            }
            ids[0] = base[pos.?];
            for (ids, 0..) |id, index| {
                if (mem.indexOfScalar(u32, ids[index + 1 ..], id) != null) {
                    return error.DuplicateLabelToken;
                }
            }
            slot.* = .{
                .pos = pos.?,
                .label_ids = ids,
            };
        }
        return .{
            .tokens = base,
            .slots = slots,
        };
    }

    fn runGroup(self: *Decision, group: []const usize, earlier: []const []const u8) ![]const u8 {
        const session = self.session;
        const canvas_length = session.shapes.canvas_length;
        const width = inference.Read.max_labels;

        var prompt = self.chat;
        var head = self.scaffold;
        var lead: []const u8 = "";

        if (earlier.len != 0) {
            const join = self.format.join();
            const lines = try mem.join(self.arena, join, earlier);
            prompt = try mem.concat(self.arena, u32, &.{ self.chat, self.scaffold, try self.encode(lines) });
            head = &.{};
            lead = join;
        }

        if (prompt.len + canvas_length > session.shapes.seqlen) {
            return error.PromptTooLong;
        }

        const template = try self.resolve(group, head, lead);
        try session.runPrefill(prompt);

        const labels = try self.arena.alloc(u32, canvas_length * width);
        @memset(labels, 0);
        for (template.slots) |slot| {
            @memcpy(labels[slot.pos * width ..][0..slot.label_ids.len], slot.label_ids);
        }

        const canvas = try self.arena.alloc(u32, canvas_length);
        const log_probs = try self.arena.alloc(f32, canvas_length * width);
        const entropy = try self.arena.alloc(f32, canvas_length);
        const sums = try self.arena.alloc([]f64, group.len);
        for (sums, group) |*sum, item_index| {
            sum.* = try self.arena.alloc(f64, self.items[item_index].labels.len);
            @memset(sum.*, 0);
        }

        var reads: u32 = switch (self.samples) {
            .fixed => |count| @min(count, max_samples),
            .auto => 1,
        };
        var read_index: u32 = 0;
        while (read_index < reads) : (read_index += 1) {
            @memset(canvas, pad_token);
            @memcpy(canvas[0..template.tokens.len], template.tokens);
            canvas[template.tokens.len] = try session.tokenId("<turn|>");
            for (template.slots) |slot| {
                canvas[slot.pos] = self.random.uintLessThan(u32, session.config.text_config.vocab_size);
            }

            try session.read(canvas, @intCast(prompt.len), labels, log_probs, entropy);

            var max_entropy: f32 = 0;
            for (template.slots, sums) |slot, sum| {
                addSoftmax(sum, log_probs[slot.pos * width ..][0..sum.len]);
                max_entropy = @max(max_entropy, entropy[slot.pos]);
            }
            if (self.samples == .auto and read_index == 0 and max_entropy > self.auto_threshold) {
                reads = self.auto_max;
            }
        }

        const picks = try self.arena.alloc(usize, group.len);
        for (group, sums, picks) |item_index, sum, *pick| {
            for (sum) |*value| {
                value.* /= @floatFromInt(reads);
            }
            pick.* = 0;
            for (sum, 0..) |value, index| {
                if (value > sum[pick.*]) pick.* = index;
            }
            self.answers[item_index] = .{ .probabilities = sum, .top = pick.* };
        }
        self.output_tokens += template.tokens.len + 1;

        const line = try self.answerText(group, picks);
        log.info("Decision read: {s} ({d} reads)", .{ line, reads });
        return line;
    }

    fn askIfHolds(self: *const Decision, item: Item) bool {
        const ask_if = item.question.ask_if orelse return true;
        for (ask_if.map.keys(), ask_if.map.values()) |dependency_id, wanted| {
            const dependency = self.indexOf(dependency_id) orelse return false;
            const answer = self.answers[dependency] orelse return false;
            const name = self.items[dependency].names[answer.top];
            for (wanted) |label| {
                if (mem.eql(u8, label, name)) break;
            } else return false;
        }
        return true;
    }

    fn indexOf(self: *const Decision, id: []const u8) ?usize {
        for (self.items, 0..) |item, index| {
            if (mem.eql(u8, item.id, id)) return index;
        }
        return null;
    }
};

fn addSoftmax(sum: []f64, log_probs: []const f32) void {
    const max = mem.max(f32, log_probs);
    var total: f64 = 0;
    for (log_probs) |value| {
        total += @exp(@as(f64, value - max));
    }
    for (sum, log_probs) |*out, value| {
        out.* += @exp(@as(f64, value - max)) / total;
    }
}

fn systemText(arena: Allocator, instructions: ?[]const u8, items: []const Item, format: Format) ![]const u8 {
    const ws = &std.ascii.whitespace;

    var out: std.Io.Writer.Allocating = .init(arena);
    const writer = &out.writer;

    try writer.writeAll("Answer a fixed set of questions about the state the user provides. " ++
        "Each question lists its allowed answers; reply with exactly one label per question.\n");

    if (instructions) |text| {
        try writer.print("\n{s}\n", .{mem.trim(u8, text, ws)});
    }

    for (items) |item| {
        try writer.print("\nQuestion {s}: {s}\n", .{ item.id, mem.trim(u8, item.question.instructions, ws) });

        for (item.names, item.descriptions, item.labels) |name, description, label| {
            if (item.question.kind == .noul) {
                if (description) |text| {
                    try writer.print("  {s}: {s}\n", .{ label, mem.trim(u8, text, ws) });
                } else {
                    try writer.print("  {s}\n", .{label});
                }
            } else if (description) |text| {
                try writer.print("  {s}: {s} ({s})\n", .{ label, name, mem.trim(u8, text, ws) });
            } else {
                try writer.print("  {s}: {s}\n", .{ label, name });
            }
        }
    }
    try writer.print("\n{s}", .{format.reply()});
    return out.written();
}

fn chatPrompt(decision: *Decision, system: []const u8, state: []const u8) ![]const u32 {
    const session = decision.session;
    const start_of_turn = try session.tokenId("<|turn>");
    const end_of_turn = try session.tokenId("<turn|>");
    const newline = try session.tokenId("\n");
    const ws = &std.ascii.whitespace;

    return mem.concat(
        decision.arena,
        u32,
        &.{
            &.{ session.config.text_config.bos_token_id, start_of_turn },
            try decision.encode("system"),
            &.{newline},
            try decision.encode(mem.trim(u8, system, ws)),
            &.{ end_of_turn, newline, start_of_turn },
            try decision.encode("user"),
            &.{newline},
            try decision.encode(mem.trim(u8, state, ws)),
            &.{ end_of_turn, newline, start_of_turn },
            try decision.encode("model"),
            &.{newline},
        },
    );
}

pub fn run(session: *Session, allocator: Allocator, request: systemone.Request, writer: *std.Io.Writer) !void {
    if ((request.steps orelse 1) != 1 or (request.think orelse 0) != 0 or request.images.len != 0) {
        log.err("Decision mode supports one step, no think budget and no images", .{});
        return error.UnsupportedDecisionOption;
    }

    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const questions = &request.questions.map;
    var items: std.ArrayList(Item) = .empty;

    for (questions.keys(), questions.values()) |id, question| {
        if (request.ask) |ask| {
            for (ask) |asked| {
                if (mem.eql(u8, asked, id)) {
                    break;
                }
            } else {
                continue;
            }
        }

        try items.append(arena, try .init(arena, id, question));
    }

    const answers = try arena.alloc(?Answer, items.items.len);
    @memset(answers, null);

    var prng: std.Random.DefaultPrng = .init(request.seed orelse 42);

    var decision: Decision = .{
        .arena = arena,
        .session = session,
        .items = items.items,
        .answers = answers,
        .format = if (questions.count() <= 10) .lines else .indexed,
        .samples = request.samples orelse .auto,
        .auto_max = std.math.clamp(request.auto_max orelse 4, 1, max_samples),
        .auto_threshold = request.auto_threshold orelse 0.1,
        .random = prng.random(),
        .chat = &.{},
        .scaffold = &.{},
    };

    const state = switch (request.state) {
        .text => |text| text,
        .fields => |fields| try json.Stringify.valueAlloc(arena, fields, .{}),
    };

    decision.chat = try chatPrompt(&decision, try systemText(arena, request.instructions, items.items, decision.format), state);
    decision.scaffold = try decision.encode(scaffold_text);

    const done = try arena.alloc(bool, items.items.len);
    @memset(done, false);
    const ready = try arena.alloc(bool, items.items.len);
    var lines: std.ArrayList([]const u8) = .empty;
    var remaining = items.items.len;

    while (remaining != 0) {
        for (items.items, ready, done) |item, *is_ready, is_done| {
            is_ready.* = !is_done;
            var dependencies = item.question.dependencyIterator();
            while (dependencies.next()) |dependency_id| {
                const dependency = decision.indexOf(dependency_id) orelse continue;
                if (!done[dependency]) is_ready.* = false;
            }
        }

        const earlier = lines.items.len;
        var group: std.ArrayList(usize) = .empty;

        for (items.items, ready, 0..) |item, is_ready, index| {
            if (!is_ready or !decision.askIfHolds(item)) {
                continue;
            }

            if (item.question.alone) {
                try lines.append(arena, try decision.runGroup(&.{index}, lines.items[0..earlier]));
            } else {
                try group.append(arena, index);
            }
        }

        if (group.items.len != 0) {
            try lines.append(arena, try decision.runGroup(group.items, lines.items[0..earlier]));
        }

        const before = remaining;
        for (ready, done) |is_ready, *is_done| {
            if (!is_ready) {
                continue;
            }

            is_done.* = true;
            remaining -= 1;
        }

        if (remaining == before) {
            return error.DependencyCycle;
        }
    }

    try writeResponse(&decision, request.model orelse "diffusiongemma", writer);
}

fn writeResponse(decision: *const Decision, model: []const u8, writer: *std.Io.Writer) !void {
    var jw: json.Stringify = .{ .writer = writer, .options = .{ .whitespace = .indent_2 } };
    var key: [16]u8 = undefined;

    try jw.beginObject();
    try jw.objectField("model");
    try jw.write(model);
    try jw.objectField("answers");
    try jw.beginObject();

    for (decision.items, decision.answers) |item, maybe_answer| {
        try jw.objectField(item.id);

        const answer = maybe_answer orelse {
            try jw.write(null);
            continue;
        };

        const probabilities = answer.probabilities;
        try jw.beginObject();
        try jw.objectField("type");
        try jw.write(@tagName(item.question.kind));

        switch (item.question.kind) {
            .noul => {
                try jw.objectField("noul");
                try jw.write(probabilities[0]);
            },
            .choice => {
                try jw.objectField("choice");
                try jw.write(item.names[answer.top]);
                try jw.objectField("probabilities");
                try jw.beginObject();

                for (item.names, probabilities) |name, probability| {
                    try jw.objectField(name);
                    try jw.write(probability);
                }

                try jw.endObject();
            },
            .score => {
                var score: f64 = 0;
                for (probabilities, 0..) |probability, index| {
                    score += @as(f64, @floatFromInt(index)) * probability;
                }

                try jw.objectField("score");
                try jw.write(score);
                try jw.objectField("legend");
                try jw.beginObject();

                for (item.names, 0..) |name, index| {
                    try jw.objectField(try std.fmt.bufPrint(&key, "{d}", .{index}));
                    try jw.write(name);
                }

                try jw.endObject();
                try jw.objectField("probabilities");
                try jw.beginObject();

                for (probabilities, 0..) |probability, index| {
                    try jw.objectField(try std.fmt.bufPrint(&key, "{d}", .{index}));
                    try jw.write(probability);
                }

                try jw.endObject();
            },
        }

        if (item.question.kind != .noul) {
            try jw.objectField("confidence");
            try jw.write(probabilities[answer.top]);
        }

        try jw.endObject();
    }

    try jw.endObject();
    try jw.objectField("usage");
    try jw.beginObject();
    try jw.objectField("input_tokens");
    try jw.write(decision.chat.len);
    try jw.objectField("output_tokens");
    try jw.write(decision.output_tokens);
    try jw.endObject();
    try jw.endObject();
    try writer.writeByte('\n');
}
