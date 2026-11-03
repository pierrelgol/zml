const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const JsonValue = json.Value;
const ParseOptions = json.ParseOptions;
const ParseFromValueError = json.ParseFromValueError;

pub const Questions = json.ArrayHashMap(Question);
pub const Labels = []const []const u8;
pub const max_think = 4096;
pub const max_request_file_size = 16 * 1024 * 1024;
const QuestionMap = @FieldType(Questions, "map");

fn writeQuoted(writer: *Io.Writer, text: []const u8) Io.Writer.Error!void {
    return json.Stringify.encodeJsonString(text, .{}, writer);
}

fn writeLabels(writer: *Io.Writer, labels: Labels) Io.Writer.Error!void {
    try writer.writeByte('[');
    for (labels, 0..) |label, index| {
        if (index != 0) {
            try writer.writeAll(", ");
        }
        try writeQuoted(writer, label);
    }
    try writer.writeByte(']');
}

pub const Request = struct {
    model: ?[]const u8 = null,
    instructions: ?[]const u8 = null,
    seed: ?u64 = null,
    state: State,
    images: []const Image = &.{},
    questions: Questions,
    ask: ?Labels = null,
    samples: ?Samples = null,
    auto_max: ?u32 = null,
    auto_threshold: ?f32 = null,
    steps: ?u32 = null,
    think: ?u32 = null,
    chunk_prompt: ?ChunkPrompt = null,

    pub fn format(request: Request, writer: *Io.Writer) Io.Writer.Error!void {
        try writer.writeAll("Request {\n");

        if (request.model) |model| {
            try writer.writeAll("  model: ");
            try writeQuoted(writer, model);
            try writer.writeByte('\n');
        }

        if (request.instructions) |instructions| {
            try writer.writeAll("  instructions: ");
            try writeQuoted(writer, instructions);
            try writer.writeByte('\n');
        }

        if (request.seed) |seed| {
            try writer.print("  seed: {d}\n", .{seed});
        }

        try writer.print("  state: {f}\n", .{request.state});

        if (request.images.len != 0) {
            try writer.writeAll("  images:\n");
            for (request.images) |image| {
                try writer.print("    - {f}\n", .{image});
            }
        }

        const questions = &request.questions.map;
        try writer.writeAll("  questions:\n");
        for (questions.keys(), questions.values()) |question_id, question| {
            try writer.print("    {s}: {f}\n", .{ question_id, question });
        }

        if (request.ask) |ask| {
            try writer.writeAll("  ask: ");
            try writeLabels(writer, ask);
            try writer.writeByte('\n');
        }

        if (request.samples) |samples| {
            try writer.print("  samples: {f}\n", .{samples});
        }

        if (request.auto_max) |auto_max| {
            try writer.print("  auto_max: {d}\n", .{auto_max});
        }

        if (request.auto_threshold) |auto_threshold| {
            try writer.print("  auto_threshold: {d}\n", .{auto_threshold});
        }

        if (request.steps) |steps| {
            try writer.print("  steps: {d}\n", .{steps});
        }

        if (request.think) |think| {
            try writer.print("  think: {d}\n", .{think});
        }

        if (request.chunk_prompt) |chunk_prompt| {
            try writer.print("  chunk_prompt: {f}\n", .{chunk_prompt});
        }

        try writer.writeByte('}');
    }
};

pub const ChunkPrompt = enum {
    shared,

    pub fn format(chunk_prompt: ChunkPrompt, writer: *Io.Writer) Io.Writer.Error!void {
        return writer.writeAll(@tagName(chunk_prompt));
    }
};

pub const State = union(enum) {
    text: []const u8,
    fields: json.ArrayHashMap(JsonValue),

    pub fn jsonParse(allocator: Allocator, source: anytype, options: ParseOptions) !State {
        const value = try json.innerParse(JsonValue, allocator, source, options);
        return jsonParseFromValue(allocator, value, options);
    }

    pub fn jsonParseFromValue(allocator: Allocator, value: JsonValue, options: ParseOptions) ParseFromValueError!State {
        switch (value) {
            .string => |text| {
                return .{
                    .text = text,
                };
            },
            .object => {
                const fields = try json.innerParseFromValue(json.ArrayHashMap(JsonValue), allocator, value, options);
                return .{
                    .fields = fields,
                };
            },
            else => {
                return error.UnexpectedToken;
            },
        }
    }

    pub fn format(state: State, writer: *Io.Writer) Io.Writer.Error!void {
        switch (state) {
            .text => |text| try writeQuoted(writer, text),
            .fields => |fields| try json.Stringify.value(fields, .{}, writer),
        }
    }
};

pub const Samples = union(enum) {
    fixed: u32,
    auto,

    pub fn jsonParse(allocator: Allocator, source: anytype, options: ParseOptions) !Samples {
        const value = try json.innerParse(JsonValue, allocator, source, options);
        return jsonParseFromValue(allocator, value, options);
    }

    pub fn jsonParseFromValue(_: Allocator, value: JsonValue, _: ParseOptions) ParseFromValueError!Samples {
        switch (value) {
            .integer => |count| {
                const fixed_count = std.math.cast(u32, count) orelse {
                    return error.Overflow;
                };

                return .{
                    .fixed = fixed_count,
                };
            },
            .string => |text| {
                if (std.mem.eql(u8, text, "auto")) {
                    return .auto;
                }

                return error.UnexpectedToken;
            },
            else => {
                return error.UnexpectedToken;
            },
        }
    }

    pub fn format(samples: Samples, writer: *Io.Writer) Io.Writer.Error!void {
        switch (samples) {
            .fixed => |count| try writer.print("{d}", .{count}),
            .auto => try writer.writeAll("auto"),
        }
    }
};

pub const Image = union(enum) {
    data_url: []const u8,
    raw: Raw,

    pub const Raw = struct {
        content_type: []const u8,
        base64: []const u8,

        pub fn format(raw: Raw, writer: *Io.Writer) Io.Writer.Error!void {
            try writer.print("raw({s}, {d} base64 bytes)", .{ raw.content_type, raw.base64.len });
        }
    };

    pub fn format(image: Image, writer: *Io.Writer) Io.Writer.Error!void {
        switch (image) {
            .data_url => |url| {
                const comma_index = std.mem.indexOfScalar(u8, url, ',') orelse url.len;
                const header = url[0..comma_index];
                const payload_len = url.len - @min(url.len, comma_index + 1);
                try writer.print("data_url({s}, {d} bytes)", .{ header, payload_len });
            },
            .raw => |raw| try raw.format(writer),
        }
    }

    pub fn jsonParse(allocator: Allocator, source: anytype, options: ParseOptions) !Image {
        const value = try json.innerParse(JsonValue, allocator, source, options);
        return jsonParseFromValue(allocator, value, options);
    }

    pub fn jsonParseFromValue(allocator: Allocator, value: JsonValue, options: ParseOptions) ParseFromValueError!Image {
        switch (value) {
            .string => |url| {
                return .{
                    .data_url = url,
                };
            },
            .object => {
                const raw = try json.innerParseFromValue(Raw, allocator, value, options);
                return .{
                    .raw = raw,
                };
            },
            else => {
                return error.UnexpectedToken;
            },
        }
    }
};

pub const Question = struct {
    instructions: []const u8,
    depends_on: Labels = &.{},
    ask_if: ?json.ArrayHashMap(Labels) = null,
    alone: bool = false,
    kind: Kind,

    pub const Kind = union(enum) {
        noul: ?NoulCriteria,
        choice: ChoiceCriteria,
        score: Labels,

        pub fn format(kind: Kind, writer: *Io.Writer) Io.Writer.Error!void {
            try writer.writeAll(@tagName(kind));

            switch (kind) {
                .noul => |noul_criteria| {
                    if (noul_criteria) |criteria| {
                        try writer.print("{f}", .{criteria});
                    }
                },
                .choice => |choice_criteria| {
                    try writer.writeByte('[');

                    for (choice_criteria.map.keys(), choice_criteria.map.values(), 0..) |label, description, index| {
                        if (index != 0) {
                            try writer.writeAll(", ");
                        }

                        try writeQuoted(writer, label);
                        if (description) |text| {
                            try writer.writeAll(": ");
                            try writeQuoted(writer, text);
                        }
                    }

                    try writer.writeByte(']');
                },
                .score => |score_labels| {
                    try writeLabels(writer, score_labels);
                },
            }
        }
    };

    pub const KindTag = std.meta.Tag(Kind);

    pub const ChoiceCriteria = json.ArrayHashMap(?[]const u8);

    pub const NoulCriteria = struct {
        true: ?[]const u8 = null,
        false: ?[]const u8 = null,

        pub fn format(criteria: NoulCriteria, writer: *Io.Writer) Io.Writer.Error!void {
            try writer.writeByte('(');

            if (criteria.true) |text| {
                try writer.writeAll("true: ");
                try writeQuoted(writer, text);
            }

            if (criteria.false) |text| {
                if (criteria.true != null) {
                    try writer.writeAll(", ");
                }
                try writer.writeAll("false: ");
                try writeQuoted(writer, text);
            }

            try writer.writeByte(')');
        }
    };

    const CommonFields = struct {
        instructions: []const u8,
        depends_on: Labels = &.{},
        ask_if: ?json.ArrayHashMap(Labels) = null,
        alone: bool = false,
    };

    pub fn jsonParse(allocator: Allocator, source: anytype, options: ParseOptions) !Question {
        const value = try json.innerParse(JsonValue, allocator, source, options);
        return jsonParseFromValue(allocator, value, options);
    }

    pub fn jsonParseFromValue(allocator: Allocator, value: JsonValue, options: ParseOptions) ParseFromValueError!Question {
        if (value != .object) {
            return error.UnexpectedToken;
        }

        var remaining_fields = try value.object.clone(allocator);

        const type_entry = remaining_fields.fetchOrderedRemove("type") orelse {
            return error.MissingField;
        };

        const type_value = type_entry.value;

        var criteria_value: ?JsonValue = null;
        if (remaining_fields.fetchOrderedRemove("criteria")) |criteria_entry| {
            criteria_value = criteria_entry.value;
        }

        if (type_value != .string) {
            return error.UnexpectedToken;
        }

        const kind_tag = std.meta.stringToEnum(KindTag, type_value.string) orelse {
            return error.InvalidEnumTag;
        };

        const kind = try parseKind(allocator, kind_tag, criteria_value, options);

        const common = try json.innerParseFromValue(
            CommonFields,
            allocator,
            .{ .object = remaining_fields },
            options,
        );

        return .{
            .instructions = common.instructions,
            .depends_on = common.depends_on,
            .ask_if = common.ask_if,
            .alone = common.alone,
            .kind = kind,
        };
    }

    fn parseKind(allocator: Allocator, kind_tag: KindTag, criteria_value: ?JsonValue, options: ParseOptions) ParseFromValueError!Kind {
        switch (kind_tag) {
            .noul => {
                if (criteria_value) |criteria| {
                    const noul_criteria = try json.innerParseFromValue(?NoulCriteria, allocator, criteria, options);
                    return .{
                        .noul = noul_criteria,
                    };
                }
                return .{
                    .noul = null,
                };
            },
            .choice => {
                const criteria = criteria_value orelse {
                    return error.MissingField;
                };

                const choice_criteria = try json.innerParseFromValue(ChoiceCriteria, allocator, criteria, options);
                return .{
                    .choice = choice_criteria,
                };
            },
            .score => {
                const criteria = criteria_value orelse {
                    return error.MissingField;
                };
                const score_labels = try json.innerParseFromValue(Labels, allocator, criteria, options);
                return .{
                    .score = score_labels,
                };
            },
        }
    }

    pub fn format(question: Question, writer: *Io.Writer) Io.Writer.Error!void {
        try writer.print("{f} ", .{question.kind});
        try writeQuoted(writer, question.instructions);

        if (question.depends_on.len != 0) {
            try writer.writeAll(" depends_on=");
            try writeLabels(writer, question.depends_on);
        }

        if (question.ask_if) |ask_if| {
            try writer.writeAll(" ask_if={");
            for (ask_if.map.keys(), ask_if.map.values(), 0..) |dependency_id, required_labels, index| {
                if (index != 0) {
                    try writer.writeAll(", ");
                }
                try writer.print("{s}: ", .{dependency_id});
                try writeLabels(writer, required_labels);
            }
            try writer.writeByte('}');
        }

        if (question.alone) {
            try writer.writeAll(" alone");
        }
    }

    pub fn hasLabel(question: Question, label: []const u8) bool {
        switch (question.kind) {
            .noul => {
                return std.mem.eql(u8, label, "true") or std.mem.eql(u8, label, "false");
            },
            .choice => |choice_criteria| {
                return choice_criteria.map.contains(label);
            },
            .score => |score_labels| {
                return containsLabel(score_labels, label);
            },
        }
    }

    pub fn dependencyIterator(question: Question) DependencyIterator {
        return .{
            .question = question,
        };
    }
};

pub const DependencyIterator = struct {
    question: Question,
    position: usize = 0,

    pub fn next(iterator: *DependencyIterator) ?[]const u8 {
        defer iterator.position += 1;

        const explicit_dependencies = iterator.question.depends_on;
        if (iterator.position < explicit_dependencies.len) {
            return explicit_dependencies[iterator.position];
        }

        const ask_if = iterator.question.ask_if orelse {
            return null;
        };

        const ask_if_dependencies = ask_if.map.keys();
        const ask_if_position = iterator.position - explicit_dependencies.len;

        if (ask_if_position < ask_if_dependencies.len) {
            return ask_if_dependencies[ask_if_position];
        }

        return null;
    }
};

pub const ValidationError = error{
    EmptyQuestions,
    InvalidId,
    TooFewLabels,
    SamplesZero,
    ThinkTooLarge,
    ImageNotDataUrl,
    UnknownDependency,
    SelfDependency,
    DependencyCycle,
    BadAskIfValue,
    AskMissingDependency,
    UnknownQuestion,
};

pub fn validateRequest(gpa: Allocator, request: Request) (ValidationError || Allocator.Error)!void {
    const questions = &request.questions.map;

    if (questions.count() == 0) {
        return error.EmptyQuestions;
    }

    try validateSamples(request.samples);
    try validateThink(request.think);
    try validateImages(request.images);

    for (questions.keys(), questions.values()) |question_id, question| {
        try validateQuestionId(question_id);
        try validateQuestionLabels(question);
        try validateQuestionDependencies(questions, question_id, question);
    }

    try ensureNoDependencyCycles(gpa, questions);

    if (request.ask) |asked_question_ids| {
        try validateAskSubset(questions, asked_question_ids);
    }
}

fn validateSamples(samples: ?Samples) ValidationError!void {
    const configured_samples = samples orelse {
        return;
    };

    switch (configured_samples) {
        .fixed => |count| {
            if (count == 0) {
                return error.SamplesZero;
            }
        },
        .auto => {},
    }
}

fn validateThink(think: ?u32) ValidationError!void {
    const think_budget = think orelse {
        return;
    };

    if (think_budget > max_think) {
        return error.ThinkTooLarge;
    }
}

fn validateImages(images: []const Image) ValidationError!void {
    for (images) |image| {
        switch (image) {
            .data_url => |url| {
                if (!std.mem.startsWith(u8, url, "data:")) {
                    return error.ImageNotDataUrl;
                }
            },
            .raw => {},
        }
    }
}

fn validateQuestionId(question_id: []const u8) ValidationError!void {
    if (question_id.len == 0) {
        return error.InvalidId;
    }

    if (std.mem.indexOfScalar(u8, question_id, ':') != null) {
        return error.InvalidId;
    }
}

fn validateQuestionLabels(question: Question) ValidationError!void {
    switch (question.kind) {
        .noul => {},
        .choice => |choice_criteria| {
            if (choice_criteria.map.count() < 2) {
                return error.TooFewLabels;
            }
        },
        .score => |score_labels| {
            if (score_labels.len < 2) {
                return error.TooFewLabels;
            }
        },
    }
}

fn validateQuestionDependencies(questions: *const QuestionMap, question_id: []const u8, question: Question) ValidationError!void {
    for (question.depends_on) |dependency_id| {
        if (std.mem.eql(u8, dependency_id, question_id)) {
            return error.SelfDependency;
        }

        if (!questions.contains(dependency_id)) {
            return error.UnknownDependency;
        }
    }

    const ask_if = question.ask_if orelse {
        return;
    };

    for (ask_if.map.keys(), ask_if.map.values()) |dependency_id, required_labels| {
        if (std.mem.eql(u8, dependency_id, question_id)) {
            return error.SelfDependency;
        }

        const dependency = questions.get(dependency_id) orelse {
            return error.UnknownDependency;
        };

        for (required_labels) |label| {
            if (!dependency.hasLabel(label)) {
                return error.BadAskIfValue;
            }
        }
    }
}

fn validateAskSubset(questions: *const QuestionMap, asked_question_ids: Labels) ValidationError!void {
    for (asked_question_ids) |asked_question_id| {
        const asked_question = questions.get(asked_question_id) orelse {
            return error.UnknownQuestion;
        };

        var dependencies = asked_question.dependencyIterator();
        while (dependencies.next()) |dependency_id| {
            if (!containsLabel(asked_question_ids, dependency_id)) {
                return error.AskMissingDependency;
            }
        }
    }
}

const VisitState = enum(u8) {
    unvisited,
    in_progress,
    finished,
};

fn ensureNoDependencyCycles(gpa: Allocator, questions: *const QuestionMap) (ValidationError || Allocator.Error)!void {
    const visit_states = try gpa.alloc(VisitState, questions.count());
    defer gpa.free(visit_states);
    @memset(visit_states, .unvisited);

    for (0..questions.count()) |question_index| {
        try visitQuestionDepthFirst(questions, visit_states, question_index);
    }
}

fn visitQuestionDepthFirst(questions: *const QuestionMap, visit_states: []VisitState, question_index: usize) ValidationError!void {
    switch (visit_states[question_index]) {
        .finished => {
            return;
        },

        .in_progress => {
            return error.DependencyCycle;
        },

        .unvisited => {},
    }

    visit_states[question_index] = .in_progress;

    var dependencies = questions.values()[question_index].dependencyIterator();
    while (dependencies.next()) |dependency_id| {
        const dependency_index = questions.getIndex(dependency_id).?;
        try visitQuestionDepthFirst(questions, visit_states, dependency_index);
    }

    visit_states[question_index] = .finished;
}

fn containsLabel(labels: Labels, wanted: []const u8) bool {
    for (labels) |label| {
        if (std.mem.eql(u8, label, wanted)) {
            return true;
        }
    }

    return false;
}

pub fn parseRequest(gpa: Allocator, json_bytes: []const u8) !json.Parsed(Request) {
    const parsed_request = try json.parseFromSlice(Request, gpa, json_bytes, .{});
    errdefer parsed_request.deinit();
    try validateRequest(gpa, parsed_request.value);
    return parsed_request;
}

fn readExampleFile(comptime relative_path: []const u8) ![]u8 {
    return Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        "examples/" ++ relative_path,
        std.testing.allocator,
        .limited(max_request_file_size),
    );
}

fn expectValidExample(comptime relative_path: []const u8) !void {
    const file_contents = try readExampleFile(relative_path);
    defer std.testing.allocator.free(file_contents);

    const parsed_request = parseRequest(std.testing.allocator, file_contents) catch |parse_error| {
        std.debug.print("{s}: {t}\n", .{ relative_path, parse_error });
        return parse_error;
    };
    parsed_request.deinit();
}

fn expectInvalidExample(comptime relative_path: []const u8) !void {
    const file_contents = try readExampleFile(relative_path);
    defer std.testing.allocator.free(file_contents);

    if (parseRequest(std.testing.allocator, file_contents)) |parsed_request| {
        parsed_request.deinit();
        std.debug.print("{s}: unexpectedly valid\n", .{relative_path});
        return error.ShouldHaveFailed;
    } else |_| {}
}

test "valid examples are accepted" {
    const valid_example_names = .{
        "01-minimal-noul",
        "02-moderation-three-types",
        "03-dependencies-and-ask-if",
        "04-fixed-samples-steps-think",
        "05-image-and-ask-subset",
    };
    inline for (valid_example_names) |example_name| {
        try expectValidExample("valid/" ++ example_name ++ ".json");
    }
}

test "invalid examples are rejected" {
    const invalid_example_names = .{
        "01-missing-state",
        "02-empty-questions",
        "03-questions-is-array",
        "04-unknown-type",
        "05-choice-one-option",
        "06-choice-criteria-missing",
        "07-score-criteria-as-object",
        "08-id-with-colon",
        "09-think-too-large",
        "10-samples-zero",
        "11-image-not-data-url",
        "12-noul-criteria-wrong-shape",
        "13-semantic-unknown-dependency",
        "14-semantic-cycle",
        "15-semantic-ask-if-bad-value",
        "16-semantic-ask-missing-dependency",
        "17-semantic-self-dependency",
    };

    inline for (invalid_example_names) |example_name| {
        try expectInvalidExample("invalid/" ++ example_name ++ ".json");
    }
}

fn expectFormats(comptime relative_path: []const u8) !void {
    const gpa = std.testing.allocator;

    const file_contents = try readExampleFile(relative_path);
    defer gpa.free(file_contents);

    const parsed_request = try parseRequest(gpa, file_contents);
    defer parsed_request.deinit();

    var formatted: Io.Writer.Allocating = .init(gpa);
    defer formatted.deinit();
    try formatted.writer.print("{f}", .{parsed_request.value});
    try std.testing.expect(std.mem.startsWith(u8, formatted.written(), "Request {\n"));
    try std.testing.expect(std.mem.endsWith(u8, formatted.written(), "}"));
}

test "valid examples format" {
    const valid_example_names = .{
        "01-minimal-noul",
        "02-moderation-three-types",
        "03-dependencies-and-ask-if",
        "04-fixed-samples-steps-think",
        "05-image-and-ask-subset",
    };
    inline for (valid_example_names) |example_name| {
        try expectFormats("valid/" ++ example_name ++ ".json");
    }
}

test "format leaf types" {
    const gpa = std.testing.allocator;
    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();

    const samples: Samples = .auto;
    try out.writer.print("{f}", .{samples});
    try std.testing.expectEqualStrings("auto", out.written());
    out.clearRetainingCapacity();

    try out.writer.print("{f}", .{Samples{ .fixed = 3 }});
    try std.testing.expectEqualStrings("3", out.written());
    out.clearRetainingCapacity();

    const image: Image = .{ .raw = .{ .content_type = "image/png", .base64 = "AAAA" } };
    try out.writer.print("{f}", .{image});
    try std.testing.expectEqualStrings("raw(image/png, 4 base64 bytes)", out.written());
    out.clearRetainingCapacity();

    const question: Question = .{
        .instructions = "Is it \"spam\"?",
        .depends_on = &.{"topic"},
        .kind = .{ .score = &.{ "low", "high" } },
    };
    try out.writer.print("{f}", .{question});
    try std.testing.expectEqualStrings("score[\"low\", \"high\"] \"Is it \\\"spam\\\"?\" depends_on=[\"topic\"]", out.written());
}
