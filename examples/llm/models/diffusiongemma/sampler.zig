const zml = @import("zml");

const Generation = @import("Generation.zig");

pub const Settings = struct {
    max_steps: u32,
    t_min: f32,
    t_max: f32,
    entropy_bound: f32,
    confidence_threshold: f32,
    stability_threshold: u32,

    pub fn init(generation: Generation) Settings {
        return .{
            .max_steps = generation.max_denoising_steps,
            .t_min = generation.t_min,
            .t_max = generation.t_max,
            .entropy_bound = generation.sampler_config.entropy_bound,
            .confidence_threshold = generation.confidence_threshold,
            .stability_threshold = generation.stability_threshold,
        };
    }

    pub fn historyLength(self: Settings) u32 {
        return @max(self.stability_threshold, 1);
    }
};

pub const Noise = struct {
    gumbel: zml.Tensor,
    tokens: zml.Tensor,

    pub fn draw(rng: zml.Tensor.Rng, len: i64, vocab_size: i64) struct { Noise, zml.Tensor.Rng } {
        const after_gumbel, const gumbel = rng.gumbel(zml.Shape.init(.{ .s = len, .voc = vocab_size }, .f32));
        const shape = zml.Shape.init(.{ .s = len }, .f32);
        const after_tokens, const uniform = after_gumbel.uniform(shape, .{ .min = 0, .max = @floatFromInt(vocab_size) });

        return .{
            .{ .gumbel = gumbel, .tokens = uniform.floor().convert(.u32) },
            after_tokens,
        };
    }
};

pub const Step = struct {
    canvas: zml.Tensor,
    argmax: zml.Tensor,
    history: zml.Tensor,
    processed: zml.Tensor,
    entropy: zml.Tensor,
    accepted: zml.Tensor,
    done: zml.Tensor,
};

pub fn step(settings: Settings, logits: zml.Tensor, history: zml.Tensor, step_index: zml.Tensor, noise: Noise) Step {
    const progress = step_index.convert(.f32).scale(1.0 / @as(f32, @floatFromInt(settings.max_steps)));
    const temperature = progress.scale(settings.t_max - settings.t_min).addConstant(settings.t_min);
    const processed = logits.div(temperature.broad(logits.shape()));

    const sampled = processed.add(noise.gumbel).argMax(.voc).indices.squeeze(.voc).convert(.u32);
    const argmax = processed.argMax(.voc).indices.squeeze(.voc).convert(.u32);

    const per_position = entropy(processed);
    const accepted = acceptance(per_position, settings.entropy_bound);
    const canvas = accepted.select(sampled, noise.tokens);

    const mean_entropy = per_position.mean(.s).squeeze(.s);
    const confident = mean_entropy.cmp(.LT, .scalar(settings.confidence_threshold, .f32));

    var stable = zml.Tensor.scalar(true, .bool);
    var next_history = history;
    if (settings.stability_threshold > 0) {
        const mismatches = history.cmp(.NE, argmax.broad(history.shape())).convert(.u32);
        stable = mismatches.sum(.hist).sum(.s).squeeze(.hist).squeeze(.s).cmp(.EQ, .scalar(0, .u32));
        next_history = pushHistory(history, argmax);
    }

    return .{
        .canvas = canvas,
        .argmax = argmax,
        .history = next_history,
        .processed = processed,
        .entropy = per_position,
        .accepted = accepted,
        .done = stable.logical(.AND, confident),
    };
}

pub fn entropy(logits: zml.Tensor) zml.Tensor {
    const log_probabilities = logits.sub(logits.logSumExp(.voc).broad(logits.shape()));
    return log_probabilities.exp().mul(log_probabilities).sum(.voc).squeeze(.voc).negate();
}

pub fn acceptance(per_position: zml.Tensor, bound: f32) zml.Tensor {
    const sorted = per_position.sort(.s, .{});

    const before = prefixSum(sorted.values).sub(sorted.values);
    const selected = before.cmp(.LE, .scalar(bound, .f32)).rename(.{ .s = .rank });

    const rank = sorted.indices.argsort(.s, .{});
    return selected.gather(.{ .rank = rank }, .{});
}

fn prefixSum(x: zml.Tensor) zml.Tensor {
    const n = x.dim(.s);
    const pairs = zml.Shape.init(.{ .s = n, .j = n }, .i32);
    const lower = zml.Tensor.iota(pairs, .j).cmp(.LE, zml.Tensor.iota(pairs, .s)).convert(.f32);
    return lower.dot(x.convert(.f32).rename(.{ .s = .j }), .j).convert(x.dtype());
}

fn pushHistory(history: zml.Tensor, argmax: zml.Tensor) zml.Tensor {
    const newest = argmax.insertAxes(0, .{.hist});
    if (history.dim(.hist) == 1) {
        return newest;
    }

    return zml.Tensor.concatenate(&.{ history.slice(.hist, .{ .start = 1 }), newest }, .hist);
}
