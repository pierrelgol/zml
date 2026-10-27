const std = @import("std");
const json = std.json;

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");
const inference = @import("inference.zig");
const session = @import("session.zig");

const log = std.log.scoped(.diffusiongemma);

pub const Config = DiffusionGemmaConfig;

pub const DiffusionGemmaConfig = struct {
    boi_token_id: u64,
    eoi_token_id: u64,
    eos_token_id: EosTokens,
    text_config: TextConfig,

    pub const EosTokens = union(enum) {
        int: u32,
        ints: []u32,

        const Helpers = stdx.json.UnionHelpers(@This());
        pub const jsonParse = Helpers.jsonParse;
        pub const jsonParseFromValue = Helpers.jsonParseFromValue;
        pub const jsonStringify = Helpers.jsonStringify;
    };

    pub const LayerType = enum {
        sliding_attention,
        full_attention,
    };

    pub const RopeParameters = struct {
        sliding_attention: zml.nn.RopeOpts.Scaling,
        full_attention: zml.nn.RopeOpts.Scaling,
    };

    pub const TextConfig = struct {
        model_type: enum { diffusion_gemma_text },
        vocab_size: u64,
        hidden_size: u64,
        intermediate_size: u64,
        num_hidden_layers: u64,
        num_attention_heads: u64,
        num_key_value_heads: u64,
        head_dim: u64,
        global_head_dim: u64,
        num_global_key_value_heads: u64,
        hidden_activation: []const u8,
        max_position_embeddings: u64,
        initializer_range: f32,
        rms_norm_eps: f32,
        pad_token_id: ?u64,
        eos_token_id: ?u64,
        bos_token_id: ?u64,
        tie_word_embeddings: bool,
        attention_bias: bool,
        attention_dropout: ?f32,
        sliding_window: u64,
        layer_types: []const LayerType,
        final_logit_softcapping: f32,
        use_bidirectional_attention: ?enum { all, vision },
        num_experts: u64,
        top_k_experts: u64,
        moe_intermediate_size: u64,
        rope_parameters: RopeParameters,
    };

    pub fn format(self: @This(), w: *std.Io.Writer) !void {
        try std.json.Stringify.value(self, .{}, w);
    }
};

const Options = struct {
    sampling_strategy: ?zml.nn.SamplingStrategy,
    max_seq_len: u32,

    pub fn format(self: @This(), w: *std.Io.Writer) !void {
        try std.json.Stringify.value(self, .{}, w);
    }
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: Options = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = @intCast(parsed_config.value.text_config.max_position_embeddings),
        };

        return .{
            .inner = try .init(allocator, store, parsed_config.value, options),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);

        var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
        errdefer self.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        try loader.load(io, Model, &self.inner, &buffers, store, .{ .progress = progress });
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadBuffers(_: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        Model.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        meshes: common.Meshes,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        const experts_dtype = self.inner.text_model.layers[0].experts.gate_up_proj.dtype();
        const moe_backend = zml.moe.Backend.auto(platform, null, experts_dtype) catch .stablehlo;
        log.info("Selected MoE backend: {}", .{moe_backend});

        const params: inference.CompilationParameters = .init(self.inner, self.parsed_config.value, @intCast(seqlen), moe_backend, meshes);
        return .init(allocator, io, platform, self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    text_model: TextModel,

    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        options: Options,
    ) !Model {
        return .{
            .text_model = try .init(
                allocator,
                store.withPrefix("model.decoder"),
                store.withPrefix("model.encoder.language_model"),
                config,
            ),
            .gen_opts = options.sampling_strategy orelse .{},
            .config = config,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.text_model.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        TextModel.unloadBuffers(&self.text_model, allocator);
    }

    fn lmHead(self: Model) LmHead {
        return .init(self);
    }

    pub fn forward(
        self: Model,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
        moe_backend: zml.moe.Backend,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        const hidden, const updated_kv_cache = self.text_model.forward(
            tokens.withPartialTags(.{.s}),
            token_index,
            kv_cache,
            moe_backend,
        );

        const sample = LmHead.forward(.{
            .lm_head = self.lmHead(),
            .hidden = hidden,
            .active_length = active_length,
            .rng = rng,
        });

        return .{ sample.tokens, updated_kv_cache, sample.rng };
    }
};

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    embed_scale: f32,
    layers: []TransformerLayer,
    norm: RmsNorm,
    layer_types: []const Config.LayerType,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        scalar_store: zml.io.TensorStore.View,
        config: Config,
    ) !TextModel {
        const text_config = config.text_config;
        const layers = try allocator.alloc(TransformerLayer, text_config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, text_config.layer_types, 0..) |*layer, layer_type, i| {
            layer.* = .init(
                store.withPrefix("layers").withLayer(i),
                scalar_store.withPrefix("layers").withLayer(i),
                config,
                layer_type,
            );
        }

        return .{
            .embed_tokens = .{
                .weight = store.createTensor(
                    "embed_tokens.weight",
                    .{
                        .voc,
                        .d,
                    },
                    .model,
                    .{
                        .voc = .replicated,
                        .d = .model,
                    },
                ),
            },
            .embed_scale = @sqrt(@as(f32, @floatFromInt(text_config.hidden_size))),
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), text_config.rms_norm_eps),
            .layer_types = text_config.layer_types,
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        RmsNorm.unloadBuffers(&self.norm);

        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }

        allocator.free(self.layers);
    }

    pub fn forward(
        self: TextModel,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        moe_backend: zml.moe.Backend,
    ) struct { zml.Tensor, KvCache } {
        var hidden = self.embed_tokens.forward(tokens)
            .scale(self.embed_scale)
            .withPartitioning(.model, .{ .d = .replicated });

        var updated_kv_cache = kv_cache;
        var num_sliding: u32 = 0;
        var num_full: u32 = 0;
        for (self.layers) |layer| {
            const kv_cache_index: u32 = switch (layer.self_attn.layer_type) {
                .sliding_attention => b: {
                    defer num_sliding += 1;
                    break :b num_sliding;
                },
                .full_attention => b: {
                    defer num_full += 1;
                    break :b num_full;
                },
            };

            const result = TransformerLayer.forward(.{
                .layer = layer,
                .hidden = hidden,
                .token_index = token_index,
                .kv_cache = updated_kv_cache,
                .kv_cache_index = .scalar(kv_cache_index, .u32),
                .moe_backend = moe_backend,
            });
            hidden = result.hidden;
            updated_kv_cache = result.kv_cache;
        }

        return .{ self.norm.forward(hidden), updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const LmHead = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    final_logit_softcapping: f32,
    gen_opts: zml.nn.SamplingStrategy,

    pub fn init(mdl: Model) LmHead {
        return .{
            .embed_tokens = mdl.text_model.embed_tokens,
            .final_logit_softcapping = mdl.config.text_config.final_logit_softcapping,
            .gen_opts = mdl.gen_opts,
        };
    }

    pub const Input = struct {
        lm_head: LmHead,
        hidden: zml.Tensor,
        active_length: zml.Tensor,
        rng: zml.Tensor.Rng,
    };

    pub const Output = struct {
        tokens: zml.Tensor,
        rng: zml.Tensor.Rng,
    };

    pub fn forward(input: Input) Output {
        const self = input.lm_head;
        const last_index = input.active_length.convert(.i32).addConstant(-1);
        const hidden = input.hidden
            .slice(.s, .dyn(last_index, 1))
            .withPartitioning(.model, .{ .d = .replicated });
        const logits = hidden.dot(self.embed_tokens.weight.convert(hidden.dtype()), .d).convert(.f32);
        const cap = self.final_logit_softcapping;
        const capped = logits.scale(1.0 / cap).tanh().scale(cap);
        const new_tokens, const new_rng = zml.nn.sampleTokens(capped, self.gen_opts, input.rng);

        return .{
            .tokens = new_tokens.convert(.u32),
            .rng = new_rng,
        };
    }
};

pub const TransformerLayer = struct {
    mlp: Mlp,
    router: Router,
    experts: Experts,
    layer_scalar: zml.Tensor,
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm_1: RmsNorm,
    post_feedforward_layernorm_2: RmsNorm,
    pre_feedforward_layernorm_2: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, scalar_store: zml.io.TensorStore.View, config: Config, layer_type: Config.LayerType) TransformerLayer {
        const eps = config.text_config.rms_norm_eps;
        return .{
            .mlp = .init(store.withPrefix("mlp")),
            .router = .init(store.withPrefix("router"), config),
            .experts = .init(store.withPrefix("experts")),
            .layer_scalar = scalar_store.createTensor("layer_scalar", .{.scalar}, .model, .{ .scalar = .replicated }),
            .input_layernorm = .init(store.withPrefix("input_layernorm"), eps),
            .self_attn = .init(store.withPrefix("self_attn"), config, layer_type),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), eps),
            .pre_feedforward_layernorm = .init(store.withPrefix("pre_feedforward_layernorm"), eps),
            .post_feedforward_layernorm = .init(store.withPrefix("post_feedforward_layernorm"), eps),
            .post_feedforward_layernorm_1 = .init(store.withPrefix("post_feedforward_layernorm_1"), eps),
            .post_feedforward_layernorm_2 = .init(store.withPrefix("post_feedforward_layernorm_2"), eps),
            .pre_feedforward_layernorm_2 = .init(store.withPrefix("pre_feedforward_layernorm_2"), eps),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        Mlp.unloadBuffers(&self.mlp);
        Router.unloadBuffers(&self.router);
        Experts.unloadBuffers(&self.experts);
        self.layer_scalar.deinit();
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        RmsNorm.unloadBuffers(&self.pre_feedforward_layernorm);
        RmsNorm.unloadBuffers(&self.post_feedforward_layernorm);
        RmsNorm.unloadBuffers(&self.post_feedforward_layernorm_1);
        RmsNorm.unloadBuffers(&self.post_feedforward_layernorm_2);
        RmsNorm.unloadBuffers(&self.pre_feedforward_layernorm_2);
    }

    pub const Input = struct {
        layer: TransformerLayer,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        kv_cache_index: zml.Tensor,
        moe_backend: zml.moe.Backend,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
        kv_cache: KvCache,
    };

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const x0 = input.hidden.withPartitioning(.model, .{ .d = .replicated });

        const attn, const updated_kv_cache = self.self_attn.forward(
            self.input_layernorm.forward(x0),
            input.token_index,
            input.kv_cache,
            input.kv_cache_index,
        );
        const x1 = x0.add(self.post_attention_layernorm.forward(attn));

        const dense = self.post_feedforward_layernorm_1.forward(self.mlp.forward(self.pre_feedforward_layernorm.forward(x1)));
        const routing = self.router.forward(x1);
        const sparse = self.post_feedforward_layernorm_2.forward(
            self.experts.forward(self.pre_feedforward_layernorm_2.forward(x1), routing, input.moe_backend),
        );
        const x2 = x1.add(self.post_feedforward_layernorm.forward(dense.add(sparse)));

        const layer_scalar = self.layer_scalar.asScalar().convert(x2.dtype()).broad(x2.shape());
        return .{
            .hidden = x2.mul(layer_scalar).withPartitioning(.model, .{ .d = .replicated }),
            .kv_cache = updated_kv_cache,
        };
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,

    v_proj: ?zml.nn.Linear,
    o_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,

    layer_type: Config.LayerType,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    sliding_window: i64,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_type: Config.LayerType) SelfAttn {
        const text_config = config.text_config;
        const eps = text_config.rms_norm_eps;
        const head_dim, const num_kv_heads, const rope_scaling = switch (layer_type) {
            .sliding_attention => .{
                text_config.head_dim,
                text_config.num_key_value_heads,
                text_config.rope_parameters.sliding_attention,
            },
            .full_attention => .{
                text_config.global_head_dim,
                text_config.num_global_key_value_heads,
                text_config.rope_parameters.full_attention,
            },
        };

        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model }), null, .d),
            .v_proj = if (store.maybeCreateTensor("v_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model })) |weight|
                .init(weight, null, .d)
            else
                null,
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .dout, .d }, .model, .{ .d = .model }), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), eps),
            .k_norm = .init(store.withPrefix("k_norm"), eps),
            .v_norm = .initNoScale(eps),
            .layer_type = layer_type,
            .num_heads = @intCast(text_config.num_attention_heads),
            .num_kv_heads = @intCast(num_kv_heads),
            .head_dim = @intCast(head_dim),
            .sliding_window = @intCast(text_config.sliding_window),
            .rope_opts = .{
                .layout = switch (layer_type) {
                    .sliding_attention => .real_im_pass,
                    .full_attention => .real_pass_im_pass,
                },
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        zml.Buffer.deinitAll(SelfAttn, self);
    }

    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        kv_cache_index: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        const x_qkv = x.withPartitioning(.model, .{ .d = .replicated });

        var q = self.q_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });
        var v = if (self.v_proj) |v_proj|
            v_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto })
        else
            k;

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        v = self.v_norm.forward(v.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };
        q = zml.nn.rope(q, pos_index, self.rope_opts).rename(.{ .s = .q });
        k = zml.nn.rope(k, pos_index, self.rope_opts).rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const layer_cache = switch (self.layer_type) {
            .sliding_attention => kv_cache.sliding,
            .full_attention => kv_cache.full,
        };

        const new_layer_cache = layer_cache.updateAt(k, v, token_index, kv_cache_index);
        k = new_layer_cache.keysAt(kv_cache_index).convert(dtype);
        v = new_layer_cache.valuesAt(kv_cache_index).convert(dtype);

        const seq_len = k.dim(.k);
        const window: ?u32 = switch (self.layer_type) {
            .sliding_attention => @intCast(self.sliding_window),
            .full_attention => null,
        };

        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, dtype, window);
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()), token_index.appendAxes(.{.coord}), .{});

        const last_position = token_index.convert(.i32).addConstant(q.dim(.q) - 1);
        const written = zml.Tensor.iota(k.shape(), .k).cmp(.LE, last_position.broad(k.shape().withDtype(.i32)));
        k = written.select(k, .zeroes(k.shape()));
        v = written.select(v, .zeroes(v.shape()));

        const attn_output = zml.nn.sdpa(q, k, v, .{
            .attn_mask = attn_mask,
            .scale = .scalar(1.0, dtype),
        });

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn, attn.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(.model, .{ .d = .replicated });

        var new_kv_cache = kv_cache;
        switch (self.layer_type) {
            .sliding_attention => new_kv_cache.sliding = new_layer_cache,
            .full_attention => new_kv_cache.full = new_layer_cache,
        }
        return .{
            delta,
            new_kv_cache,
        };
    }
};

pub const Mlp = struct {
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model }), null, .d),
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .model, .{ .d = .model }), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        zml.Buffer.deinitAll(Mlp, self);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const gate = self.gate_proj.forward(x, x.dtype());
        const up = self.up_proj.forward(x, x.dtype());
        const activated = gate.gelu().mul(up).rename(.{ .dout = .d });
        return self.down_proj.forward(activated, activated.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(.model, .{ .d = .replicated });
    }
};

pub const Router = struct {
    norm: RmsNorm,
    proj: zml.nn.Linear,
    scale: zml.Tensor,
    per_expert_scale: zml.Tensor,
    top_k: u32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Router {
        return .{
            .norm = .initNoScale(config.text_config.rms_norm_eps),
            .proj = .init(store.createTensor("proj.weight", .{ .expert, .d }, .replicated, .{ .expert = .replicated, .d = .replicated }), null, .d),
            .scale = store.createTensor("scale", .{.d}, .replicated, .{ .d = .replicated }),
            .per_expert_scale = store.createTensor("per_expert_scale", .{.expert}, .replicated, .{ .expert = .replicated }),
            .top_k = @intCast(config.text_config.top_k_experts),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        zml.Buffer.deinitAll(Router, self);
    }

    pub const Output = struct {
        weights: zml.Tensor,
        indices: zml.Tensor,
    };

    pub fn forward(self: Router, x: zml.Tensor) Output {
        const root_size = 1.0 / @sqrt(@as(f32, @floatFromInt(x.dim(.d))));
        const normalized = self.norm.forward(x);
        const scaled = normalized.mul(self.scale.convert(normalized.dtype()).broad(normalized.shape())).scale(root_size);

        const probs = self.proj.forward(scaled, scaled.dtype()).convert(.f32).softmax(.expert);
        const top_k = probs.topK(.{ .topk = .expert }, self.top_k, .{});
        const indices = top_k.indices.convert(.i32);

        const renormalized = top_k.values.div(top_k.values.sum(.topk).broad(top_k.values.shape()));
        const per_expert_scale = self.per_expert_scale.convert(.f32).gather(.{ .expert = indices }, .{});

        return .{
            .weights = renormalized.mul(per_expert_scale),
            .indices = indices,
        };
    }
};

pub const Experts = struct {
    gate_up_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Experts {
        return .{
            .gate_up_proj = store.createTensor(
                "gate_up_proj",
                .{
                    .expert,
                    .dout,
                    .d,
                },
                .experts,
                .{
                    .expert = .experts,
                    .dout = .replicated,
                    .d = .replicated,
                },
            ),
            .down_proj = store.createTensor(
                "down_proj",
                .{
                    .expert,
                    .d,
                    .dout,
                },
                .experts,
                .{
                    .expert = .experts,
                    .d = .replicated,
                    .dout = .replicated,
                },
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Experts)) void {
        zml.Buffer.deinitAll(Experts, self);
    }

    pub fn forward(self: Experts, x: zml.Tensor, routing: Router.Output, moe_backend: zml.moe.Backend) zml.Tensor {
        return zml.moe.forwardMoe(
            x.rename(.{ .s = .t }),
            routing.indices.rename(.{ .s = .t }),
            routing.weights.rename(.{ .s = .t }),
            .init(self.gate_up_proj, null, .d),
            .init(self.down_proj, null, .dout),
            moe_backend,
            .{
                .activation = .geglu_tanh,
                .quantize_input = false,
                .routing_weight_placement = .after_down,
            },
        ).rename(.{ .t = .s });
    }
};

pub const RmsNorm = struct {
    weight: ?zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .model, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn initNoScale(eps: f32) RmsNorm {
        return .{
            .weight = null,
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        zml.Buffer.deinitAll(RmsNorm, self);
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps);
        const scaled = if (self.weight) |weight| normalized.mul(weight.convert(.f32).broad(normalized.shape())) else normalized;
        return scaled.convert(x.dtype());
    }
};

pub const KvCache = struct {
    sliding: LayerCache,
    full: LayerCache,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(config: Config, seqlen: u32, dtype: zml.DataType, mesh: *const zml.Mesh) KvCache {
        const text_config = config.text_config;
        var num_sliding: i64 = 0;
        var num_full: i64 = 0;

        for (text_config.layer_types) |layer_type| switch (layer_type) {
            .sliding_attention => num_sliding += 1,
            .full_attention => num_full += 1,
        };

        return .{
            .sliding = .init(
                .init(
                    .{
                        .layer = num_sliding,
                        .k = seqlen,
                        .h = text_config.num_key_value_heads,
                        .hd = text_config.head_dim,
                    },
                    dtype,
                ),
                mesh,
            ),
            .full = .init(
                .init(
                    .{
                        .layer = num_full,
                        .k = seqlen,
                        .h = text_config.num_global_key_value_heads,
                        .hd = text_config.global_head_dim,
                    },
                    dtype,
                ),
                mesh,
            ),
        };
    }

    pub fn initBuffer(kv: KvCache, io: std.Io, platform: *const zml.Platform) !Buffer {
        var sliding = try kv.sliding.initBuffer(io, platform);
        errdefer LayerCache.deinitBuffer(&sliding);

        return .{
            .sliding = sliding,
            .full = try kv.full.initBuffer(io, platform),
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        LayerCache.deinitBuffer(&kv.sliding);
        LayerCache.deinitBuffer(&kv.full);
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .sliding = kv.sliding.reuseBuffer(other.sliding),
            .full = kv.full.reuseBuffer(other.full),
        };
    }

    pub const LayerCache = struct {
        k: zml.Tensor,
        v: zml.Tensor,

        pub fn init(kv_shape: zml.Shape, mesh: *const zml.Mesh) LayerCache {
            const partitions = mesh.numPartitionsForLogicalAxis(.model);
            const sharded_shape = if (@rem(kv_shape.dim(.h), partitions) == 0)
                kv_shape.withPartitioning(mesh, .{ .h = .model })
            else
                kv_shape.withPartitioning(mesh, .{ .h = .replicated });

            return .{
                .k = .fromShape(sharded_shape),
                .v = .fromShape(sharded_shape),
            };
        }

        pub fn initBuffer(kv: LayerCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(LayerCache) {
            var k = try zml.Buffer.uninitialized(io, platform, kv.k.shape(), .{});
            errdefer k.deinit();

            return .{
                .k = k,
                .v = try zml.Buffer.uninitialized(io, platform, kv.v.shape(), .{}),
            };
        }

        pub fn deinitBuffer(kv: *zml.Bufferized(LayerCache)) void {
            kv.k.deinit();
            kv.v.deinit();
        }

        pub fn keysAt(kv: LayerCache, layer_index: zml.Tensor) zml.Tensor {
            return kv.k.slice(.layer, .dynSingle(layer_index));
        }

        pub fn valuesAt(kv: LayerCache, layer_index: zml.Tensor) zml.Tensor {
            return kv.v.slice(.layer, .dynSingle(layer_index));
        }

        pub fn updateAt(
            kv: LayerCache,
            new_k: zml.Tensor,
            new_v: zml.Tensor,
            token_index: zml.Tensor,
            layer_index: zml.Tensor,
        ) LayerCache {
            return .{
                .k = kv.k.scatterSlices(.{ .layer = layer_index, .k = token_index }, new_k.convert(kv.k.dtype()).transpose(kv.k.shape().drop(.layer)), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
                .v = kv.v.scatterSlices(.{ .layer = layer_index, .k = token_index }, new_v.convert(kv.v.dtype()).transpose(kv.v.shape().drop(.layer)), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
            };
        }

        pub fn reuseBuffer(kv: LayerCache, other: LayerCache) LayerCache {
            return .{
                .k = kv.k.reuseBuffer(other.k),
                .v = kv.v.reuseBuffer(other.v),
            };
        }
    };
};
