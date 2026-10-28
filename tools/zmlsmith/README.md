# zmlsmith

A Csmith-style random program generator and differential tester for the ZML API. It
generates valid ZML programs, runs them on a platform, and compares every result with
an independent reference interpreter. Failures are deduplicated, reduced automatically
and turned into standalone Zig regression tests.

```
bazel build //tools/zmlsmith
bazel-bin/tools/zmlsmith/zmlsmith fuzz --seed 1 --count 2000 --jobs 12 --out /tmp/zs
```

## How it works

1. **Generation** (`gen.zig`). A program is a list of nodes (`ir.zig`), each with the
   shape, tags and dtype the generator predicts ZML returns. The generator runs the
   reference interpreter while it builds the program, so it can look at concrete values:
   ops whose result would be non-finite, imprecise, or on a discontinuity of an imprecise
   input (floor, compare, argmax, sort indices) are rejected and another op is tried.
   Programs are valid by construction.
2. **Lowering** (`lower.zig`). The program is traced through the public ZML API, and every
   op's inferred shape and tags are checked against the prediction (the shape oracle).
3. **Execution** (`runner.zig`). The executable runs twice; the outputs must match the
   interpreter within a propagated error bound (zero for integers and exact floats, so
   most comparisons are bit exact), and the two runs must agree bit for bit. On a
   mismatch the program is recompiled returning every node, to report the earliest
   diverging op.
   Families that track per element bounds (`Value.errs`: unary, binary, convert) are held
   to them: a correctly rounded op (`+ - * /`, sqrt, abs, negate, convert) on exact inputs
   must be bit exact, sign of zero included; imprecise inputs propagate their own bound
   plus one ulp of that element. bf16/f16 consumers allow half an ulp for the excess
   precision of CPU (computed in f32, rounding skipped inside fusions). Documented XLA
   deviations are modeled: `x / broadcast(y)` is computed as `x * (1/y)`, e8m0 ties round
   up, a rounded product feeding an add or subtract may be fused into an fma. Pure index maps (`kit.remap`: transpose, broadcast, slices, pad, roll, ...) carry
   the bounds through, and reductions, cumsum, dot, conv and reduceWindow compute one per
   output lane, scatter one per element, the nn families one per element, lane or query row. Emitted tests
   check the same per element bounds.
4. **Isolation** (`supervisor.zig`, `child.zig`). Workers are child processes running
   batches of seeds, so ZML panics, XLA aborts and hangs only cost one seed. A batch that
   times out is split at the seed in progress (read from the `--status` file), which is
   rerun alone with the full `--timeout`. Crashes are
   attributed to the op being lowered and signatures are normalized for deduplication.
5. **Reduction** (`reduce.zig`). Each new signature is reduced: keep only the failing node,
   drop outputs, replace computations by literals of their values, reroute operands,
   bypass nodes (inside loop and branch bodies too), simplify data, with region aware
   dead code elimination, then bisect each dim down to
   the smallest that keeps the failure (literals are cropped). Candidates run with
   `--no-trace`, which skips the 12 s stack trace symbolization of a Debug build when
   they crash. Then `emit.zig` writes a
   standalone `test "..."` calling the public API with the data embedded.

## What it covers

Op families live in `ops/*.zig` and are registered in `ops.zig`. Each provides `propose`
(generation), `eval` (reference semantics, from the doc comments and the StableHLO spec),
`lower` (the ZML calls) and `emit` (the same calls as Zig source).

- **Tensor**: elementwise math, all binary ops with ZML's broadcasting, constants, compare,
  select, clamp, convert, bitCast, reshape and axis manipulation, transpose, broadcast,
  static and dynamic slices, concatenate, stack, pad, reverse, repeat, stutter, chunk,
  split, roll, triangular, toDiagonal, cartesianProduct, gather, gatherSlices,
  scatterSlices, dynamicUpdateSlice, sort, argsort, topK, argMax, reductions, cumsum,
  softmax, logSumExp, dot, dotGeneral, outer, conv1d/2d, maxPool1d/2d, reducePrecision,
  fmod, complex numbers and FFTs, cholesky, triangularSolve, random numbers (checked
  through properties), reuseBuffer, toMemory, withPartitioning. Options: scatter
  `indices_are_unique`/`indices_are_sorted` and gather `indices_are_sorted` (programs
  breaking the promise are rejected), conv feature and batch groups and negative padding,
  maxPool2d indices (the column of the max).
  Arguments are passed both as slices and as tuples, as Shapes and as struct literals,
  by tag, by (negative) index and as `._N`; reshape dims also use `-1` and `.auto`.
- **zml.ops**: `while`, `if`, `if2` with randomly generated nested regions, `reduce` with
  several inputs and axes and custom bodies, `sort` with custom comparators,
  `reduceWindow`, `manualComputation`, `gather`.
- **zml.quantization**: quantizeBlockFp8 (e4m3fn/fnuz values, f32/e8m0 scales, any block
  size), quantizeNvfp4 (f32 and bf16, with a global scale), GlobalScale.asMultiplier,
  nn.unpackFp4 and the bf16 ops.scaled_dot. The quantizers are emulated bit for bit.
- **zml.nn**: activations, LayerNorm, rmsNorm, normalizeVariance, normalizeL2, rope (all
  layouts and scalings), splitRealImg/mergeRealImg, causalAttnMask, sdpa, Linear,
  TokenEmbedding, nearest/upsample, resize linear/cubic, sampleTokens(Dynamic),
  GatedDeltaNet.forward (the recurrence against a host reference, outputs and final state,
  clamped lengths).
- **zml.attention**: `attention` with the vanilla backend (causal mask gathered at
  `token_index`, batched or not, clamped indices). The paged attention stablehlo backend
  has a differential test against a host reference in `regressions/found/paged_attn.zig`.
- **Runtime**: sharded programs on a `data` x `model` mesh over all devices, structured
  arguments and outputs through `zml.meta` / `Bufferized`, executables run twice,
  sub-byte and f8/f4 dtypes through ZML's packing.
- **Input space presets** (by seed): default, `edge` (rank 8, size-0 and size-1 axes,
  signed zeros), `big` (dims up to 129 favoring sizes around tile widths, rank 4, 32k
  elements), `long` (30 to 80 nodes). Swarm testing: half the programs drop a random 40%
  of the families.
- **Parsers** (`zmlsmith parse-fuzz`): HTTP answers served from a local socket to the
  `VFS.HTTP`, S3 and GCS backends (`vfs`, `s3`, `gcs`, endpoints pointed at the socket),
  sentencepiece ModelProto files built and mutated (`spm`), safetensors files,
  tokenizer.json through
  `Tokenizer.fromBytes` (then encode, decode and streaming decode of odd ids), the
  homemade normalizer config, and rope scaling configs. Valid seeds are mutated
  structurally and byte by byte; errors are fine, panics, aborts and hangs are findings.
  Crashes are bucketed by panic message and first ZML frame, and reduced by chunk removal.
- **HF backend** (`regressions/hf_vfs.zig`): `VFS.HF` hardcodes `https://huggingface.co`,
  so its client goes through a local https proxy (`regressions/found/hf_fixture.zig`,
  python3) that answers as huggingface.co with a test certificate trusted only by that
  client. Random repository layouts are checked for open, stat, listing and ranged reads.
- **Host checks** (`zmlsmith host`): every f8/f4 code through `zml.floats`, `Shape`
  invariants on random shapes, and random `Slice` views (subSlice, squeeze) against an
  independent model. Regression tests cover `zml.meta` traversals, DataType metadata,
  homemade tokenizer round trips, `TensorStore.View` and `VFS.HTTP` against a local server.
- **Buffer API** (`zmlsmith buffers`): host/device round trips (fromBytes, fromSlice and
  their Opts variants), views, shards and device pointers, uninitialized buffers, await,
  memory kinds, deinitAll.
- **Robustness audit** (`zmlsmith robust`): invalid uses of the API, each in its own
  process, classified as cleanly rejected or not (Zig safety panics, invalid MLIR,
  backend aborts mean missing validation).

Out of scope on CPU: the triton, fly, cute, cuda tile and NKI kernels, the attention and MoE
backends, and `customCall` / `composite` which need registered targets. `nn.scaledDot` (and
so quantized `nn.Linear`) doesn't compile on CPU: see `regressions/found/scaled_dot_cpu.zig`.
`toMemoryAll`/`onMemoryAll` are identities on CPU.

## Commands

| command | |
|---|---|
| `gen --seed S` | print the program of a seed |
| `run --seed S --count N` | run seeds in process |
| `repro FILE` | run a program file (`-v` prints it, `--all-nodes` returns every node) |
| `fuzz --seed S --count N --jobs J --out DIR` | crash isolated campaign with automatic reduction (`--no-reduce`) |
| `reduce FILE\|--seed S` | reduce a failing program |
| `emit FILE\|--seed S` | print the program as a Zig test |
| `emit-batch --seed S --count N --out FILE` | write the tests of every passing seed; `bazel test //tools/zmlsmith/regressions:test` must pass |
| `selfcheck --count N` | print/parse round trip and replay determinism of the IR |
| `stats --count N` | how often each op kind and dtype is generated |
| `host`, `buffers`, `robust` | the host, Buffer and robustness checks |
| `parse-fuzz --target T --seed S --count N --jobs J --out DIR` | parser fuzzing (`T`: safetensors, tokenizer, spm, homemade, rope, vfs, s3, gcs), writes `summary.txt` as it goes |
| `parse-run --target T FILE`, `parse-gen --target T --seed S` | run one parser input, print the input of a seed |

Sharding flags: `--mesh2d` uses a two level physical mesh with `data` and `model` on
distinct axes (the default CPU mesh binds both to one bus), `--gspmd` partitions with GSPMD
instead of Shardy. Runtime properties of executables (donation, repeated, unawaited and
concurrent calls, concurrent compiles, FnExe, Rng) are in `regressions/runtime.zig`, and
`zmlsmith robust` also misuses executables at run time.

Common flags: `--only fam1,fam2` and `--exclude fam` restrict the families (names are the
fields of `ops.impls`), `--known` generates known bugs again, `--mutate M` injects a bug in
the lowering (`ops.Mutation`: truncating_convert, drop_neg_zero, sqrt_ulp) to check that
the oracle catches it.

## Known bugs

`gen.KnownBug` lists the ZML bugs zmlsmith found that are still open. The generator avoids
them by default, so they don't mask new findings; `--known` generates them again, to check
a fix. When a bug is fixed, delete its entry and the compiler points at the avoid sites.

Avoiding a bug also hides everything else in the inputs that trigger it, so 15% of the
seeds (`gen.liftedBug`) stop avoiding one random known bug. A failure saved from such a
seed says so in its `.zs` file (`# known bug not avoided in this seed`): check that bug
first.

## Adding a family

Write a namespace with `Attrs`, `weight`, `propose`, `eval`, `lower` and `emit` in
`ops/`, add it to `ops.impls`, then check:

```
zmlsmith selfcheck --count 2000
zmlsmith fuzz --only <family>,unary,binary --count 300 --no-reduce
zmlsmith emit-batch --only <family> --count 60 --out tools/zmlsmith/regressions/generated.zig
bazel test //tools/zmlsmith/regressions:test
```
