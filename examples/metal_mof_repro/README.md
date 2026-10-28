# Metal multi-output fusion miscompile

On Metal, the DiffusionGemma port printed `<|channel>` and then garbage. The cause was not the model:
XLA's `multi_output_fusion` pass produces a kernel with two outputs, and the Metal backend only
writes the first one. This directory contains a 100-line repro and the method used to find it.

## Live demo (2 minutes)

```sh
# 1. Correct output (workaround enabled in zml/Compiler.zig)
bazel run //examples/metal_mof_repro --@zml//platforms:metal=true -- --seq=1
#   gate*up max|got|=7136.0000 max|ref|=7135.0103 ... -> OK

# 2. Comment out this line in zml/Compiler.zig (`.metal` branch):
#      try setXlaOverrideFlag(overrides_map, "xla_disable_hlo_passes", "multi_output_fusion", upb_arena);
bazel run //examples/metal_mof_repro --@zml//platforms:metal=true -- --seq=1 --dump=/tmp/mof_s1
#   gate*up max|got|=0.0000 max|ref|=7135.0103 ... -> WRONG

# 3. Same graph, two rows instead of one: correct again
bazel run //examples/metal_mof_repro --@zml//platforms:metal=true -- --seq=2 --dump=/tmp/mof_s2
#   ... -> OK

# 4. Compare which kernels XLA actually launched
cat /tmp/mof_s1/*thunk_sequence.txt
cat /tmp/mof_s2/*thunk_sequence.txt
```

`thunk_sequence.txt` is the readable list of the kernels in the compiled program:

| | seq=1, pass enabled (bug) | seq=1, pass disabled (fix) | seq=2 |
|---|---|---|---|
| gate dot | `input_reduce_fusion_1 grid: [1056, 1, 2]` (one kernel, 2 outputs) | `input_reduce_fusion_1 grid: [1056, 1, 1]` | `kGemm` |
| up dot | *(merged into the kernel above)* | `input_reduce_fusion_1 grid: [1056, 1, 1]` | `kGemm` |

The `z = 2` in the grid shows that both reductions run in one launch. Only output 0 gets written, so
`up` is all zeros and `gate * up` is 0.

## How to find a bug like this yourself

Each step halves the search space. Write down each result before moving on.

### 1. Get a ground truth and find the first wrong token

Run the HF reference greedily (`../temp/ar.py`) and the Zig binary with `--topk=1` on the same
prompt, then compare token IDs. Here the first token matched and the second did not, so **prefill is
correct and decode is wrong**. Everything that prefill and decode share (weights, loading, rope
tables, the tokenizer) is therefore not the cause.

### 2. Rule out the model logic with the "move it to prefill" trick

Append the first generated token to the prompt and run prefill only. If prefill now predicts the
correct second token, then the KV cache, positions and masking are consistent, and the problem is
in the decode executable itself.

### 3. Change one compile-time shape

Decode is compiled for `s = 1`. Compile it for `s = 2`, feeding the token twice and taking row 0.
If the output becomes correct, the math is fine and **the compiler handles `s = 1` differently**.
That is a strong signal of a compiler or backend bug rather than a model bug: the graph is the
same, only its shape changed.

### 4. Shrink to a standalone program

Copy the smallest block that still breaks (here the MLP: norm, gate/up dots, activation, down dot)
into a tiny `zig_binary`. Use random weights and compare against a plain host loop. Keep the real
dimensions and the real weight layout: with `{in, out}` weights this repro does **not** fail,
because XLA picks different kernels. Then strip parts until removing anything makes the bug go
away. Inspecting intermediates at this point showed `up == 0`.

### 5. Read what XLA actually compiled

Pass `.xla_dump_to = "/abs/dir"` in `zml.Compiler.Options` (this repro exposes it as `--dump`).
Useful files in the dump directory:

- `*thunk_sequence.txt`: kernels in launch order, with grid sizes. Diff it between the good and bad
  configurations.
- `*before_optimizations.hlo.pb` / `*_gpu_after_optimizations.hlo.pb`: the graph before and after
  XLA's passes (protobuf; `strings file | grep fusion` is enough for a quick look).
- `*.debug_options`: the XLA flags that were actually applied, to confirm your override took
  effect.

### 6. Bisect the XLA passes

Once you know which kernel is wrong, disable the passes that could have produced it, one at a time,
in the `.metal` branch of `zml/Compiler.zig`:

```zig
try setXlaOverrideFlag(overrides_map, "xla_disable_hlo_passes", "<pass-name>", upb_arena);
```

Several names can be comma-separated. Here `dot-merger` made no difference and
`multi_output_fusion` fixed it. Confirm in `*.debug_options` that the flag was applied, and in
`thunk_sequence.txt` that the merged kernel is gone.

### 7. Verify on the real workload

Rerun the full model and compare against HF token for token, not just "the text looks fine". Check
that compile time and tok/s did not regress.

## Takeaways

- A `s=1` vs `s=2` difference with identical math points to the compiler, not the model.
- Output that is exactly zero (not just noisy) usually means a missing write, not a numerical issue.
- Reproduce with the real dimensions and layouts; the bug depends on XLA's kernel choice.
- The workaround disables the pass for all Metal compilations. The underlying bug is in the
  prebuilt Metal PJRT plugin and should be reported upstream with this repro.
