---
written_on: 2026-11-27
written_by: agent:claude (steering feasibility, Sonnet 5.5)
---
Status: final

# M3 steering vector: technical feasibility on vLLM 0.11.0 / transformers 4.57.1

Labels: **[V-src]** read in source (file:line), **[V-test]** run on serrano CPU (tiny random Olmo3, transformers 4.57.1;
`proto_steer/test_steer_cpu.py`), **[EST]** my estimate, **[GUESS]** unsupported guess, **[UNTESTED]** code written but not run.
vLLM files: raw copies of tag v0.11.0 in `/scratch/cowifornia/nd-rl/olmo_elicit/metrics2/ext/vllm_v0.11.0/` (paths below are
relative to its `vllm/`). HF file: `/scratch/cowifornia/venvs/olmo-edl/lib/python3.11/site-packages/transformers/models/olmo3/modeling_olmo3.py`.

## 1. What "the residual stream after layer l" is

**vLLM 0.11.0.** `Olmo3ForCausalLM` maps to `("olmo2", "Olmo2ForCausalLM")` (`model_executor/models/registry.py:132`); there is no
separate Olmo3 class. Decoder layer `Olmo2DecoderLayer.forward` (`olmo2.py:258-268`): `residual = h; h = self_attn(positions, h);
h = post_attention_layernorm(h); h = h + residual; residual = h; h = mlp(h); h = post_feedforward_layernorm(h); h = residual + h;
return h`. The layer takes and returns ONE tensor; both norms are called with a single argument (`RMSNorm(x)`, no `residual=` kwarg),
so vLLM's fused-add-RMSNorm path (`layernorm.py:216-218`) is never used here. Nothing is fused across layers: the tensor passed from
layer to layer (`olmo2.py:324`, `hidden_states = layer(positions, hidden_states)`) is the true residual stream. Final norm at `olmo2.py:331`.
The model class is `@support_torch_compile` (`olmo2.py:271`).

**HF 4.57.1.** `Olmo3DecoderLayer.forward` (`modeling_olmo3.py:258-277`) is the same order: attention, `post_attention_layernorm`,
`residual + h`; MLP, `post_feedforward_layernorm`, `residual + h`. Attention input is the raw residual (no pre-norm); q_norm/k_norm
are applied to the full q/k projections before RoPE in both.

**Same quantity: yes** [V-src]. Differences are numerical only: HF's RMSNorm multiplies the weight in fp32 and then casts
(`modeling_olmo3.py:60`); vLLM's native form casts back to bf16 first and then multiplies by the weight (`layernorm.py:196-202`), and
the CUDA op is a fused kernel [GUESS: its rounding matches neither exactly]. On the tiny model in bf16 the two norm orderings differ by
up to 1 bf16 ulp (max rel 0.0078) [V-test]. So expect bf16 noise, not a definitional gap.

**Index convention (use "after l layers", l = 0..64, 0-based layer modules):**
- Residual after l layers = output of `layers[l-1]` = input of `layers[l]` = HF `hidden_states[l]`.
- HF 4.57.1 `output_hidden_states` collects the input of the first layer, then each layer's output, and then REPLACES the last entry
  with the post-final-norm `last_hidden_state` (`utils/generic.py:1015-1016` and `1085-1089`). So `hidden_states` has 65 entries;
  `[0]` = embeddings, `[l]` = raw residual for l = 1..63, `[64]` = after the final norm, NOT the raw residual after layer 63. [V-src + V-test:
  `hs[L] != raw output of layer L-1` but `== model.norm(raw)`.] Steering and the mean difference should therefore use l in 1..63.
- Trap [V-test]: HF's recorder wraps `layer.forward`, and a user `register_forward_hook` runs AFTER the wrapper has recorded. A hook that
  adds v to `layers[l-1]`'s output does NOT change the recorded `hidden_states[l]` (it stays pre-hook), but the next layer's input is
  `hs[l] + v` exactly (checked with a pre-hook on `layers[l]`). Never validate steering by reading `hidden_states` of the steered run.

## 2. Can a weight edit give a constant residual addition at a middle layer?

**Only at l = 0, confirmed** (argument from the architecture, plus one test).
- Every residual increment is `RMSNorm(sublayer(x))` (`olmo2.py:260,266`). A weight-only edit can change what the sublayer outputs, but
  there is no bias anywhere in the model (`named_parameters` has no `.bias` [V-test]; vLLM builds all projections with `bias=False`,
  `olmo2.py:101,149,200,212`) and the sublayer inputs (raw residual, no pre-norm) are not constant. A constant output would
  need a constant input; the only constant anywhere is nothing. The output norm also rescales whatever comes out, so even a hand-built
  "constant feature" (e.g. a massive-activation dimension) would give a norm-fixed, input-dependent, not additive, effect [argument, not a theorem].
- At l = 0 the addition is exact: adding v to every row of `embed_tokens.weight` equals a hook adding v to the embedding output
  [V-test, fp32]. In bf16 each row is re-rounded once, the same rounding as adding v at runtime. That is the only exact weight route and
  it is useless for the question (the final-minus-dpo mean difference at the embedding is about 0 because RL barely moves embeddings; I did not
  measure that).
- LoRA does not help: vLLM LoRA adds low-rank terms to linear-layer outputs before the post-norm, with no constant term.
So a runtime change to the forward pass is required.

## 3. Routes to generate with steering

Dan's sampler: `olmo_ifladder/sample.py` imports vLLM inside `main()` (line ~182), builds `LLM(...)` at line 261 with
`enable_prefix_caching=True`, `generation_config="vllm"`, bf16, `seed`, and then drives `llm.llm_engine.add_request` / `eng.step()`
(lines 277-281, 296). T and top_p are module constants. Nothing in it needs to change.

**Route A (recommended): patch the decoder layer class before `LLM()`, one process per setting.** `proto_steer/steer_vllm_patch.py` +
`steer_sample.py` [UNTESTED, compiled only]: the wrapper reads `STEER_LAYER` and `STEER_VEC` (a file holding the final alpha*v),
monkeypatches `Olmo2DecoderLayer.__init__` (registers a bf16 buffer on layer L-1 only) and `.forward` (adds the buffer to the output),
then imports Dan's `sample.py` as a module and calls `main()`. Unset env = stock model.
- Why the patch must precede `LLM()`: the model is traced and compiled during engine start-up (`compilation/decorators.py`,
  `backends.py:476-551`), so a hook added later is not in the compiled graph.
- Processes: with the default `VLLM_ENABLE_V1_MULTIPROCESSING=1` the engine core runs in a child created by `get_mp_context()`
  (`v1/engine/utils.py:96`); the default start method is fork (`envs.py:65`), and vLLM forces spawn only if CUDA is already initialised in
  the parent (`vllm/utils/__init__.py::_maybe_force_spawn`, read from the v0.11.0 file, not downloaded to ext/). If the parent has
  not touched CUDA, the fork inherits the patched class and the engine keeps its separate process (CPU scheduling overlaps GPU).
  If a check shows the patch is not active in the child (the patch prints `STEER_ACTIVE` from the worker), fall back to
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` (`envs.py:1018`, `v1/engine/llm_engine.py:156,172`): same code, engine in-process.
- torch.compile: the patched forward is traced like any other; the compile-cache key hashes the contents of every traced file, including
  the patch file (`backends.py:476-516`; `decorators.py:280-286` adds every inlined function's file), so a patched compile can never be
  confused with the stock cache entry [V-src]. The vector is a module buffer (a fixed device address), not a Python constant, so
  inductor does not bake the values in [EST: standard torch 2.8 behaviour with `inline_inbuilt_nn_modules`; to be confirmed by the pod test].
- CUDA graphs: default mode is `FULL_AND_PIECEWISE` (`config/__init__.py:359-366`). Graph replay reads the buffer from its captured
  address, so a constant add is captured correctly; `enforce_eager` is NOT needed [EST; the verification plan in section 4 tests this].
  Eager would also cost speed (section 6).
- Prefix caching: a prompt block's hash is built from tokens + LoRA id + `cache_salt` only (`v1/core/kv_cache_utils.py:425-440,
  534-540`); it knows nothing about steering. Within one process the vector is fixed, so cached prefixes are always consistent. Never
  switch the vector inside one engine without `llm.reset_prefix_cache()` (`entrypoints/llm.py:1415`) or a per-setting `cache_salt` (a
  `cache_salt` string key in the prompt dict, `inputs/data.py:43,83,95`; Dan's `add_request` passes `{"prompt_token_ids": ids}`, so a salt would
  need a wrapper around `eng.add_request`). One process per setting makes this a non-issue, which is the main reason to prefer it.
- Changing vector/alpha without reload: possible (in-place `copy_` into the buffer through `llm.apply_model(fn)`, `entrypoints/llm.py:509`,
  `worker/worker_base.py:67`; under multiprocessing the callable is cloudpickled and needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
  (`envs.py:1220`), or use `--worker-extension-cls` (`worker/worker_base.py:225`) with a named method). It saves about 1 to 2 min per setting
  [EST] and needs the prefix-cache reset above plus a graph-replay-sees-new-value test. Given 11 settings, saving ~15 min (~$1.3),
  I do not recommend it; it trades a small cost for a stale-state risk.
- Sliding-window layers: 48 of 64 layers use a 4,096 window (`olmo2.py:129` passes `per_layer_sliding_window`). The add is on the
  residual, independent of the attention variant. The only coupling is that later layers' KV (full and windowed) are computed from steered
  activations; both engines do that per request. HF applies the window in its mask; they differ only after 4,096 tokens, so the
  verification must include a teacher-forced response longer than 4,096 tokens.
- Layer 0 / 64 limits: l = 0 would be an embedding edit; l = 64 has no raw residual in HF output. Use l in 1..63.

**Route B: out-of-tree registration** (`ModelRegistry.register_model("Olmo3ForCausalLM", "pkg:Cls")` overwrites the entry with a
warning, `registry.py:561-601`; engine-core process needs it too via the `vllm.general_plugins` entry point,
`plugins/__init__.py:11,59-72`, `v1/engine/core.py:73-74`). Same forward code as A, but needs a pip-installable package with an
entry point, a subclass of `Olmo2ForCausalLM` that rebuilds the decoder layers, and attention to the architecture-inspection
subprocess. More plumbing, same speed and same compile/graph/cache behaviour. Only worth it if fork inheritance fails AND single-process
mode is too slow, which I think is unlikely.

**Route C: HF `generate` with forward hooks.** Exact, simple, no compile questions, a single process can switch vectors for free.
But no continuous batching (finished sequences wait for the longest, and a draw that loops to 32,768 tokens holds its whole batch),
and eager HF decode for 64 layers is python-overhead-bound. Back-of-envelope [EST/GUESS]: weights alone 64.5 GB / ~4.8 TB/s =
~13.5 ms per decode step; eager HF with a dynamic cache plausibly takes 60 to 120 ms per step regardless of batch up to the memory
limit; KV per token is 64 layers x 2 x 8 x 128 x 2 B = 262 KB (full attention), so 4,500-token sequences take 1.2 GB each and the ~65 GB
left after weights hold about 50 sequences, more because the 48 sliding layers only keep 4,096 tokens. At batch ~64 and ~100 ms per step
that is ~600 tok/s before padding and straggler loss, so I put real throughput at **200 to 500 tok/s** (4 to 7 times slower than vLLM).
This is a guess; I measured nothing.

**Recommendation: Route A, one process per (layer, alpha, direction) setting, fork-inherited patch (MP engine), fallback
`VLLM_ENABLE_V1_MULTIPROCESSING=0`.** Dan's sampler is imported unchanged; the whole change is two ~40-line files; the same Docker
stack; the vector is fixed per process so there is no cache-staleness path. HF is used only for the activation means and for verification.

## 4. Verification on the pod, before any sampling

All with the dpo weights, both sides bf16, one H200, about 15 min (~$1.3) [EST].
1. **Stock noise floor.** Pick 4 fixed (prompt, response) pairs: two ~1k tokens, one ~4.3k, one ~5.5k (so the sliding window is crossed;
   take dpo or s2300 responses from the stored draws). vLLM: `LLM(...)` with `SamplingParams(max_tokens=1, prompt_logprobs=0)` on
   prompt+response token ids; HF: teacher-forced log_softmax gather on the same ids. Record per-token |delta logprob| (mean, p99, max)
   between stock vLLM and stock HF. This is the tolerance (set from data, not assumed; bf16 stacks typically differ by about 1e-2 mean
   [GUESS]).
2. **Zero vector.** Patched vLLM with alpha*v = 0 against stock vLLM: logprobs should be bit-identical (x + 0 is exact in bf16) and a
   short greedy generation identical. A mismatch means the patch changes numerics or the graph.
3. **Steering applied as intended.** Choose a deliberately large effect (e.g. alpha*v with norm = a good fraction of the typical
   residual norm at layer l). Compare steered-vLLM logprobs with HF + hook (`register_forward_hook` on `layers[l-1]` returning
   `out + vec_bf16`) on the same 4 sequences. Pass: steered-vs-steered delta within the step-1 tolerance, while steered-vs-unsteered
   delta is much larger than that tolerance (so the test has power). Negative control: HF hooked at layer l+1 or l-1 must differ from vLLM
   by far more than the tolerance. Repeat for two layers (e.g. l = 20 and 50) to cover a windowed and a full layer at the add site.
4. **Graph/compile check.** Run step 3 once with default settings (compile + CUDA graphs on) and once with `enforce_eager=True`; the two
   must agree within tolerance. It also checks the prefix cache on a repeated prompt (second request must give the same logprobs as the first).
5. **Active in the engine process.** The patch prints `STEER_ACTIVE` from the worker; the wrapper should write l, norm and a hash of the
   vector file into the sampler's meta sidecar. If it is absent from the log, abort.
Optional: capture the activation in eager vLLM through `llm.apply_model` with a forward hook on `layers[l-1]` (prefill of one prompt) and
compare it to HF `hidden_states[l] + vec` directly.

**Tested on serrano CPU** (`proto_steer/test_steer_cpu.py`, tiny random Olmo3 in the olmo-edl venv, no vLLM): `hidden_states` indexing
(65-entry analogue, last entry is post-norm), hook semantics and the recorder trap, alpha = 0 exact, input of layer l = `hs[l] + v` exactly,
embedding-row edit equals the l = 0 hook, absence of biases, bf16 norm-ordering difference. **Not testable there:** anything about vLLM
(patch, fork inheritance, compile cache, CUDA graphs, prefix cache, logprob agreement) and real-weight numerics.

## 5. Computing the mean activations

- v_l = mean over prompt token positions of (`hidden_states[l]` of s2300 minus that of dpo), l = 1..63, with HF `output_hidden_states=True`
  (index convention in section 1), same chat-template token ids as the sampler (`prompt_ids` in `sample.py`, including the template's system
  prompt and generation-prompt tokens), no padding positions in the mean (use per-prompt unpadded lengths or batch size 1). Do the two models
  one after the other in separate processes, accumulating fp32 sums per layer in place of storing activations: 63 x 5,120 fp32 = 1.3 MB.
- Decide up front, with a look at the data: the first token position typically carries an attention-sink residual of very large norm in
  LLMs [GUESS for this model]; a plain mean over positions can then be dominated by position 0 (and by the repeated template tokens, which
  the mean weights by how many prompts share them). Save per-position-group means (position 0, template tokens, user tokens) so the choice
  is data, not a surprise. This is a design point for the orchestrator, not a feasibility blocker.
- Memory/time [EST]: one 64.5 GB bf16 model on the H200 leaves ~75 GB; a forward of a few hundred prompts of a few hundred tokens needs a few
  GB of activations. Load 2 to 10 min each (from the volume; 12 s from page cache), forward passes under a minute per model for ~300 prompts.
  About 15 to 25 min for both, ~$1.5 to $2.2.
- Disk/download: dpo is needed on the pod anyway (it is the generator) and s2300 only for this step: two 64.5 GB checkpoints = 129 GB on
  `/workspace` plus the HF cache copy if `HF_HOME` is on the same volume (download then copy would double it; use `--local-dir` or point
  HF_HOME at the volume and keep one copy). Size the volume at >= 200 GB [EST]. 2026-11-25 saw 20 s to 3 min per checkpoint (CONTEXT.md).
  Download s2300 only after the dpo steps, delete after the means are written; the vectors (1.3 MB) go back to serrano.
- Ordering: means (HF, s2300 then dpo) -> verification (section 4, dpo) -> dev settings -> pick on dev -> evaluation + random control.

## 6. Cost table

Assumptions: $5.29/h; 4,500 tokens per draw; 9 dev settings x 64 draws = 2.59 M tokens; 2 evaluation settings x 820 draws = 7.38 M tokens;
total 9.97 M. vLLM throughput from the 2026-11-25 runs (1,400 to 1,700 tok/s at ~500 concurrent sequences, CONTEXT.md); the dev runs have only
64 draws, so they run at lower concurrency and are tail-bound [EST]. Startup per setting: model load ~12 s from page cache (2 to 10 min if
not cached), compile 28 s at first start and less with a warm cache [log observation; the cache hit is my EST], CUDA-graph capture and
warm-up [GUESS ~40 to 60 s]; I use ~2 min per setting, 11 settings = 22 min.

| Route | Build/prep | Per-setting overhead | Throughput | Generation time (9.97 M tok) | Startup total | Total GPU h | $ |
|---|---|---|---|---|---|---|---|
| A: patch, MP engine (fork) | ~1 h human, 0 GPU | ~2 min (new process) | ~1,500 tok/s [EST, = Dan's rate] | 1.85 h | 0.37 h | ~2.2 h | ~$12 |
| A': same, `VLLM_ENABLE_V1_MULTIPROCESSING=0` | same | ~2 min | ~1,250 to 1,400 [GUESS, -10 to -15 % without CPU/GPU overlap] | 2.0 to 2.2 h | 0.37 h | ~2.5 h | ~$13 |
| A'': A with `enforce_eager` | same | ~1 min (no compile/capture) | ~1,100 to 1,300 [GUESS] | 2.1 to 2.5 h | 0.2 h | ~2.5 h | ~$13 |
| B: registered model/plugin | ~2 to 3 h human | as A | as A | 1.85 h | 0.37 h | ~2.2 h | ~$12 |
| C: HF generate + hooks | ~1 h | ~0 (one process, vector swapped in place) | 200 to 500 tok/s [GUESS] | 5.5 to 13.8 h | ~0.1 h | 5.6 to 14 h | $30 to $74 |

Fixed extras for every route (not in the table): mean activations ~0.3 to 0.4 h, verification ~0.25 h, pod setup/download ~0.3 to 0.5 h:
about +1 h, +$5. So Route A all-in is about $17 [EST]; above the $15 M3 split, and the eval draws dominate (82 min, $7.2). Dev-run
tails: 64 draws at ~5k tokens with a 32,768 cap; if a steered setting loops, a single draw can add up to ~30 min of low-concurrency decode.
[GUESS: ~25 to 40 ms per step at batch 64]. A watchdog and a smaller dev `--max_tokens` (e.g. 12,000) are advisable; that is a design choice
for M3 since it changes what a draw scores. Reducing the dev grid to fewer than 9 settings saves ~$1.0 per setting.

## 7. Risks

1. Fork inheritance of the patch is untested; if `import vllm` initialises CUDA in the parent, vLLM forces spawn and the child sees stock code.
   Mitigation: the `STEER_ACTIVE` print and step 5; fallback to in-process mode.
2. The compiled graph reading a mutable buffer correctly is standard torch behaviour but unverified here; steps 3 and 4 test it directly.
   Because vectors are fixed per process in the recommended route, an in-place-update failure cannot occur.
3. bf16 noise: the add is done in bf16 at every position; HF and vLLM differ by ~1 ulp scale per layer; the test tolerance is set from the
   stock-vs-stock difference, not guessed.
4. Steering at a large alpha may produce loops to the 32,768-token cap: up to 7x cost on affected draws.
5. Mean-vector definition (attention-sink position, template tokens) can change what v is; handle by saving group means (section 5).
6. HF `hidden_states[64]` is post-norm; l = 64 is unavailable. HF-recorded hidden states do not see forward-hook edits (section 1).
7. A new vLLM point release or a different image would change all line citations: they are for tag v0.11.0 only.

## 8. What I could not check

- Nothing was run with vLLM, a GPU, or real weights; every vLLM-side statement is from source reading; the draft patch files are only
  byte-compiled.
- `vllm/utils/__init__.py` (spawn forcing) was read from a one-off fetch, not saved to ext/; the other cited files are in ext/.
- The actual CUDA RMSNorm kernel's rounding (`csrc/layernorm_kernels.cu`) was not read.
- The `AutoWeightsLoader` behaviour for an extra non-persistent buffer was not read in source; I expect it ignores it [GUESS].
- Dan's sampler was read, not run; I did not read its request-preparation lines (about 200-251) in detail.
- All throughput, startup and HF-speed numbers are estimates; the only measured inputs are the 2026-11-25 rates and load times.
- I did not measure the embedding-level mean difference, the attention-sink norm, or whether the two checkpoints' embeddings differ.
