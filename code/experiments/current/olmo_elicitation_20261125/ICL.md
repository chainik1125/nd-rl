---
written_on: 2026-11-25
written_by: agent:claude (EDL build agent)
papers: []
---

# In-context description length on the Olmo 3.1 Think 32B checkpoints (method note)

## Summary

This note covers the in-context arm (item R2 of `docs/literature/2026-11-25-elicitation-metrics/REVIEW.md`, in the scope
the coordinator set on 2026-11-25). It measures the NLL of the final model's (`s2300`'s) responses under a checkpoint
when k of `s2300`'s responses to other prompts sit in the context. No weights change, so the reading does not depend on
the LoRA optimizer, learning rate or adapter, which makes it an independent check on EDL. This note describes the
method and the CPU checks. **It contains no 32B results.**

- Code: `olmo_icl.py` (`run`, `bench`), `analyze_icl.py`, `fig_icl.py`, tests in `test_olmo_edl.py` (`-k icl`).
- Branch: `leon/olmo-elicitation` (uncommitted at the time of writing).

## Design

### Targets, demonstrations and k

- **Targets.** These are the pool's 32 test responses and 38 reach responses (`s2300`'s strict passes, ≤ 6,144 tokens; see `EDL.md`), teacher-forced.
- **k.** k ∈ {0, 1, 2, 4}. k = 0 is step 0's L_0 exactly.
- **Demonstrations.**
  - They come from the 26 genuine-labelled train rows (`s2300`'s hand-labelled strict passes to train prompts).
  - The rows are shuffled once with `random.Random("0:icl")`, and draw d takes block d of 4.
  - Within a draw the sets are nested: the k = 1 demo is the first of the k = 2 set.
  - The two draws are disjoint.
  - The sets depend only on the pool and the seed, so they are identical at every checkpoint.
  - Train, test and reach prompts are disjoint, and the code asserts that no target is among the demos.
  - Seed 0 gives:

| Draw | Demonstrations (response tokens) | k = 1 / 2 / 4 prefix tokens | Longest context (prefix + target prompt + response) |
|---|---|---|---|
| 0 | ifbench_126 (2,407), ifbench_32 (3,653), ifbench_148 (5,285), ifbench_115 (3,865) | 2,483 / 6,248 / 15,638 | 21,848 at k = 4 |
| 1 | ifbench_85 (3,030), ifbench_28 (3,304), ifbench_297 (2,012), ifbench_274 (3,000) | 3,128 / 6,588 / 11,678 | 17,888 at k = 4 |

In total there are 490 scored responses per checkpoint and 5.39 M forward tokens without prefix caching.

### How the context is built

The context is built in the olmo_thinker template at the token level:

```
<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n
[<|im_start|>user\n{demo prompt}<|im_end|>\n<|im_start|>assistant\n<think>{demo response ids, EOS dropped}<|im_end|>\n] × k
<|im_start|>user\n{target prompt}<|im_end|>\n<|im_start|>assistant\n<think>{target response ids: scored}
```

- **Thinking in history.** The template keeps the thinking of earlier assistant turns verbatim; it does not strip it. The review asked to check this.
  - Checked: `apply_chat_template` on a history turn renders `<|im_start|>assistant\n<think>T</think>A<|im_end|>\n`.
  - The test `test_icl_context_matches_chat_template_text` decodes a real k = 2 context and checks that it equals the template's text render.
- **k = 0 equivalence.** The piecewise encoding of the system turn, user turn and generation prompt equals Dan's `prompt_ids` for all 134 pool rows, using the real tokenizer. `run` asserts this for every row.

### Model limits (recorded, not worked around)

- Olmo 3.1 Think 32B has `max_position_embeddings` 65,536 (YaRN from 8,192). The longest context here is 21,848.
- It has a 4,096-token sliding window on 48 of its 64 layers (three sliding-window layers to one full-attention layer).
  - At k ≥ 2, most demonstration tokens are more than 4k positions before the target. They reach it only through the 16 full-attention layers.
  - So a small in-context gain at k = 4 can partly reflect the architecture rather than the checkpoint.

### Scoring

- **k = 0.** Uses the EDL run's batched scorer with the same `eval_batch_tokens` and `loss_chunk`, read from the checkpoint's `edl_rows.jsonl` recipe.
  - The model is LoRA-wrapped with B = 0, exactly as in the EDL run, which makes the adapter exactly inert.
  - It is compared with `step0.json`. The maximum absolute difference per example is stored in the row as `k0_check_max_abs_diff`.
  - Above `--k0-tol` (default 0), the run stops before any k > 0 row.
- **k ≥ 1.** Batch 1, forward only, under `inference_mode`.
  - With `--prefix-cache`, the demo prefix's KV cache is built once per (k, draw) and deep-copied for each target. The first target of every (k, draw) is also scored without the cache, and the difference is stored as `cache_check_abs_diff`.
  - On CPU in fp32 with the tiny real-vocabulary model, that difference was 1.3e-5 and 3.0e-5 nats per example sum.
  - `test_icl_prefix_cache_matches_full_forward` checks the cached path against the full forward with contexts longer than the sliding window.
- **Output.**
  - `icl_rows.jsonl`: one row per (k, draw), with per-example think/answer nats and token counts for test and reach, the context length per target, the demo pids and token counts, timing and peak memory.
  - Also `heartbeat.json` and `DONE`.
  - Runs are resumable; the exit code is 0 only when every requested row exists, and 3 on a fatal error, which also writes `ERROR.json`.

### Readouts (`analyze_icl.py`)

These are computed per checkpoint, target set (test, reach) and segment (total, think, answer):

- **L(k):** nats per response token, summed over targets and draws.
- **gain(k) = L(0) − L(k).**
- **share(k) = (L(0, t) − L(k, t)) / (L(0, t) − L(0, s2300)):** the share of the gap to the final model closed by k demonstrations. It is undefined at `s2300`, where the raw gain is the readout.
- **L(k) per draw.**
- **Intervals:** 95 % percentile bootstrap over targets and, for k > 0, over draws. gain and share pair the same resampled targets.
- **Trends:** Spearman correlation with RL step.
- **Reading rule:** the review's rule ("elicitable in context" if share(4) ≥ 0.25 with the lower bound above 0) is **descriptive and not pre-registered**.
- **Figure (`fig_icl.py`):**
  - Panels (a, b): L(k) over RL step, one line per k, for total and answer.
  - Panels (c, d): L against k, one line per checkpoint.

## Deviations from review R2, set by the coordinator's scope

| Review R2 | This build |
|---|---|
| k up to 8 | k up to 4 |
| 3 demo sets | 2 demo draws |
| 42 reach responses | 38 (the pool's genuine reach rows under the 6,144 cap) |

Not built:

- control (a), the unrelated-domain demonstrations;
- control (b), the n-gram overlap split;
- the 8-example in-context prequential code;
- the behavioural arm.

Without control (a), a gain cannot be told apart from generic "thinking style" imitation. Without control (b), it cannot be told apart from n-gram copying.

## CPU checks (observed)

- Tests (`pytest -k "icl or partial"`): 4 passed, on CPU on serrano.
  - k = 0 reproduces `step0.json` with a difference of 0.0.
  - The demo sets are identical across two checkpoints, nested, disjoint across draws and genuine-only, with no target among them.
  - Runs resume.
  - The analysis matches hand-computed L0, gain and share.
  - The figure renders.
  - The prefix cache matches the full forward.
  - The context matches the template text.
- CLI on the tiny real-vocabulary model, fp32, CPU, real pool:
  - `run --ks 0,1 --draws 1 --prefix-cache` wrote 2 rows and DONE.
  - `bench` measured a 21,848-token longest context and a 15,627-token longest prefix. The cached and uncached results differed by 2.0e-5 nats.
  - Timings and projections on CPU say nothing about the H100.

## 32B cost (inferred, to be replaced by the pod `bench`)

The estimate rests on step-0 scoring on the dpo H100 pod: 3.9–4.0 k tokens/s for 135 k, 179 k and 263 k tokens in 34.1 s, 45.6 s and 64.9 s.

| | Tokens per checkpoint | Estimated time per checkpoint |
|---|---|---|
| Without prefix cache | 5.39 M | ≈ 23 min, plus about 10 % attention overhead at 10–22 k context |
| With `--prefix-cache` | ≈ 2.4 M (tails, k = 0, six prefixes, 12 cache checks) | ≈ 10–12 min |

Peak memory is not measured yet. Two estimates:

- About 71 GB for the longest uncached forward.
- About 73–76 GB with a 15.6 k-token prefix cache resident plus its copy. The KV cache is about 0.26 MB per token if every layer stores every position.

`bench` prints both peaks in its `MEMORY` line.
