---
written_on: 2026-11-25
written_by: agent:claude (EDL build agent)
papers: []
---

# Stage 1: EDL on Olmo 3.1 Think 32B checkpoints (method note)

## Summary

This note describes how Stage 1 measures the excess description length (EDL) of the final model's own successful
responses at each RL checkpoint, and what each output field means. It describes the method only and contains no
results. The data are one stored strict-pass response of `s2300` per prompt. Each checkpoint gets a single online pass
of LoRA fine-tuning over those responses. Contract: [SPEC.md](SPEC.md) Stage 1. Fixed readings:
[PREREG.md](PREREG.md), Amendments 1–2.

Code in this folder:

- `build_pool.py`: the data.
- `olmo_edl.py`: the runner (`run`, `bench`).
- `analyze_edl.py`: the readouts.
- `fig_edl.py`: the figure.
- `test_olmo_edl.py`: CPU tests.
- `olmo_template.py`: Dan's prompt encoding, copied verbatim.

## Definition used

For one data order x_1..x_n (n = 64), with l_i the online loss of example i scored before the update on it:

    EDL(n') = sum_{i<=n'} l_i - n' * L_test(theta_n')

- l_i is the NLL in nats of the stored response tokens given the templated prompt, scored before the update on x_i
  (θ_{i-1}).
- L_test is the mean of the same NLL over the m = 32 test responses.

This is the paper's definition (Donoway et al. 2026, arXiv 2601.04728) and `code/ndbench/metrics.edl`; a test checks
the two agree.

Two unit conventions are reported, because train and test responses differ in length:

- **Per-example sums** (the paper's form): the formula above. Per token, it is divided by the online tokens
  sum_{i<=n'} T_i.
- **Length-normalised** (the lead form; PREREG states its units as nats per response token):
  EDL/token(n') = sum_{i<=n'} l_i / sum_{i<=n'} T_i  -  sum_j L_j / sum_j T_j. In nats, multiply by the online tokens.

**EDL\*** (PREREG Amendment 2): pick the lr with the smallest mean-over-orders online code length sum_i l_i, take
EDL(64) at that lr averaged over orders, and add ln(number of lrs). This is ndbench's `summarise_edl` rule.

- Secondary: SPEC's min over lr of EDL(64) + ln(number of lrs), with the lr re-chosen inside every bootstrap replicate.

**Segments.**

- `think` is the stored ids up to and including the `</think>` marker. The marker is ids 524 `</`, 27963 `think`, and
  then 1363 `>\n\n` or 397 `>\n`. The last marker counts, as in Dan's grader.
- `answer` is the rest, EOS included.
- Each segment gets its own EDL\*: the lr is chosen by that segment's own code length, and ln k is added.
- `EDL_at_total_lr` decomposes the total at the total's lr; think + answer = EDL\*(total) − ln k.
- The answer segment is a first-class readout everywhere: tables, Spearman trends, checks C1–C3, and figure panel (d).
- Additivity: in the per-example form, think + answer = total. In the length-normalised form the segment EDLs do
  not sum to the total, because each segment uses its own per-token rates.

**Intervals.** 95 % percentile bootstrap over test prompts. Where a checkpoint has two orders, the orders are
resampled jointly with the prompts. The online stream is a fixed sequence and is not resampled, so its sampling noise
appears only through the order replicate. The gap share pairs the test prompts of the checkpoint and of `s2300`.

**Secondary readout, "convergence" residual** (orchestrator, after the companion paper "Bits That Count"):

- `EDL_conv = sum_i l_i - n * L_test_conv`, where L_test_conv is the test loss after 3 more epochs over the same 64
  examples.
- The 3 epochs use the same adapter, AdamW state and constant lr.
- They run for one row only (`--conv-rows best`, the shortest-code-length lr at order 0), or for every row
  (`--conv-rows all`).

## Protocol and recipe

- **Data** (`build_pool.py`):
  - Source draws: every stored plain-prompt `s2300` job (revision `main` = `step_2300`). The jobs are listed in
    `pool/MANIFEST.json`.
  - A draw must be a strict pass (finish = stop) and must contain `</think>`.
  - Draws labelled gamed are excluded, as are prompts whose only passes are gamed. Labels come from five of Dan's hand
    gates, plus identical-answer inheritance; a conflict counts as gamed.
  - Response cap: 6144 tokens.
  - One response per prompt: a genuine-labelled one if there is one, otherwise a seeded pick.
  - Sets:
    - train: 64 prompts. Arm A minus reach, seeded split.
    - test: 32 prompts, same rule.
    - reach: 38 prompts. Reach prompts whose pick is genuine.
- **Prompt ids**:
  - Built with Dan's encoding: the olmo_thinker template with add_generation_prompt, no BOS, pad ids dropped.
  - They are stored in the pool, and the runner re-tokenises and asserts equality.
  - They are identical under the dpo, s2300 and Olmo-3-7B-Think tokenizers (tokenizer.json blob `5fe17212`).
  - 14 of them were checked against the sampler's own hashes in B_s2300's meta.
- **Learner** (`olmo_edl.py`):
  - PEFT LoRA, r 16, alpha 32, dropout 0, bias none, on Dan's targets (q, k, v, o, gate, up, down).
  - bf16 base; fp32 LoRA (asserted).
  - AdamW (0.9, 0.999, eps 1e-8, weight decay 0), constant lr, grad clip 1.0, batch 1, one step per example.
  - Training loss: mean token cross-entropy over the response tokens, as in Dan's `lora.py`.
  - Gradient checkpointing (non-reentrant) during training steps only.
- **Per (lr, order)**:
  - The LoRA is reset: A gets PEFT's kaiming init from a seed fixed by (seed, order); B = 0, so the adapter is exactly
    inert. The optimizer is fresh.
  - For i = 1..64: score x_i in eval mode under `torch.inference_mode`, then take one training step.
  - The test set is scored after n' ∈ {8, 16, 32, 64} steps, and the reach set after 64.
  - Grid: lr ∈ {3e-5, 1e-4, 3e-4}. Two orders at dpo, s0500 and s2300; one at the others.
- **Step 0**:
  - The train stream, test and reach under the untouched checkpoint are scored once per checkpoint; this is the
    likelihood readout L_0.
  - Every run's i = 1 online score must equal its step-0 value. The difference is recorded as
    `reset_check.abs_diff`; it is 0.0 in the CPU tests and in the CPU CLI runs.
- **Scoring**:
  - Hidden states come from the decoder. The LM head is applied to the response positions only, in chunks, with bf16
    logits upcast to fp32. This matches HF's causal-LM loss and Dan's `nll()`; a test compares against `model(labels=)`.
  - Test and reach sets are scored in length-sorted right-padded batches (`--eval-batch-tokens`, default 16384). The
    same batching is used at every eval point. Batched and batch-1 scores agree to 1e-4 relative in fp32, including
    past the sliding window.

## Deviations

From Dan's `lora.py` recipe:

- Dropout 0 instead of his 0.05, so the online score does not depend on a random mask (SPEC).
- Constant lr with no warm-up or cosine, one step per example, and one pass, instead of 4 epochs.
- An lr grid instead of his single 1e-4.
- Per-segment sums instead of the mean `.loss` only.
- No merged model.

From the EDL paper:

- A single online pass with one step per example, rather than training to convergence at every prefix. The
  `EDL_conv` readout approximates the converged residual at n = 64 only.
- LoRA rather than full fine-tuning.
- The test residual is L_test after the same online pass.
- The training responses are the final model's own samples. So EDL at `s2300` is a null control (PREREG C1), not a
  capability claim, and every checkpoint's number measures distance to the final model's way of answering these
  prompts.
- Prompts with very long responses are excluded by the 6144-token cap, which biases the prompt set toward shorter
  responses. 28 arm-A prompts and 6 reach prompts were dropped for this reason.

## Output fields (`runs/<ckpt>/edl/`)

- `edl_rows.jsonl`: one line per (ckpt, lr, order), schema `olmo_edl_row/1`.
  - Identity: `ckpt`, `identity` (model path, revision, commit, config sha256), `pool_manifest_sha256`, `seed`,
    `recipe`, `env` (library versions, GPU), `train_pids`.
  - Run: `lr`, `order`, `perm` (stream order over the pool's train rows), `init_seed`, and `status` (ok / nonfinite)
    with `error`.
  - `online`: in stream order — `pids`, `think_nats`, `answer_nats`, `n_think`, `n_answer` (per example), plus
    `train_loss` (mean token CE of the step), `grad_norm` (before clipping), `score_secs` and `train_secs`.
  - `evals["<n'>"]["test"|"reach"]`: per-example `think_nats` and `answer_nats` in the order of `test.pids` /
    `reach.pids`. `evals["0"]` is copied from step 0.
  - `reset_check`, `times` (per phase), `peak_gpu_gb`.
  - With `--conv-rows all`, also `conv`: epochs, per-epoch losses, and test / reach after the extra epochs.
- `edl_rows.jsonl`, schema `olmo_edl_conv/1` (`--conv-rows best`): one line, the extra epochs of the best order-0 row.
  It was loaded from that row's saved θ_64 adapter and AdamW state, with no re-run.
  - `load_check_max_abs_diff_test_nats` re-scores the test set after the load and compares it with the row's
    L_test(64).
  - `mode` says how the line was made.
- `step0.json`: per set (train in pool order, test, reach) — `pids`, `think_nats`, `answer_nats`, `n_think`,
  `n_answer`, plus timing and peak memory.
- `step0_tokens.npz`: per pool row, under the untouched checkpoint.
  - Per row: `pid`, `set`, `think_end`, `n_tok`, `offset`.
  - Per response position: the stored `token`, its `nll` (fp32), its `rank` (0 = top-1), the next-token `entropy`
    (fp16), and the top-20 ids (int32) and log-probs (fp16).
  - Loader: `olmo_edl.load_token_stats`.
- `heartbeat.json` is rewritten at least every 60 s. Its `last_progress_utc` moves only when the run makes progress.
- `DONE`: written when every requested row exists. `ERROR.json` on a fatal error.
- `adapters/best_order0/`: θ_64 of the shortest-code-length order-0 row, plus `optimizer.pt` with `--conv-rows best`.
  `BEST.json` names the row.
- Exit codes: 0 when all rows are present; 2 when some are missing; 3 on a fatal error (OOM, pool or tokenizer
  mismatch). There is never any truncation.

`analyze_edl.py` writes `/scratch/cowifornia/nd-rl/olmo_elicit/edl_summary.{json,csv}`. Per checkpoint:

- `L0` by set (train / test / reach / train+test) and segment.
- `EDL_star[conv][rule][segment]`: nats and per token, with intervals and the lr used.
- `EDL_at_total_lr`.
- `curve[conv][segment]`: EDL per token at n' = 8, 16, 32, 64, with intervals and a shape word.
- `L64`, `C2_test_drop`, `per_order_EDL64` (for C3).
- `EDL_conv` and `L_conv`.
- `gap_share` and `gap_share_conv`: (L_0 − L_x)/(L_0 − L_0(s2300)), per token on the test set.

Across checkpoints: `trends` (Spearman with RL step) and `checks` (C1–C3 for total and answer). The CSV is the same
content flattened to (ckpt, metric, convention, segment, value, lo, hi, unit, note).

## Checks run

- 14 CPU tests (`test_olmo_edl.py`; venv `/scratch/cowifornia/venvs/olmo-edl`: torch 2.8.0, transformers 4.57.1,
  peft 0.17.1, the pod's versions):
  - EDL arithmetic against `ndbench.metrics.edl`.
  - Both lr rules by hand.
  - Token NLL against HF's `.loss`.
  - Segment split.
  - Batched against batch-1 scoring.
  - The adapter reset restores step-0 scores exactly.
  - Null control: a model whose loss cannot change gives EDL = 0 in the normalised form. The per-example form is 0 only
    when train and test lengths are equal; otherwise it charges the length difference.
  - Resume skips finished rows and reuses step 0.
  - A non-finite row is recorded and the next row runs.
  - Extra epochs.
  - `--conv-rows best` equals the in-place continuation.
  - CE checkpointing gives the same loss and gradients.
  - Token-stats sanity checks.
  - Pool-builder rules on a synthetic clone.
- CPU end-to-end CLI runs (`bench`, and `run` with conv-best) on a tiny random Olmo3 with the real vocabulary,
  tokenizer and pool.
- Lab GPU smoke test, 2026-11-25 (ynez RTX 3090; Olmo-3-7B-Think; 6 train, 4 test and 3 reach real rows with
  responses of at most 2048 tokens; one lr; `--conv-rows best`):
  - Exit 0.
  - `reset_check` and `load_check` are both exactly 0 in bf16.
  - Training step 2.3 s at 2.1k tokens, peak 16.9 GB on a 13.7 GB footprint.
  - Outputs: `/scratch/cowifornia/nd-rl/olmo_elicit/smoke7b/`. Details in [log.md](log.md).
- Memory knobs for an 80 GB card (defaults unchanged): `--ce-checkpoint`, `--loss-chunk 512`,
  `--eval-batch-tokens 0`. `--max-memory-gb 78` emulates the cap on an H200.
