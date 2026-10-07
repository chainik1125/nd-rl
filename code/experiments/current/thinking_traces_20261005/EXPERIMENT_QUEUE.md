---
written_on: 2026-10-07
written_by: agent:claude
---

# Experiment queue: what is running, queued, held and done

This is the single tracking list for the thinking-traces runs. Update it whenever a job starts, finishes or
changes; Git keeps the history. All paths are under `/scratch/cowifornia/nd-rl/think/runs/` and all times are PDT.

**Matched budget.** One EI round is 5,000 × 32 = 160k attempts, and 39 GRPO iterations are 256 × 16 × 39 =
160k. So EI r1 ↔ GRPO it39 and EI r2 ↔ GRPO it78.

## Running (2026-10-07)

| what | where | GPU | stops | notes |
|---|---|---|---|---|
| GRPO no-think (T0_s0), fixed code | `restart3_grpofix/grpo_T0` | 2 | it78, ~1:55 pm (`stop_at_78.sh`) | restarted from scratch after serrano's crash |
| GRPO thinking with effort levels (T1v1_s0) | `restart3_grpofix/grpo_T1v1e` | 3 | it78, ~3:45 pm | budgets 0/16/64/256 |
| matched evals: EI r1, r2 vs GRPO it39, it78, both arms | `eval/matched/`, log `logs/matched_eval.log` | 1, 4 | ALL DONE ~5–6 pm | sets: hard12_test_1k (k 16), textbook72 (k 64), test_13_20 if ready; thinking at B 0/64/256 |
| 13–20-line test set (agent) | `pools/test_13_20/` | 0 (lock) | — | the 21–30 set is at the floor |

## Queued (pipelines in `followup_20261007/`, logs `logs/followup_gpu*.log`)

Every queued GRPO run keeps all of its checkpoints, one every 5 iterations (20k attempts), via
`--keep_rolling 100`. That gives full learning curves. The two running GRPO runs keep only it39 and it78, plus
any alarm checkpoints.

1. **Thinking GRPO, one budget, no effort token** (`grpo_T1v1_b256`, T1v1_s0, B 256, to it78). GPU 2 after
   `grpo_T0` exits. This is the cleanest test of RL'd thinking against RL'd no-think, because updates are not
   split over four effort levels.
2. **The same with a token cost** (`grpo_T1v1_b256_lp02`, `--len_penalty 0.2` per 1k sampled tokens on
   successes). GPU 3 after `grpo_T1v1e` exits. Question: can RL learn to think only where it helps?
3. **Second seed of the main pair** (`grpo_T0_s1` from T0_s1 and `grpo_T1v1e_s1` from T1v1_s1, RL seed 1, to
   it78). GPUs 1 and 4 after the matched evals log ALL DONE.
4. **Budget forcing, eval only** (s1-style, no new training). The sampler bans `</think>` until `--min_think`
   tokens per block, with the cap raised to B. Configs (B, min): (256, 0), (256, 128), (512, 256), (1024, 512).
   - Models: Stage-1 T1v1_s0 and `grpo_T1v1_b256` it78. Both have no effort token, so B is never shown to the model.
   - Sets: textbook72 (k 32) and the first 300 items of hard12_test_1k (k 16).
   - Runs on GPU 2 after item 1.

## Held

- **hard12r** (≤12-line targets that stay hard after RL). Pilot done, 0.3 GPU-h; see
  [pools/hard12r/README.md](pools/hard12r/README.md).
  - RL'd no-think GRPO (it35) solves 7–8 of 8 on 88.5% of fresh ≤12-line generator candidates, and on 85.9%
    of hard12's spare in-band targets. Only 2.7% of fresh candidates land at 1–2 of 8.
  - Structural features barely move this (3–9% at ≤2/8 across goal types, depth and length).
  - So ≤12-line headroom after RL is close to exhausted. Any test split must be an unfiltered random sample.
- **A harder pool selected by model-independent features.** Revisit on the same condition.
- **EI with smaller rounds** (k 8 per target, 40k attempts per round, a fine-tune after each). This gives a
  checkpoint every 10 GRPO iterations, so EI and GRPO can be compared below one full round. Decide after the
  r1 vs it39 evals: if EI is already near ceiling at r1, the finer curve adds little. Cost: about 4 h no-think
  and 6 h thinking, on two GPUs.

## Done (details in the linked notes)

- Stage 1 (4 arms × 3 seeds), goal-stack and filler controls, and the hard12 seed control: see
  [STAGE1_STATS.md](STAGE1_STATS.md) and [`stage1_stats/`](stage1_stats/).
- Adaptivity across RL checkpoints: [ADAPTIVITY.md](ADAPTIVITY.md).
- GRPO scoring bug (fixed in 7b082b7) and overlong-masking drift: [GRPO_COLLAPSE_AUDIT.md](GRPO_COLLAPSE_AUDIT.md).
- EI on hard12: both arms stopped on 2026-10-07 at Leon's call (stop early). Checkpoints r1–r4 are kept (r5 for
  no-think).
- The 21–30-line sealed test set: [test_21_30/README.md](test_21_30/README.md).
