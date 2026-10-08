---
written_on: 2026-10-07
written_by: agent:claude
---

# Experiment queue: what is running, queued, held and done

This is the single tracking list for the thinking-traces runs. Update it whenever a job starts, finishes or
changes; Git keeps the history. All paths are under `/scratch/cowifornia/nd-rl/think/runs/` and all times are PDT.

**Matched budget.** One EI round is 5,000 × 32 = 160k attempts, and 39 GRPO iterations are 256 × 16 × 39 =
160k. So EI r1 ↔ GRPO it39 and EI r2 ↔ GRPO it78.

## Running (updated 2026-10-07, 6:50 pm PDT)

| what | where | GPU | ETA | notes |
|---|---|---|---|---|
| GRPO from graded Stage-1 (T1v1g4), plain effort | `restart3_grpofix/grpo_T1v1g4e` | 1 | ~9 pm | to it78; keeps all rolling ckpts |
| GRPO from T1v1g4 with the LCPO length reward | `restart3_grpofix/grpo_T1v1g4e_lcpo` | 4 | ~9:15 pm | `--len_target_coef 0.2 --len_target_mode exact --len_target_frac 0.5` |
| seed-1 matched evals, it39 and it78 | `eval/matched/*s1*`, `followup_20261007/gpu2c.sh` | 2 | ~8 pm | |
| seed-1 13–20 learning curve, it10–75 at B0 and B256 | `eval/matched/curve/`, `gpu3_curve.sh` | 3 | ~8 pm | auto-run under the 30-min rule |
| graded Stage-1 evals on t1320, then the token-cost run's it78 evals | `graded_rl.sh`, `gpu0d.sh` | 0 | ~8 pm | GPU-0 queues are now serialised (an OOM collision cost one eval) |

## Done today (2026-10-07; numbers in chat, write-up pending)

- **Matched EI vs GRPO** (r1/r2 vs it39/it78, both arms, seed 0) on hard12_test, textbook72 and t1320 (986 valid items). Tables come from `matched_eval/tables.sh`.
- **Seed-1 GRPO pair.** The no-think runs replicate. The thinking gain on 13–20 at it39 is resolved in both seeds, +4.8 and +1.6 pass@1.
- **Single-budget thinking GRPO** (`grpo_T1v1_b256`) and the token-cost variant (`grpo_T1v1_b256_lp02`), both to it78.
- **Budget forcing (s1-style).** Forcing any minimum think length lowers accuracy, for the Stage-1 and RL'd models alike.
- **Locating the 13–20 drop in the seed-0 effort run.** The gain is gone by it49 and never recovers.
- **Graded-length traces** (`graded_traces/`) and Stage-1 T1v1g4. Effort is monotone. On hard12 the no-think penalty is gone and thinking beats T0 by about 6.5 pp. On textbook72 there is no gain.
- **LCPO reward** implemented (`grpo.py`, 4608e60).

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
