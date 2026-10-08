---
written_on: 2026-10-07
written_by: agent:claude
---

# Experiment queue: what is running, queued, held and done

This is the single tracking list for the thinking-traces runs. Update it whenever a job starts, finishes or
changes; Git keeps the history. All paths are under `/scratch/cowifornia/nd-rl/think/runs/` and all times are PDT.

**Matched budget.** One EI round is 5,000 × 32 = 160k attempts, and 39 GRPO iterations are 256 × 16 × 39 =
160k. So EI r1 ↔ GRPO it39 and EI r2 ↔ GRPO it78.

## Running and queued (updated 2026-10-08, 2:45 pm PDT; trimmed at Leon's prompt to short, decision-tied runs)

Every GPU job answers one question with a pre-registered decision rule. Pilots come before extensions.

| job | GPU | ETA | question → decision rule |
|---|---|---|---|
| Stage-1 9.6M: honest-search thinking S1g3 ×2 seeds; no-think control T0S ×2 | 1–4 | ~4:30 pm | Honest vs staged traces. **If** S1g3 B256 ≥ T1v1g4 B256 on 13–20 pass@1 and S1g3 − T0S > 0 (pooled interval), honest traces become the default Stage 1; **else** keep the staged graded traces. |
| Step-redraw read of EI thinking r2 (13–20) | 0 | ~3:15 pm | Completes the logical-arm table. |
| GRPO graded continued +39 iterations (it117 ≙ EI r3); 13–20, 21–30 and hard12 at B256 | 0 | ~5:30 pm | GRPO vs EI at matched attempts. **If** GRPO it117 is within 2 pp of EI r3 on 13–20 pass@1, call it a tie; **else** the RL method follows the winner. (Trimmed from +78.) |
| Size sweep 35.6M: S1g3 and T0S, seed 0; hard12, textbook72 and 13–20 | 3–4 after T0S | ~9 pm | Is capacity the limit? **If** the thinking gain (S1g3 − T0S on 13–20 pass@1) grows by ≥ 3 pp over 9.6M, or the invalid-step rate falls by ≥ 25% relative, run 90.6M; **else** stop the sweep. |
| CPU agents: hard712 pool; search completeness on hard theorems; GRPO safeguards + E1 script | — | — | — |

**Next, as pilots** (start when their inputs land):
- **EI pilot, 1 round per arm** instead of 3: hard712-plain, hard712-logical and hard12-logical, against the existing hard12-plain r1. About 1.7 GPU-h each, in parallel.
  - Extend only arms that beat hard12-plain r1 by ≥ 2 pp on 13–20 pass@1.
- **E1 collapse confirmation: deferred.** It only matters if GRPO stays the RL method; it runs only if GRPO wins the it117 comparison.
- **Dropped:** textbook72 logical reads (dev58 is now a design set, and it is small).

## Proposed (not run; waiting for Leon, 30-min auto-run rule applies)

From [RL_RESULTS.md](RL_RESULTS.md#proposals-not-run-need-leons-call):
1. **Second seeds of the graded arm and of one budget + token cost.** About 3 GPU-h each, plus 1.5 GPU-h of evals.
   Seed spread is currently as large as the differences between variants.
2. **Restart the seed-0 effort run from it39**, once with the same settings and once at half the learning rate,
   to see whether the it48 clip-low spike and the 13–20 drop recur.

## Done 2026-10-07 (write-up: [RL_RESULTS.md](RL_RESULTS.md); tables: [matched_eval/final_tables.md](matched_eval/final_tables.md))

- **Matched EI vs GRPO** (r1/r2 vs it39/it78, both arms, seed 0) on hard12_test, textbook72 and t1320 (986 valid
  items).
- **Seed-1 GRPO pair**, plus the seed-1 13–20 learning curve (it10–75).
- **Single-budget thinking GRPO** (`grpo_T1v1_b256`) and the token-cost variant (`grpo_T1v1_b256_lp02`), both to
  it78.
- **Budget forcing (s1-style).** Forcing any minimum think length lowers accuracy.
- **Locating the 13–20 drop in the seed-0 effort run.** The gain is gone by it49 and never recovers.
- **Graded-length traces** (`graded_traces/`), Stage-1 T1v1g4, and GRPO from it, plain and with LCPO. All evals
  are done.
- **LCPO reward** implemented (`grpo.py`, 4608e60).
- **Process fix:** eval queues now count an eval as done only when every budget's rows file exists
  (`followup_20261007/lib.sh` `evdone`). `summary.json` is rewritten after each budget, so it is not enough.

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
