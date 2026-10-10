---
written_on: 2026-10-07
written_by: agent:claude
---

# Experiment queue: what is running, queued, held and done

This is the single tracking list for the thinking-traces runs. Update it whenever a job starts, finishes or
changes; Git keeps the history. All paths are under `/scratch/cowifornia/nd-rl/think/runs/` and all times are PDT.

**Matched budget.** One EI round is 5,000 × 32 = 160k attempts, and 39 GRPO iterations are 256 × 16 × 39 =
160k. So EI r1 ↔ GRPO it39 and EI r2 ↔ GRPO it78.

## Running (updated 2026-10-09, 5:50 pm PDT)

**Goal since the 2026-10-09 group meeting:** a better GRPO recipe. Leon's answers (2026-10-09, 4:30 pm): GRPO +
verifier = verifier-guided rollouts (redraw steps a Lean-sound check rejects) plus a penalty on each rejected draft;
curriculum 10–12 lines, then 13–15; GRPO on both the thinking (S1g3) and no-think (T0S) models; priorities: GRPO vs
EI in a fair setup, what each change buys, then the model-chosen think budget.

| job | GPUs | ETA | why |
|---|---|---|---|
| GRPO + thinking v1 (`grpo_think/line.sh`): S1g3 seeds 0 and 1, GRPO g1 rl_hard12 → g2 hard712 a → g3 hard712 b, 39 iterations each, safeguards on; reads after g2 and g3 (13–20 k 32 B64, textbook72 k 64) | 4 (s0), 2 (s1) | g2 reads ~6:15 pm, g3 reads ~8:45 pm | GRPO-trained thinking vs the EI no-think baseline, today's recipe. Seed 0's g2 hit the long-context canary at it30 and it32: 2 rollbacks, lr × 0.25, then finished cleanly |
| Seed-2 EI pair (`seed2/line.sh`) | 0, 1 | ~8:30–9 pm | third seed of thinking + EI vs the EI baseline |
| Charles held-out set reads (`followup_20261007/charles_reads.sh`): 149 problems of Charles's release 2026-10-02 that share no class with our pools, k 64, old and current models | 3 | ~6:30 pm | larger textbook-like test (Leon: "start using the larger set of Charles data"); EI vs GRPO confusion matrix |

## Queued: GRPO recipe v2 (`grpo_v2/line.sh`, branch `leon/grpo-v2`)

Launch about 9 pm once the curriculum pools (`pools/curriculum/L2`, `L3`) pass `grpo_v2/preflight.py` and the
review fixes are in (step cap 96 for GRPO and EI, renaming-class preflight, accepted-token normalisation of the
loss). Stages: L2 (10–12 lines) then L3 (13–15), 39 GRPO iterations or one k-32 EI round each. Reads after L3:
13–20 (the 16–20 part is out of distribution) k 16, textbook72 k 64, Charles 173 (report the clean 149) k 64,
hand_proved_14 k 64, 1,000 of 21–30 k 8; plain arm for all, guided arm on 13–20, textbook72 and Charles.
About 78 GPU-hours, so two waves on five GPUs.

| wave | GPU queue | arms |
|---|---|---|
| 1 (~9 pm → ~4–5 am) | one per GPU | seed 0: `grpo_think_v`, `ei_think_v`, `grpo_think`, `grpo_nothink_v`, `ei_nothink_v` |
| 2 (~4–5 am → ~noon) | same GPUs | `ei_think` seed 0; seed 1 of `grpo_think_v`, `ei_think_v`, `grpo_nothink_v`, `ei_nothink_v` |

Then: the model-chosen think budget (`leon/thinking-auto`) on top of the winning recipe.

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
