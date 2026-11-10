---
written_on: 2026-11-10
written_by: agent:claude (orchestrator; not the agent that built the pipeline)
---

# fable_rl pipeline: held-out and sealed test results

Run by the orchestrator on 2026-11-10, 11:19 am to 2:54 pm PT, on one temescal GPU, after the
pipeline was handed back and its checkpoints were fixed. The building agent never saw these sets.
Plan written before the first run: `code/experiments/current/fable_rl_pipeline_20261109/testside/TEST_PLAN.md`
(also the queue script and the premise-count script). Score reports, no proof text:
`/scratch/cowifornia/nd-rl/fable_rl/testside/results/{fable_full,fable_plain,fable_fullL,fable_ablplain_s0}/report.md`.

All numbers are pass@1 / pass@5 as % of the set: k independent attempts per theorem, unbiased
per-theorem estimate, mean over theorems, then over the 3 seeds. Every proof was re-checked by Lean
against our own statements. Differences are in points with a 95% paired bootstrap interval over
theorems. "Full" is the pipeline's verifier-screened solver (4 tries per attempt); "plain" is one
rollout at temperature 0.8 with no verifier. No wall-clock cut was applied to the solver
(`SOLVER_MAX_SECONDS=0`), because the references had none.

## Observed

### Sealed textbook72 (32 attempts per seed)

| reference length | n | full | plain | best model (v2 GRPO L3 end) | full − best, pass@1 |
|---|---|---|---|---|---|
| all | 72 | 51.4 / 60.9 | 31.3 / 42.2 | 74.0 / 82.3 | −22.6 [−30.4, −15.1] |
| 1–5 | 5 | 100.0 / 100.0 | | 99.6 / 100.0 | +0.4 [0.0, +1.3] |
| 6–10 | 24 | 74.5 / 83.3 | | 92.1 / 95.6 | −17.6 [−31.1, −5.5] |
| 11–15 | 16 | 38.7 / 52.2 | | 73.2 / 85.6 | −34.5 [−48.3, −21.7] |
| 16+ | 13 | 10.5 / 18.7 | | 26.0 / 45.4 | −15.5 [−27.2, −5.4] |
| no reference length | 14 | 47.2 / 57.8 | | 79.5 / 83.7 | −32.3 [−54.8, −10.6] |

Per seed, full, all 72: 46.9 / 60.5 / 46.9 pass@1. Other references on all 72: best model's base
38.6 / 55.0 (full − base +12.9 [+6.3, +19.4] pass@1); Dan's expert iteration (la_T1_best12_s0_r8)
38.5 / 54.4; Dan's full-corpus base 16.1 / 27.0. Generated tokens per attempt: full 1,667 to 2,012,
plain 156 to 181.

### 832 held-out transfer theorems (32 attempts per seed)

| group | n | full | plain | Dan's expert iteration | Dan's full-corpus base |
|---|---|---|---|---|---|
| all | 832 | 83.8 / 90.6 | 72.4 / 83.7 | 66.5 / 81.9 | 15.2 / 27.6 |
| the 106 strict | 106 | 79.9 / 84.8 | 71.5 / 78.1 | 29.1 / 35.7 | 3.6 / 8.4 |
| 7 lines | 241 | 90.5 / 95.3 | | 76.3 / 91.6 | 29.3 / 47.3 |
| 8 | 85 | 93.7 / 98.1 | | 81.9 / 90.6 | 17.5 / 38.1 |
| 9 | 166 | 89.6 / 94.6 | | 58.8 / 70.4 | 12.8 / 24.6 |
| 10 | 206 | 80.2 / 89.2 | | 65.1 / 85.1 | 8.3 / 17.9 |
| 11–12 | 115 | 68.7 / 79.8 | | 54.6 / 72.6 | 1.7 / 4.0 |
| 13–14 | 19 | 35.0 / 43.0 | | 26.2 / 41.0 | 1.2 / 4.4 |

Full − Dan's expert iteration: +17.4 [+15.0, +19.7] pass@1, +8.7 [+6.4, +11.0] pass@5; on the 106,
+50.8 [+41.6, +59.6] pass@1. Plain − Dan's expert iteration: +5.9 [+3.6, +8.3] pass@1, +1.8
[−0.5, +4.1] pass@5. Per seed, full, all: 84.3 / 88.3 / 79.0 pass@1. Tokens per attempt: full 1,005
to 1,258, plain 197 to 209. The best model has no rows on this set.

### Longer than anything in x or y

| set | mode, attempts | n | pipeline | best model | best model's base | Dan's expert iteration |
|---|---|---|---|---|---|---|
| 13–20 lines | full, 8 | 320 (fixed subsample) | 25.2 / 40.1 | 33.5 / 59.2 | 5.6 / 16.5 | 13.9 / 28.9 |
| 13–20 lines | plain, 32 | 986 | 4.5 / 10.5 | 32.1 / 57.8 | 5.5 / 16.2 | 11.8 / 25.8 |
| 21–30 lines | full, 8 | 320 (fixed subsample) | 1.4 / 4.1 | 7.1 / 19.1 | 0.1 / 0.7 | 1.0 / 2.9 |
| 21–30 lines | plain, 32 | 1,000 | 0.1 / 0.3 | 7.2 / 19.4 | 0.3 / 1.3 | 0.9 / 2.7 |

Full − best model: 13–20 lines −8.2 [−11.6, −4.9] pass@1 and −19.1 [−23.4, −15.0] pass@5; 21–30
lines −5.7 [−7.0, −4.3] and −15.0 [−18.0, −12.1]. By single length the pipeline is never better
than the best model with an interval excluding 0; at 13, 14, 16 and 18 lines the pass@1 interval
includes 0, and from 19 lines on it is at 7–8% against 19–24%. Full − Dan's expert iteration:
+11.3 [+8.7, +13.9] pass@1 on 13–20 lines, +0.4 [−0.2, +1.1] on 21–30. Tokens per attempt: full
2,707 to 3,309 (13–20) and 3,716 to 4,449 (21–30); plain 280 to 320.

### Ablation (plain expert iteration from seed 0's base; one seed, plain mode, 32 attempts)

| set | ablation | pipeline seed 0, plain |
|---|---|---|
| 832, all | 64.2 / 74.4 | 73.2 / 83.8 |
| 832, the 106 | 30.5 / 36.5 | 70.2 / 75.4 |
| textbook72 | 25.5 / 36.8 | 27.5 / 38.8 |
| 13–20 lines | 2.1 / 5.4 | 5.1 / 10.8 |
| 21–30 lines | 0.0 / 0.0 | 0.1 / 0.2 |

### Checks

- Solver's own verdict against Lean, full mode, 832, seed 0: 17 of 26,624 attempts claimed and
  rejected; 3,512 not claimed but accepted (the unscreened fallback try). Plain mode has no
  checker: its "yes" was rejected by Lean in 11–15% of attempts on the 832, 37–45% on 13–20 lines
  and 18–28% on 21–30 lines (range over seeds).
- No attempt was missing, duplicated or scored against a different statement in any run.
- Premise counts: x has at most 3 premises. Only one test theorem (in textbook72) has more than 3,
  so the planned extra row carries no information (full 1.0 / 5.2 on it; every reference 0.0).
- Incident: the seed-2 full-mode long-set run was killed at 2:30 pm PT by our watchdog when another
  user's process started on the same GPU; it was restarted unchanged before any of its scores were
  read (see TEST_PLAN.md).

## Interpretation (orchestrator's)

- The pipeline is the strongest model we have measured on the 832, which are drawn like y, and
  most of its margin on the 106 comes from what the ablation lacks (the step checker and subgoals
  inside sampling): 70.2% against 30.5% pass@1 in plain mode, one seed each.
- It does not transfer: on the sealed textbook set it is 22.6 points pass@1 below the best model
  and never better at any length; on longer proofs it is below the best model at both ranges, and
  near zero past 21 lines.
- The verifier-screened search buys 11 points pass@1 on the 832 and 20 on textbook72, at 5 to 12
  times the tokens (on 13–20 lines: 25.2 against 4.5, on different theorem subsets and attempt
  counts). Without it the model is below Dan's expert iteration on both long sets and on textbook72
  (−7.3 [−14.8, +0.0] pass@1 there).

## Do not over-read

- The best model was trained from a different base with a different method (GRPO with a thinking
  scratchpad); this is a comparison of two finished systems, not of two RL algorithms on equal
  footing.
- Full mode on the long sets used 8 attempts and a 320-theorem subsample per set, chosen on run
  time before any result; pass@5 from 8 attempts is unbiased but noisier than from 32.
- Seeds differ a lot (832: 79.0 to 88.3 pass@1; textbook72: 46.9 to 60.5). The ablation is one seed.
- Token counts are solver-reported and not comparable with the best model's, which was not
  re-measured here.
