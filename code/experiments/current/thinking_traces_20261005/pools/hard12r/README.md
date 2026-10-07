---
written_on: 2026-10-07
written_by: agent:claude
---

# hard12r: pilot of a post-RL-hard pool (stopped after the pilot)

**Goal (as planned).** Build an RL pool and a sealed test split of 7–12-line theorems that stay hard *after* no-think
RL. The recipe was to filter hard12-style candidates with an RL'd T0: Dan's generator unchanged, formula depth ≤ 5,
no new rules. hard12 is too easy after RL: GRPO takes its sealed test from 42% to 90% pass@1 within 39 iterations.

**Status.** Leon's review changed the scope, so only the pilot ran: the filter model at k = 8 on 6,165
candidates.

- Not run: the k = 32 confirmation, the scale-up and the hardness checks (EI r5, T1v1).
- No pool or test file exists. `runs/pools/hard12r/` holds only `meta.json`, a copy of the pilot report, and
  `work/`.
- **Why it stopped:** the current RL comparison will be judged at a fixed stopping point, on test sets that no model
  selected. A post-RL-hard pool is only a contingency, for when training signal runs out.
- **Not committed:** the code is on branch `leon/thinking`, on top of 8cc9a2c.

## Pilot

### Setup

- **Filter model:** the fixed no-think GRPO run (restart3_grpofix) at iteration 35,
  `runs/restart3_grpofix/grpo_T0_crashed_it38/ckpts/grpo_T0_it35_rolling.pt`.
  - This is the rolling checkpoint of the run that crashed at it 38; that run's folder was renamed.
  - The planned it39 checkpoint does not exist, because the run was restarted from scratch.
- **Measurement:** `code.thinking.eval`, plain arm, B 0, k 8, T 0.8, max_action 512, max_steps 96, max_rej 10, batch
  1024, seed 1, Lean-judged, as in `runs/midrun/gpu0_midrun.sh`. It ran on GPU 0 under the shared lock
  ([`run_eval.sh`](run_eval.sh)).

### Candidates (6,165 measured, 49,320 attempts)

- **prior (4,365):** hard12 candidates that T0_s0 already measured at k = 32 and that are in neither rl_hard12 nor the
  hard12 test set ([`cands.py`](cands.py) `prior`). They are T0-selected, so they are not a sample of the
  generator.
  - spare in-band targets (T0 2–16 of 32): 1,433;
  - every other candidate with T0 ≤ 1 of 32: 2,932.
  - 1 was dropped by the exclusion index.
- **fresh (1,800):** a random 300 per generator config, with hard12 band score ≥ 0.3 and at most 30% per length
  (hard12 `select.py`). They are drawn from two sources:
  - hard12's steered streams, rows never measured (564,682 after exclusion);
  - a new 300 s stream of the same configs, 342,436 distinct (314,872 after exclusion).

  This is the representative sample of the hard12 candidate stream.
- **Exclusion:** [`exclude_r.py`](exclude_r.py) applies hard12's exclusion index (K12, fork data including
  textbook72, OOD sets, rl_hard*), plus rl_hard12, the hard12 test set and its 1k subset, and the top-level sets
  of test_13_20 and test_21_30 (103 files).

### Yield

Each cell is the % of candidates solved ≤ 2, 3–6 or 7–8 times of 8. The "0/8" column is part of "≤ 2/8".

| group | n | ≤ 2/8 | 3–6/8 | 7–8/8 | 0/8 | 1–2/8 | pass@1 |
|---|---|---|---|---|---|---|---|
| **fresh stream** | 1,800 | **5.6** | 5.9 | 88.5 | 2.9 | 2.7 | 91.8 |
| prior: hard12 spare, T0 2–16/32 | 1,433 | 6.2 | 7.9 | 85.9 | 2.9 | 3.3 | 90.4 |
| prior: T0 1/32 | 1,014 | 20.1 | 14.7 | 65.2 | 12.5 | 7.6 | 73.7 |
| prior: T0 0/32 | 1,918 | 34.7 | 17.4 | 47.9 | 25.5 | 9.2 | 58.4 |
| all | 6,165 | 17.2 | 11.4 | 71.4 | 11.5 | 5.7 | 78.1 |

**By feature.** The breakdowns use the fresh stream (n = 1,800), the only group that is not selected on T0. With
n = 300, the binomial SE of a 5.6% share is about 1.3 pp.

| feature | value (n) | ≤ 2/8 | 3–6/8 | 7–8/8 | 0/8 |
|---|---|---|---|---|---|
| reference lines | 7 (255) | 5.5 | 5.1 | 89.4 | 2.4 |
| | 8 (389) | 5.4 | 5.7 | 88.9 | 3.6 |
| | 9 (285) | 2.8 | 6.7 | 90.5 | 0.4 |
| | 10 (233) | 6.0 | 7.3 | 86.7 | 3.0 |
| | 11 (328) | 4.9 | 7.0 | 88.1 | 2.4 |
| | 12 (310) | 9.0 | 3.9 | 87.1 | 5.2 |
| goal type | `>` (912) | 5.2 | 4.2 | 90.7 | 2.4 |
| | `&` (518) | 6.9 | 7.1 | 85.9 | 4.2 |
| | `~` (146) | 8.9 | 9.6 | 81.5 | 4.1 |
| | `v` (126) | 0.8 | 10.3 | 88.9 | 0.0 |
| | atom (90) | 3.3 | 4.4 | 92.2 | 1.1 |
| | F (8) | 12.5 | 0.0 | 87.5 | 12.5 |
| forward-first (first step forward from the premises) | yes (1,021) | 5.3 | 6.1 | 88.6 | 3.2 |
| | no (779) | 6.0 | 5.6 | 88.3 | 2.4 |
| premises jointly inconsistent | yes (529) | 4.5 | 5.5 | 90.0 | 2.1 |
| | no (1,271) | 6.1 | 6.1 | 87.9 | 3.2 |
| formula depth | 2 (124) | 4.8 | 2.4 | 92.7 | 3.2 |
| | 3 (597) | 4.4 | 6.5 | 89.1 | 2.8 |
| | 4 (590) | 6.6 | 5.9 | 87.5 | 2.9 |
| | 5 (486) | 6.2 | 6.0 | 87.9 | 2.9 |
| generator config (300 each) | `fwd_takehome_d2` | 9.0 | 5.3 | 85.7 | 5.3 |
| | `goal_takehome` | 6.0 | 8.3 | 85.7 | 3.3 |
| | `takehome` | 5.3 | 6.3 | 88.3 | 2.0 |
| | `fwd_strict_d1` | 4.7 | 4.3 | 91.0 | 2.3 |
| | `fwd_strict_d2` | 4.3 | 4.3 | 91.3 | 1.7 |
| | `strict_long` | 4.3 | 6.7 | 89.0 | 2.7 |
| hard12 T0 band score | 0.3–0.4 (830) | 4.6 | 4.1 | 91.3 | 2.5 |
| | 0.4–0.5 (390) | 5.1 | 4.6 | 90.3 | 2.3 |
| | 0.5–0.6 (233) | 6.9 | 8.2 | 85.0 | 3.9 |
| | ≥ 0.6 (347) | 7.8 | 10.1 | 82.1 | 3.7 |

The prior group's structural breakdowns are in `meta/pilot_report.json` (`by_group.prior`) and in
`work/pilot_report.md`. Within that group, ≤ 2/8 shows the same flat pattern:

- 19–27% at every length;
- 19.7% vs 23.0% with vs without inconsistent premises;
- 21.8% vs 22.2% forward-first vs not;
- 19.0–23.7% at depths 2–5.

**Observed:**

- On the hard12 candidate stream, the RL'd model solves 7–8 of 8 on 88.5% of candidates. Only 5.6% are at ≤ 2/8, and
  about half of those are at 0/8.
- The hard12 spare targets (T0 2–16/32) behave like the fresh stream (6.2%).
- The only strong predictor is prior T0 failure: T0 0/32 gives 34.7% at ≤ 2/8, but 25.5% at 0/8. There are only 2,932
  such candidates.
- The structural features hard12 enriched do not separate post-RL hardness in the fresh stream. The differences
  are within about 2 SE:
  - forward-first: 5.3% vs 6.0%;
  - inconsistent premises: 4.5% vs 6.1%;
  - depth 4–5 vs 2–3: about 6.4% vs 4.5%.

  Mild gradients: the T0 band score (4.6% to 7.8%), `~` and `&` goals, 12-line references, and `fwd_takehome_d2`
  (9.0%).

**Interpretation, not tested:** after RL, hardness in this stream is mostly idiosyncratic to individual theorems,
not carried by the features the generator can steer. A 6,000-target pool at 1–12/32 would need on the order of
10^5 screened fresh candidates. On the fresh stream only 2.7% are at 1–2/8, and the confirmation rate is unmeasured.
At the measured 45–51 attempts/s, that is more than the 5 GPU-hour budget, unless a learned hardness score raised
the yield.

**GPU time:** 777 s + 285 s = 1,062 s (0.30 GPU-hours), GPU 0 only. Before that, the 13–20 test-set agent's
calibration chain held the lock from 19:06 to 19:23 UTC.

## Design note for any resumption

A test split selected by one of the compared models biases that comparison. A post-RL filter makes the test
hard *for the filter model*, and re-measuring that model on it regresses toward the mean.

- Any future hard12r **test split must be an unfiltered random sample from the same candidate stream**.
- It should be binned by model-independent features: reference length, goal type, first step, premise consistency,
  formula depth and generator config.
- The filtered split may be reported only as a diagnostic.
- The RL pool may still be filter-selected.

## Files

- **Code:** this folder.
  - [`exclude_r.py`](exclude_r.py): the exclusion index and filter.
  - [`cands.py`](cands.py): the prior candidates and the eval-set rows.
  - [`run_eval.sh`](run_eval.sh): one eval under the GPU-0 lock protocol, or on GPU 1 once it is free of our
    processes. It runs under `timeout 3600` and appends to `work/logs/jobs.log`.
  - [`pilot.py`](pilot.py): the yield report.
  - [`regenerate.sh`](regenerate.sh): every step above.
  - The generator and selection are hard12's `gen_steer.py` and `select.py`, imported unchanged.
- **Meta:** [`meta/`](meta/) holds:
  - `pilot_report.json`, which also sits at `runs/pools/hard12r/meta.json`;
  - the prior-candidate report;
  - the fresh pick;
  - the new stream's generator summary.
- **Data:** `runs/pools/hard12r/work/`:
  - `cands/` (candidate records) and `sets/` (eval sets);
  - `eval/pilot_{prior,fresh}_k8/` (per-attempt rows);
  - `raw/` (excluded streams; unmeasured candidates remain there for a resumption);
  - `excl_index.pkl`, `logs/` and `pilot_report.{json,md}`.

## Caveats

- **Pilot draws:** k = 8 per candidate, one draw each. The ≤ 2/8 bin mixes truly hard targets (pass rate ≤ 0.38)
  with lucky failures of easier ones, and 0/8 includes targets the model may never solve.
- **The fresh sample is stratified:** 300 per config with band score ≥ 0.3, not the stream's natural config mix.
- **Server crash:** the server crashed at about 08:50 UTC, before any GPU work. The pilot ran after the 19:00 UTC
  restart. One of my monitoring loops matched its own command line and never exited. It was a wait loop only and
  was killed; no job was affected.
