---
written_on: 2026-11-09
written_by: agent:claude (orchestrator)
---

# Propensity vs capability: run log

Times are UTC unless marked PT.

- **2026-11-10 05:25Z (9:25 pm PT, Nov 9) — proposal to Leon.** The read at k 32, k_f 12 on groups
  `main,matched` (24 checkpoints, 346 theorems), estimated 2.3 GPU-h (2.9 at serrano's 200 W cap),
  3 h guard. Pre-registration committed first (5a97401, pushed to `leon/propensity-capability`).
- **05:54Z (9:54 pm PT) — launched under Leon's 30-minute rule** (no reply to the costed proposal).
  serrano GPU 0, which had no process from any user (8 MiB in use, 200 W cap); the other four serrano
  GPUs hold other users' jobs. Command: `./run_read.sh 0 main,matched --kf 12`. Read PID 2870200,
  watchdog PID 2870201, log `/scratch/cowifornia/nd-rl/propensity_capability/logs/read_20261110T055400Z.log`.
  Lab state at launch: 6 of 12 lab GPUs free before, 5 after; this is our only lab GPU job.
- **06:55Z (10:55 pm PT) — one-hour check-in.** Alive, sole user of GPU 0, watchdog STATUS normal, no
  errors. 9 of 24 checkpoints finished (the full base, the six 1 % checkpoints, `ko_s0`, `dko_s0`),
  `ko_s1` done and `dko_s1` running; 3,574 GPU-seconds used. Pace: bases about 4 min, 1 % ends about
  11 min, the first 0 % end 11 min; slower than the estimate on bases, close to it on ends. Projected:
  main group done near 2.2 GPU-h, matched group near 2.9, inside the 3 h guard.
- **07:55Z (11:55 pm PT) — two-hour check-in.** Alive, sole user of GPU 0, no errors. 16 of 24
  checkpoints finished (all of the full, 1 % and 0 % lines, `dose01_s0`, `dd01_s0`, `dose01_s1`);
  6,550 GPU-seconds used. The 0 % ends took 11 to 15 min each. Projected: main group done near
  8,500 s (about 12:16 am PT); the 10,800 s budget will stop the matched group after three or four
  of its five checkpoints.
- **08:54Z (12:54 am PT, Nov 10) — read ended at the 3 h timeout** (GPU phase exit 124), during the
  last matched checkpoint (`m_dd01g0_s2_it90`, the pilot line), which was not read. 23 of 24
  checkpoints are complete: the 19 of the main group and 4 of the 5 matched. GPU time in the ledger:
  10,637 s (2.95 GPU-h), serrano GPU 0, 9:54 pm to 12:54 am PT; no other user appeared on the GPU.
  Rows phase exit 0 at 08:59Z. The orchestrator then stopped the watchdog (PID 2870201) and removed
  the stale lock `locks/serrano_gpu0.lock` left by the timeout.
- **09:00Z — pre-registered analysis run once** (`analyze.py`, then `fig.py`), PREREG sha256
  `6fd57cad…2e1261`. Verdict of the frozen rule: informative (30 of 54 primary cells read; E agrees
  with U's sampled first moves in 138 of 138 cells). Outputs in `results/`.
- **Observed, to be written up:** the first-move propensity cannot read the reductio at the 1 % ends:
  0 to 0.7 % of attempts open with it, while every accepted proof on the strict 106 uses a reductio
  somewhere (opened later, usually after a conditional opening). The "uses anywhere" secondary is
  the measure that sees it. This is a limit of the primary measure chosen in SPEC.md, not a result.
- **2026-11-11 03:27Z (7:27 pm PT, Nov 10) — post-hoc amendment 1 (after results), at Leon's question "router
  or default shift: does propensity rise only where the direction is needed, or everywhere?"** By agent:claude (a
  subagent of the orchestrator), CPU only, from the result rows and `results/results.json`; no new sampling. The
  amendment was appended to PREREG.md before the new script was run on the real rows: the frozen router rule with
  propensity = "uses d anywhere" (Σ used / Σ k per class), thresholds unchanged, log10 odds of (x + 0.5) / (n + 1)
  in place of the clamped exact P̄. **Every reading from it is exploratory.** Script `posthoc_anywhere.py`
  (imports `analyze.py`, does not change it), tests `test_posthoc_anywhere.py` (12 synthetic cases, pass; the 26 of
  `test_analyze.py` still pass), outputs `results/posthoc_anywhere.{md,json,pdf,svg,png}`. Checks in the output:
  inputs identical to those of `results.json`; the per-class levels equal the frozen router table's "uses anywhere"
  column exactly; the frozen secondary's per-seed "uses anywhere" readings recompute identically.
- **03:30Z — correction to the amendment's item 1, before any reading was written.** The first run warned on rows
  where U/reductio `solved` cannot mean "accepted attempts that use d" (SPEC.md's wording): `pc_run.py` line 401
  counts accepted attempts whose proof goes *through* d (its DN line on the conclusion's dependency path), and an
  accepted proof can open a reductio as a detour. The split was redefined to what the rows hold (amendment, item 1).
  That first run had already printed the carried-over dose verdicts; the correction does not touch that measure.
  **For the orchestrator:** SPEC.md's description of U/<d> `solved` does not match the runner; `analyze.py`'s
  "finishing once chosen" (U solved / used) inherits the narrower meaning.
- **Observed (from `results/posthoc_anywhere.md`; doses at unequal RL compute, not compared).** Reductio, frozen
  classes (required 136 = strict 106 + 30 classical-only theorems of the other set; idle 194), all attempts:
  **1 %: router, 3 of 3 seeds** (required 1.5 → 30.4, 9.8 → 26.9, 3.5 → 43.0 %; idle 3.4 → 3.7, 4.3 → 2.8,
  3.6 → 8.0 %; interval of Δ(idle) − ½ Δ(required) [−1.10, −0.37], [−0.91, −0.18], [−0.59, −0.02]); seed 2's idle
  class is itself up (Δlog10 odds +0.37 [+0.12, +0.60]), so it is a router with some spill-over there (f = +0.28).
  **0 %: seeds disagree** (unresolved / router / no rise to locate); **0.1 %: seeds disagree** (unresolved / no rise
  to locate / unresolved): the required class moves +1.8 and +3.1 points on average there, from bases of 2.0 to 4.1 %.
  Sensitivities: strict 106 alone against idle gives 1 % router (3 of 3), 0 % router (2 of 3), 0.1 % seeds
  disagree; dropping the attempts whose accepted proof goes through the reductio (on the strict 106 exactly "uses
  among attempts not accepted") leaves every dose verdict as it was.
- **Found while doing it (caveats for any reading).** (1) The 30 classical-only theorems of the other set are in
  `required` but their accepted proofs almost never go through a reductio (0 to 4.8 % of accepted attempts at the RL
  ends; guess, not checked: DN applied to a premise): "required" holds for the strict 106 only. (2) Idle is all from
  the other set and required mostly from the strict 106, so the contrast is also one of sets (the frozen PREREG's
  "Confound"). (3) At the RL ends the idle class's uses sit in few theorems, several with every attempt using a
  reductio (e.g. 0 % seed 2: 2 theorems, both 32 of 32). (4) What the rows cannot separate: on the required side,
  choosing the reductio more against finishing it more (no forced arm for a reductio opened later); for idle, detours
  inside accepted proofs against uses in failed attempts.
