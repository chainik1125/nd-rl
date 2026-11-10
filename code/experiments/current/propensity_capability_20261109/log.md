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
