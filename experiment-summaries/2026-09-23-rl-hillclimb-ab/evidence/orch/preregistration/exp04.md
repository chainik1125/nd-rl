# exp04 — breadth: all 3,804 targets, k 12, skip saturated targets (prereg 2026-09-23 ~01:31 UTC)

**Question.** The seed samples a fixed random 1,500 of the 3,804 targets (~400 of the 1,004 L9 targets) at k = 32.
exp01 round 1 solved 111 of 1,500, almost all L7. Given equal samples per round, is breadth (more distinct theorems,
2.5x more L8/L9 targets, fewer duplicate proofs of easy targets) worth more than depth per target?

**Change vs incumbent (seed rl.py).** `variants/exp04.py`: n_targets 1500 -> all 3,804; k 32 -> 12 (45.6k samples
per round vs 48k); targets that already have >= max_per_thm (4) distinct proofs are not resampled.
Everything else identical (4 rounds clock-limited -> ~3, 600 ft steps, retain 20k).

**Expected.** dev = 275 (range 240–320) vs control 249 (exp01, s1) / 238 (reference, s0). More RL records per round
(solved targets ~2x). heldout_greedy ~0.91.
**Falsified** if dev <= 250.

**Seeds.** 0 first; seed 1 if dev >= 270. **Stop rule.** One run per seed; reruns only on crash.
