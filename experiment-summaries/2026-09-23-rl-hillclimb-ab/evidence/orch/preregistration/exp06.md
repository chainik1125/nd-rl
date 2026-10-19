# exp06 — use the tail of the budget + skip saturated targets (prereg 2026-09-23 ~01:44 UTC)

**Question.** The seed's stop rule skips a round that is not expected to fit, which left 40 s (exp01) to 159 s (exp04)
unused, and the round count is what drives the score (exp04: 2 rounds -> 193; exp01: 3 -> 249). Does running a
shrunk final round (k and fine-tune steps scaled to the time left) plus not resampling targets that already have
>= 4 distinct proofs (cheaper later rounds) add a round's worth of progress?

**Change vs incumbent (seed rl.py).** `variants/exp06.py`: (a) if a full round will not fit and >= 25 % of one
does, run it with k and ft_steps scaled by 0.9 x the fraction that fits; (b) skip_saturated (targets with >= 4
distinct proofs are not resampled); rounds cap 4 -> 8. Targets, k=32, batch 384, ft, retain unchanged.

**Expected.** 4 rounds (the 4th shrunk or full); dev = 270 (range 240–310) vs control 249 (s1) / 238 (ref s0).
**Falsified** if dev <= 245.

**Seeds.** 0. **Stop rule.** One run; reruns only on crash. GPU contention from reviewer re-scores is recorded, not rerun.
