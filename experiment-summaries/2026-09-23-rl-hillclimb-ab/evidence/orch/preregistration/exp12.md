# exp12 — equal weight per solved target in the fine-tune mix (prereg 2026-09-23 ~02:22 UTC)

**Question.** The seed repeats every selected proof rl_weight = 4 times, so an easy target with 4 distinct proofs gets
16 records and a frontier target solved once gets 4: the RL part of the mix is dominated by easy targets. Does giving
each solved target ~12 records in total (1 proof x 12, 2 x 6, 3 x 4, 4 x 3), which up-weights the rarely solved
frontier targets 3x relative to the seed, speed the frontier toward L8/L9?

**Change vs incumbent.** `variants/exp12.py` = `variants/exp10.py` with per-target weighting (per_target_weight 12)
in build_mix. Compared with exp10.

**Expected.** dev = exp10 + 15 (point estimate 270; range 240–305); L9 up most.
**Falsified** if dev <= exp10's score + 5.

**Seeds.** 0; seed 1 if it becomes the best. **Stop rule.** One run per seed; reruns only on crash.
