# exp11 — RL-heavy fine-tuning mix: retain 8,000 -> 2,000 (prereg 2026-09-23 ~02:10 UTC)

**Question.** In the fine-tune mix the verified RL records are a small share early on (exp01 round 1: 496 of 20.5k,
2.4 %; exp08 round 1: 472 of 8.5k, 5.6 %), so the first rounds barely move the model toward the new targets. The
runs with smaller retain (exp03, exp08) had the highest L9. Does a much smaller cap-6 retained slice (2,000 per
round, RL share ~20 % in round 1 and > 50 % later) speed acquisition enough to raise the dev score, at an acceptable
cost to heldout_greedy?

**Change vs incumbent.** `variants/exp11.py` = `variants/exp10.py` with retain 8000 -> 2000. Compared with exp10.

**Expected.** dev = 270 (range 235–310), L9 >= 130; heldout_greedy drops to ~0.85–0.89.
**Falsified** if dev <= exp10's score + 10 (or heldout_greedy < 0.80, which would make it unusable even if dev rises).

**Seeds.** 0; seed 1 if it is the best by >= 20. **Stop rule.** One run per seed; reruns only on crash.
