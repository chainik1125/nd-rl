# exp01 — control: unmodified seed rl.py, seed 1 (prereg 2026-09-23 ~01:20 UTC)

**Question.** What do the seed pipeline's score and run-to-run noise look like on this machine? What are the per-round
timings (sample secs, fine-tune secs, rounds completed)? Those timings drive every budget-allocation change that follows.

**Change vs incumbent.** None (rl.py at commit 31197e7, identical to frozen/rl.py).

**Expected.** dev pass@32 = 238 ± 30 (seed 0 reference run: 238). heldout_greedy ≈ 0.8–0.9.
Falsified (i.e. the environment differs from the reference) if the score falls outside 190–290.

**Seeds.** 1 (seed 0 = the reference 238). **Stop rule.** Single run; no reruns unless it crashes.
