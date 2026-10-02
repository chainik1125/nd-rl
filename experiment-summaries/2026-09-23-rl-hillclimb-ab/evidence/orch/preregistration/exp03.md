# exp03 — faster iteration at equal sampling budget (prereg 2026-09-23 ~01:25 UTC)

**Question.** exp01 timings: round 1 = 109 s sampling (48k samples, 441/s) + 57 s fine-tune (600 steps) -> the seed
fits only ~3 rounds in 600 s, and round 1's mix is 2.4 % RL records (496 in 20.5k). Does halving the per-round cost
(k 32 -> 16, 600 -> 300 fine-tune steps) to get ~6 rounds, with a smaller retained slice (20k -> 8k, so RL records
are a larger share of each batch), move the frontier (L7 -> L8 -> L9) further in the same wall clock?

**Change vs incumbent (seed rl.py).** `variants/exp03.py`: rounds 4 -> 12 (clock-limited, expect ~6), k 32 -> 16,
ft_steps 600 -> 300, retain 20000 -> 8000. Same 1,500 targets; total samples per target about equal (~96).

**Expected.** dev = 300 (range 255–360) vs control ~238; L9 gains most. heldout_greedy may drop by up to 0.05.
**Falsified** if dev <= 250.

**Seeds.** 0 first; seed 1 if dev >= 270. **Stop rule.** One run per seed; reruns only on crash.
