# exp08 — fast iteration on an uncontended GPU, with tail-fill (prereg 2026-09-23 ~01:48 UTC)

**Question.** exp03 (k 16, 300 ft steps, retain 8k) was confounded: a reviewer re-score shared its GPU, so it ran
4 rounds instead of ~6 (238). Given equal samples per target, do twice as many cheaper rounds beat the seed's ~3?
This retests exp03 on the exp06 code (tail-fill + skip-saturated), with the GPU marked in gpu_in_use.txt.

**Change vs incumbent.** `variants/exp08.py` = `variants/exp06.py` with k 32 -> 16, ft_steps 600 -> 300,
retain 20000 -> 8000, rounds cap 12. Compare with exp06 (same code, seed knobs) and the control.

**Expected.** 6–7 rounds; dev = 285 (range 250–330); heldout_greedy >= 0.89.
**Falsified** if dev <= exp06 + 10 (or <= 255 if exp06 is lower).

**Seeds.** 0; seed 1 if it is the best so far by >= 20. **Stop rule.** One run per seed; reruns only on crash.
