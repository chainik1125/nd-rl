# exp23 seed 1 — replicate (prereg 2026-09-23 ~03:57 UTC)

Same file `variants/exp23.py`, seed 1: the last launch before the 04:20 deadline (GPU0, after exp21_s2), so exp23
has two seeds whatever its s0 shows.

**Expected.** 5 rounds; dev = 320 (range 290–350). **Falsified** if dev <= 300 or only 4 rounds.
**Stop rule.** One run; reruns only on crash.
