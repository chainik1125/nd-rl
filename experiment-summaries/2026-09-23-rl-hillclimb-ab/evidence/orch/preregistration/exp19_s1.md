# exp19 seed 1 — replicate (prereg 2026-09-23 ~03:25 UTC)

Same file `variants/exp19.py`, seed 1. Purpose: two seeds behind the hot-exploration recipe, whose point is
*reliable* early ignition (exp17 ignited on s1 but not s0).

**Expected.** textbook ignition by round 3 (>= 20 solved); dev = 320 (range 290–350). **Falsified** if no ignition
by round 3 or dev <= 290. **Stop rule.** One run; reruns only on crash.
