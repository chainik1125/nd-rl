# exp13 seed 1 — replicate of the textbook-exploration stream (prereg 2026-09-23 ~02:39 UTC)

exp13 s0 scored 324 (109/86/129), +72 over exp10 (252) and +75 over the incumbent (249), with 145 textbook targets
solved (1/16/84/145 by round). Policy: two seeds before calling a difference. Same file (`variants/exp13.py`),
seed 1, no change.

**Expected.** dev = 310 (range 280–340); textbook targets solved cum >= 80. **Falsified** (the s0 gain was
luck) if dev <= 270 or textbook solved < 20.
**Stop rule.** One run; reruns only on crash.
