# exp23 — cheaper gen focus (k 16) in the exp21 recipe (prereg 2026-09-23 ~03:51 UTC)

**Question.** exp21's focus rounds cost ~120 s (k 32 on ~1,300 gen targets with < 2 proofs), so exp21 fitted only 4
rounds (323/318). With focus_k 16 a focus round costs about a normal round. Does that give back a 5th round (more
textbook spreading) and keep the L9 of 123–126?

**Change vs incumbent.** `variants/exp23.py` = `variants/exp21.py` with focus_k 32 -> 16.

**Expected.** 5 rounds; dev = 325 (range 295–350); L9 >= 120. **Falsified** if dev <= 320.5 (exp21 two-seed mean)
or only 4 rounds.
**Seeds.** 0 (last launch on GPU1 before the deadline). **Stop rule.** One run; reruns only on crash.
