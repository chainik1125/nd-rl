# exp19 — crack more textbook schemas: round-1 start + T 1.2 on uncracked schemas (prereg 2026-09-23 ~03:10 UTC)

**Question.** Every run so far cracked the same 2 of 19 textbook schemas (contraposition, export); ~100k samples per
run on the other 17 at T 1.0 found nothing. The model fails those with a confident wrong strategy (e.g. NEGE on the
wrong pair). Does a hotter exploration temperature (1.2) on uncracked-schema instances, with the stream starting
in round 1 (exp17: earlier ignition, 331 on s1), crack a third schema?

**Change vs incumbent.** `variants/exp19.py` = `variants/exp16.py` (schema-aware allocation: uncracked k 32,
cracked-unsolved k 8 at T 1.0) + tb_start_round 1 + T 1.2 for uncracked-schema instances. Compare with exp17 (same
start round, T 1.0, no allocation split) and exp16.

**Expected.** >= 3 schemas cracked; dev = 330 (range 295–365). **Falsified** if only the usual 2 schemas are cracked
(then T 1.2 is not the missing ingredient), or dev <= 315.

**Seeds.** 0. **Stop rule.** One run; reruns only on crash.
