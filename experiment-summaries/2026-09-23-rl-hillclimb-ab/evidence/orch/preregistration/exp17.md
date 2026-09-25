# exp17 — textbook exploration from round 1 (prereg 2026-09-23 ~02:57 UTC)

**Question.** In exp13 the dev gain depends on when the first textbook schema cracks: s0 cracked in round 2
(1/16/84/145 -> 324), s1 in round 4 (0/0/1/13 -> 269). Both seeds cracked the same two schemas (export,
contraposition). Does sampling the textbook stream from round 1 (one more exploration round, ~42 s) make ignition
earlier and the score less seed-dependent?

**Change vs incumbent.** `variants/exp17.py` = `variants/exp13.py` with tb_start_round 2 -> 1. Compared with exp13 s0/s1.

**Expected.** textbook solved cum >= 60 at the end; dev = 305 (range 270–340). **Falsified** if dev <= 297
(exp13 two-seed mean) or textbook solved < 30.

**Seeds.** 1 first (the seed where exp13 ignited late, the harder case), then 0 if the result is >= 297.
**Stop rule.** One run per seed; reruns only on crash.
