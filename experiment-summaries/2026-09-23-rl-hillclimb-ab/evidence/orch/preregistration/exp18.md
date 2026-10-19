# exp18 — textbook stream (exp13) + gen-stream frontier focus from round 4 (prereg 2026-09-23 ~02:58 UTC)

**Question.** The textbook stream lifts dev L7 (67–109 vs ~40); exp14's frontier focus (late rounds sample only gen
targets with < 2 proofs, T 1.0) gave the best L9 (131 vs 110–125). L9 is 63 % of dev. Do they add up?

**Change vs incumbent.** `variants/exp18.py` = `variants/exp13.py` + from round 4 the gen stream samples only targets
with < 2 distinct proofs, at k 32 and T 1.0 (exp14 used k 48; lower here to leave time for the textbook stream), and
the tail-fill round estimate is updated after focused rounds (exp14's bookkeeping fix). Textbook stream unchanged.

**Expected.** dev = 315 (range 280–350), L9 >= 125. **Falsified** if dev <= 297 (exp13 two-seed mean) or L9 <= 117
(exp13's seed-matched L9 is 129 on s0).

**Seeds.** 0 (exp13 s0 = 324 is the seed-matched comparison). **Stop rule.** One run; reruns only on crash.
