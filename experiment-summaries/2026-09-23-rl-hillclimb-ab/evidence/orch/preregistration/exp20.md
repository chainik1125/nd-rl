# exp20 — textbook stream from round 1 (exp17) + gen frontier focus from round 4 (exp18) (prereg 2026-09-23 ~03:16 UTC)

**Question.** exp17 (textbook stream from round 1) ignited early and scored 331 (L7 115, L9 122). The gen frontier
focus gave L9 130–131 in exp14/exp18 (vs 110–129 without). Is the combination the best recipe for the final
checkpoint?

**Change vs incumbent.** `variants/exp20.py` = `variants/exp18.py` with tb_start_round 2 -> 1, i.e. exp17 + focus.

**Expected.** dev = 335 (range 300–365); L9 >= 125; textbook solved >= 80. **Falsified** if dev <= the mean of the
exp17 seeds (or <= 320 if only one exp17 seed is in).

**Seeds.** 0, then 1 if time allows before 04:07. **Stop rule.** One run per seed; reruns only on crash.
