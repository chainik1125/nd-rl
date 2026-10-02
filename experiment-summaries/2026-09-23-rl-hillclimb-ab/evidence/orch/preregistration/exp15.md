# exp15 — textbook exploration (exp13) + equal weight per solved target (exp12) (prereg 2026-09-23 ~02:41 UTC)

**Question.** exp13's textbook stream solved 145 textbook targets (dev 324). Newly found schemas start as targets with
one proof, which the seed's weighting gives 4 records against 16 for easy targets. exp12's per-target weighting
(~12 records per solved target) gave +16 on the exp10 code. Stacked, does it spread new textbook schemas faster?

**Change vs incumbent.** `variants/exp15.py` = `variants/exp13.py` + per_target_weight 12 in build_mix (exp12's code).
Compared with exp13 s0/s1.

**Expected.** textbook solved cum >= 180 (exp13 s0: 145); dev = 340 (range 305–370).
**Falsified** if dev <= mean(exp13 s0, s1) + 5.

**Seeds.** 0; seed 1 if it is the best by >= 15. **Stop rule.** One run per seed; reruns only on crash.
