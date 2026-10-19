# exp14 — frontier focus in late rounds (prereg 2026-09-23 ~02:30 UTC)

**Question.** By round 4 the training-target frontier saturates (exp10/exp11: +108/+72, then +52/+43, +35/+25 new
proofs per round) while the later rounds keep spending most samples on targets already solved 1–3 times. From round 4
on, does sampling only targets with < 2 distinct proofs (unsolved or solved once) at k 48 and T 1.0 (more exploration)
push the frontier (more L9 solved) and the dev score?

**Change vs incumbent.** `variants/exp14.py` = `variants/exp10.py` + from round 4: active = unsaturated targets with
< 2 distinct proofs, k 16 -> 48, T 0.8 -> 1.0 for sampling (the dev scorer's T 0.8 is unaffected). The tail-fill
round estimate is also updated after focused rounds (bookkeeping fix). Compared with exp10 (252).

**Expected.** training targets solved cum at end >= 680 (exp10: 614); L9 targets solved >= 60 (exp10: 40);
dev = 262 (range 240–295). **Falsified** if cum solved <= 640 or dev <= 252.

**Seeds.** 0; seed 1 if it becomes the best by >= 15. **Stop rule.** One run per seed; reruns only on crash.
