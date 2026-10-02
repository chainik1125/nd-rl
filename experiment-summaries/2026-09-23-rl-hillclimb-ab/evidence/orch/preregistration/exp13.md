# exp13 — textbook-schema exploration stream (prereg 2026-09-23 ~02:21 UTC)

**Question / belief under test.** Every run so far lands at 217–252 whatever the allocation (exp08: 5 rounds, 252;
exp10: 6 rounds, 252), with dev L8 pinned at 87–93. In training, textbook-source targets (19 classic schemas:
disjunctive syllogism, contraposition, export/import, excluded middle, Peirce, De Morgan, distribution) are solved
0 / 179 times in exp08, while generated targets are solved at 12–64 %. Belief: a large part of dev is textbook-like
schemas the model never reaches, and that is the plateau. This run tests whether dedicated exploration can find
*any* textbook proofs in 600 s (and whether that moves dev).

**Change vs incumbent.** `variants/exp13.py` = `variants/exp10.py` plus: main targets = the seed's random 1,500 minus
textbook ones (~1,320 generated); a second stream samples all 830 textbook targets (419 pool + 411 targets_extra)
at k 32, T 1.0 from round 2 on (skip-saturated applies); verified textbook proofs join the mix like any target proof.

**Expected.** Textbook targets solved cum: 5 (range 0–40). dev = 250 (range 225–285): I expect a real but small
effect at best. **Falsified (belief supported that textbook is out of reach in 600 s)** if 0 textbook targets are
solved. dev <= 240 with textbook solved > 0 would say the stream's cost outweighs its gain.

**Seeds.** 0. **Stop rule.** One run; reruns only on crash.
