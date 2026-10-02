# exp16 — schema-aware textbook exploration (prereg 2026-09-23 ~02:46 UTC)

**Question.** exp13 cracked only 2 of 19 textbook schemas (contraposition 108/109, export 37/94, all L7), and that
alone gave dev L7 ~+60. Its stream kept spending k = 32 on unsolved instances of already-cracked schemas, which expert
iteration spreads anyway. Do more cracked schemas follow if the stream samples instances of cracked schemas lightly
(k 8, unsolved only) and keeps k 32 for uncracked schemas, so later rounds are cheaper (more rounds) or more focused?

**Change vs incumbent.** `variants/exp16.py` = `variants/exp13.py` with per-schema allocation in the textbook stream:
instances of uncracked schemas at k 32 (T 1.0), unsolved instances of cracked schemas at k 8. Start round 2 as before.
Compared with exp13 s0 (324) / s1 (pending).

**Expected.** >= 4 schemas cracked (exp13: 2); dev = 340 (range 300–380). **Falsified** if <= 2 schemas cracked or
dev <= mean(exp13 s0, s1) + 5.

**Seeds.** 0; seed 1 if it is the best by >= 15. **Stop rule.** One run per seed; reruns only on crash.
