# exp21 — exp19 (hot, round-1 textbook exploration) + gen frontier focus from round 4 (prereg 2026-09-23 ~03:21 UTC)

**Question.** The recipe's score depends on textbook ignition, which is random per run: exp17 (stream from round 1,
T 1.0) gave 331 on s1 but 236 on s0 (no ignition). exp19 (T 1.2 on uncracked schemas, schema-aware allocation,
round-1 start) cracked both schemas by round 2 (27 solved, still running). The gen focus gave the top L9 (130–131).
Is exp19 + focus the final recipe?

**Change vs incumbent.** `variants/exp21.py` = `variants/exp19.py` + exp18's gen-stream focus from round 4 (< 2
proofs, k 32, T 1.0) and its tail-fill bookkeeping fix. exp20 (exp17 + focus) was pre-registered but not launched:
exp17 s0's non-ignition made its base the wrong one to build on.

**Expected.** textbook ignition by round 3 (>= 20 solved) and >= 100 at the end; dev = 335 (range 300–365); L9 >= 125.
**Falsified** if dev <= 324 (incumbent) or no ignition by round 3.

**Seeds.** 0 now; seed 1 next if time allows. **Stop rule.** One run per seed; reruns only on crash.
