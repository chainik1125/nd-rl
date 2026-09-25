# exp10 — fast iteration (exp08) + fast sampler (exp09) (prereg 2026-09-23 ~02:08 UTC)

**Question.** exp08 (k 16, 300 ft steps, retain 8k, tail-fill, skip-saturated) ran 5 rounds and scored 252, with L9 126,
the best L9 so far. exp09's sampler cut round-1 sampling from 109 s to 95 s. Stacked, do they give a 6th round and a
higher L9?

**Change vs incumbent.** `variants/exp10.py` = `variants/exp08.py` with sample_round calling `fast_generate`
(code identical to exp09's). Compared with exp08 (252) and the control (249).

**Expected.** 6 rounds; dev = 265 (range 240–300); L9 >= 125; heldout_greedy ~0.90.
**Falsified** if dev <= 250.

**Seeds.** 0; seed 1 if dev >= 262. **Stop rule.** One run per seed; reruns only on crash.
