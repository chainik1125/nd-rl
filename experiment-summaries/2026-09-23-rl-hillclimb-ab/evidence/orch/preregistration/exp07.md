# exp07 — length curriculum over the full target pool (prereg 2026-09-23 ~01:46 UTC)

**Question.** In exp01 rounds 1–2 solved 0 L9 targets (r1: 110 L7 / 1 L8; r2: 279 / 72 / 0), so the ~26 % of
round-1/2 samples spent on L9 (and most spent on L8 in round 1) found nothing. Dev is 63 % L9. Does restricting early
rounds to lengths the model can reach (round 1: L7, round 2: <= L8, round 3+: all) and spending those samples on
more distinct reachable targets from the full 3,804 pool move the frontier to L9 sooner?

**Change vs incumbent.** Built on `variants/exp06.py` (tail-fill + skip-saturated, pending review), so this is
compared to both exp06 and the seed control. `variants/exp07.py`: target pool = all 3,804; each round samples
<= 1,500 unsaturated targets drawn at random from those with L_true <= (7, 8, 9)[round]; k = 32, batch 384,
fine-tune and retain unchanged.

**Expected.** dev = 290 (range 250–340), with L9 >= 125 (control 109/102). heldout_greedy ~0.91.
**Falsified** if dev <= exp06's score + 10, or L9 <= 110.

**Seeds.** 0; seed 1 if it beats the best reviewed score by >= 20. **Stop rule.** One run per seed; reruns only on crash.
