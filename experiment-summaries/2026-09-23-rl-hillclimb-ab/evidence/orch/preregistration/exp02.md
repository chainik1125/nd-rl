# exp02 — hindsight relabelling of failed samples (prereg 2026-09-23 ~01:22 UTC)

**Question.** Early rounds are sparse (pretrain solves ~0.4 % of dev). Most samples fail on the prompted
conclusion, but some are valid proofs of a *different* conclusion from the same premises. Does training on those
long (>= 7-line) by-products bootstrap faster and raise the dev score?

**Change vs incumbent (seed rl.py).** `variants/exp02.py` = seed + `relabel()` (same logic as frozen/ref/expert_iter.py):
each non-accepted sample whose last line validly proves another conclusion from the prompt's premises is kept
(one per relabelled prompt) if verifier length >= 7. Each round's mix adds <= 5,000 random relabelled records x 2.
Everything else identical (rounds, targets, k, temperature, fine-tune, retain).
Allowed by SPEC rule 3 ("verified by-products of your own samples").

**Expected.** dev = 275 (range 245–320) vs control ~238. heldout_greedy within 0.03 of the control.
**Falsified** if dev <= 238 (no gain) — then relabelled data is not the bottleneck signal. Risk: extra verify cost
in judging eats a round.

**Seeds.** 0 now; seed 1 if dev >= 260 (to claim a difference). **Stop rule.** Single run per seed; no reruns unless crash.
