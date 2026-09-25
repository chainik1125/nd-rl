# w1 report (GEAR-Evolve arm A)

## Final frontier (dev /1610; L7/L8/L9; heldout)
- n025 BEST 401 (104/110/187, 0.885): T=1.0 rollouts + hardest-first refill (w2)
- n017 LEAN 393 (109/100/184, 0.910): 400-target windows, 15 rounds (w2)
- n023 385 (107/107/171, 0.895): training temperature 1.0 (w2)
- n027 377 (96/105/176, 0.915): max_per_thm 8 (w2)

**Best checkpoint:** `/work/w2/runs/n025_xo-refill-t10/final.pt`: dev **401**, heldout_greedy **0.885**.
Least-damage alternative: `/work/w2/runs/n017_small-rounds/final.pt` (393, 0.910). My best: `/work/w1/runs/n026_k24/final.pt` (375).

## What moved the score
- Using the whole budget with more, shorter rounds: the seed stopped after 2 rounds (174); rotating windows scored 289.
- Sampler throughput: w2's compacting sampler scored 358. Windows of 400 with 15 rounds reached 393; windows of 300 overshot (349).
- Temperature 1.0 and hardest-first refill: within noise (385, 401).

## What did not help (my runs)
- Heavier self-generated share in the mix: 154.
- Hindsight relabelling: about 15 by-products per run, so a no-op.
- Sample batch 1024: slower than 384.
- k16 over 1,200-target windows: more training solves, no dev gain.
- GRPO pass before the SFT step: -56.
- Persistent AdamW: 288, L7 collapsed.
- targets_extra in the pool: L9 fell.
- LR 4.5e-4: 333.
- k24: 375 on n023, 329 on n025.

**Caveat:** GPU0 (mine) runs rounds about 20–25% slower than GPU1, so my children of GPU1-tuned nodes got about 12 rounds instead of 15. Noise is about ±25; the children of 393 and 401 averaged −55 and −72.

## Controller steps (mine)
- v2: share an in-flight p1 when every alternative is more than 30 below it.
- v3: exclude elites more than 50 below the best from p1 (the seed kept winning on inflated productivity).
- v4: fresh elites that lost to their parent no longer force a crossover.
- v5: no crossover with p1's own ancestor or descendant.
- v8: mutate a leader that is more than 30 ahead instead of a due crossover.
(w2: v6, v7.)

## Next
Seed-replicate n025 and n017. Size rounds to the measured sampler speed. Remove the sampler's host syncs. Try an L9-stratified refill.
