# w2 report (GEAR-Evolve, arm A)

**Best:** `/work/w2/runs/n025_xo-refill-t10/final.pt` (n025): **dev 401/1610** (L7 104 / L8 110 / L9 187), heldout_greedy 0.885.
Runner-up with less in-distribution damage: `/work/w2/runs/n017_small-rounds/final.pt`, 393 (109/100/184), heldout 0.910. The 8-point gap is noise.
Seed n000: 174. Reference seed run: 238.

## Final frontier
| node | role | dev | L7/L8/L9 | heldout | change |
|---|---|---|---|---|---|
| n025 | BEST | 401 | 104/110/187 | 0.885 | T=1.0 rollouts on hardest-first refill |
| n017 | LEAN | 393 | 109/100/184 | 0.910 | 400-target windows, 135 ft steps, 15 rounds |
| n023 | DIVERSE | 385 | 107/107/171 | 0.895 | T=1.0 rollouts on n017 |
| n027 | DIVERSE | 377 | 96/105/176 | 0.915 | 8 proofs/theorem on refill line |

## What moved the score
- **More, shorter rounds using the full 600 s:** seed 2 rounds 174; rotating 600-target windows 7 rounds 289 (n001); final round shrunk to fit the time left 294.
- **Compacting sampler (n013, 358, +64):** drops finished rows from the batch and KV cache, batch 768. Sampling went from 50 s to 30 s per round, so 10 rounds fit. Bigger lib batches are slower.
- **400-target rounds on it (n017, 393, 15 rounds).** 300-target rounds overshot (349).
- T=1.0 rollouts plus hardest-first refill (n025, 401): gain is within noise.

## What did not move it
Length curriculum (L9 up, L7 down, net ~0), hindsight relabelling (few by-products), rarity weighting, lighter cap-6 retention (−60), GRPO pass (w1, −56), persistent AdamW (w1, −105), k16 breadth (w1). n017's children average ~360, so its 393 is partly luck.

## Controller steps (w2)
- **v6:** forced crossovers fall back to mutate when every other elite is p1's ancestor or descendant (triggered when BEST n013's frontier was all its own lineage).
- **v7:** drop p2 partners that scored >15 below their own parent (triggered when n017 was paired with the losing GRPO node).
- Adapted tests are at `/work/test_controller.py`. w1 added v3/v4/v5/v8.

## Next
Replicate n025 and n017 over seeds. Combine T=1.0 with 8 proofs/theorem. Cut per-round overhead (pre-encode cap-6, skip round checkpoints). Grow the pool with `targets_extra`.
