---
written_on: 2026-09-24
---
# w6 report (GEAR run 2, ab2)

(Saved by the monitor from w6's session hand-back; the session's own file write was refused.)

**Frontier at close.** Score = dev + 4 x frontier (frontier on the 1,000-theorem subset). BEST n078 (w2), 1175 on
two seeds (dev 383, frontier 198). LEAN n092, 1212.5 on two seeds (dev 372.5, frontier 210), held back by the
40-point margin. Pending at close: n096, 1311 on one seed.

**My best checkpoint.** n086, `/scratch/cowifornia/nd-rl/ab2/w6/runs/x_avg3_s0/final.pt`, one seed: dev 381
(L8 112, L9 106, L10 86, L11 77); frontier 198 (L12 161, L13 36, L14 1, the run's first length-14 solve); held-out
0.934; score 1173. n024 and n038 were each BEST for a time (n024 seeds 550 and 632, mean 591; n038 seeds 897 and
641, mean 769).

## What moved the scores
- Line-verified sampling, the biggest lever. Each generated line is checked; a rejected line is removed from the
  cache and resampled from the start of that line. My implementation agrees exactly with the verifier on 120k test
  proofs. On the GEM-loss node n017 it took dev from 188 to 314 and frontier from 13 to 59 (n024). Adding GEM to w1's
  line-verified node n010 took the score from 300.5 to 460 (n031), so the two combine.
- Long-heavy target subset (length weights L6 0.5 up to L11 3). With line-verified sampling it doubled the frontier,
  69.5 to 150 (n038, +306). Under plain sampling the same weights had hurt. Nothing on the schema-coverage line (n080,
  -183); a heavier L11 weight of 4 gave only +7.5 (n052).
- Averaging the last 3 round-end weights (from n076): +75 on n060 (n086), but -433 on n078 (n093); unresolved.
- Prefix branching (restarting unsolved targets from the longest valid prefix of failed samples) was neutral alone
  (n005, two-seed 187.5); w1 combined it with line-verified sampling in n016, which scored well above its siblings.

## What did not help
Shape-weighted cap-6 retention (n009, -26); shortest-first proof selection on top of branching (n014, -32.5);
replacing retries with fresh targets (n019, -42.5); fine-tune batch 256 (n045, -227: doubled fine-tune time on the
capped GPUs); longest-first proof selection (n059, -303); retaining 12,000 instead of 20,000 cap-6 records (n073,
within the tie margin). The budget-aware top-up (n066) never fired at 1,200 s; `nodes/n066/rl.py` is ready for a
champion-length run. An offline CPU test of backtracking in the guided sampler found no extra solves at 2.7x the
cost, so it was never run.

## Controller
I wrote v11 (the monitor's metric change, dev + 4 x frontier, with state migration and a margin floor of 10, plus
serving the highest-score pending node first); it passed the dry-run and 25/25 tests and the monitor installed it
at 05:31. Other workers moved it to v14 and beyond.

## Sources tested
TreeRL arXiv 2506.11902; BFS-Prover 2502.03438; GEM 2408.16673; online difficulty filtering 2504.03380; "Hard
Examples Are All You Need" 2508.14094; weight ensembling 2504.10478; 2502.20122 (shortest-first selection).

## Caveats and next
GPU throttling changed sampling time from 21 to 53 s per round, confounding most single-seed differences.
Replicate n086 and n093 to settle weight averaging. The champion did best with the full pool (max_targets 15,738),
so that or the n066 top-up belongs in the champion recipe. Length 14 stays close to zero because training proofs
stop at 11 lines.
