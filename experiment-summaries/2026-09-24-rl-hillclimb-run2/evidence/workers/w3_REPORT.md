# w3 report (GEAR run 2, ab2)

**Frontier (L12-14, pass@64 on the 1,000 subset).** My best: **n096**, 229/1000 (L12 181, L13 47, L14 1), dev 395/1200, score 1311, heldout 0.929. PENDING_BEST: one seed, on the fastest GPU (GPU4). Checkpoint: `/scratch/cowifornia/nd-rl/ab2/w3/runs/mt9k_n090_s0/final.pt`. Code: `nodes/n096/rl.py`. At the deadline the confirmed BEST is w2's n078 (two-seed 1175). My earlier BEST was **n046**: two-seed 950.5, `w3/runs/lvg_n024_s0/final.pt`.

**What moved the scores**
- **Throughput on the line-verified line.** n046 is a CUDA-graph line-verified sampler (`lvg_generate`, self-checked, with a fallback). Sampling took 21 s per round instead of 38 s, giving 30 rounds instead of 22. Score went from 591 to 1083 on one seed (950.5 over two). Every later elite carries it.
- **Weight averaging.** Returning the average of the last 3 round-end weights gave n076 +152 over its parent (968 to 1120) at equal rounds and solves, and heldout rose. Averaging 5 weights gave +47 (n090).
- **max_targets 9000** on n090 (w4's idea) gave +132 (n096), partly helped by the fast GPU.
- Earlier: plain graph sampler +7 (n020), length-grouped fine-tune +55 (n026) on the old base.

**What did not help:** shortest-first proof selection (n003), length-weighted subset in the crossover (n008), near-miss refill (n011, -29), deeper line retries (n061, rows live longer, same solves per round), length-grouped fine-tune on lv nodes (n033, n069: neutral). GEM on n032 gave +57 (n040), below the promotion bar.

**Controller:** I wrote v17, installed by the monitor at 09:45. `pick_p2` now skips a partner that already received p1's change through a mutation child; n046 x n038 would have reproduced n053. I found one more bug without fixing it: a forced crossover picked a fresh elite's own parent as p1 (n084 x n090). Two assigned crossovers duplicated existing files, so I recorded n061 and n096 as mutations with `--parents`.

**Cited ideas tested:** arXiv 2502.20122 (best-of-n by length), 2502.12272 (LILO), 2408.16673 (GEM), 2502.03438 (BFS-Prover), 2504.10478 (weight ensembling).

**Rule caveat:** n020, n026 and n033 were three own-idea nodes in a row, which broke the one-in-three citation rule once.

**Next:**
1. Two-seed n096 and n092.
2. For the 2 h champion: set avg_last 3-5 and scale max_targets with the budget. The champion saturated on 6,000 targets.
3. Combine lvg with length-grouped fine-tune on a fast GPU. Fine-tune is now ~40% of each round.
4. Fix the fresh-elite parent pairing in the controller.
5. Normalise for GPU speed (1.65x spread) before comparing single seeds.
