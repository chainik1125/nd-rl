# w1 report (run ab2, GEAR-Evolve, 02:26-14:30 UTC)

**Frontier (controller, 14:20):** BEST n078 (w2; index-shift cap on the POLARIS-ramp cuda-graph line-verified node), two seeds: score 1175, dev 378/388 (mean 383 of 1200), frontier 205/191 (mean 198 of 1000), heldout 0.924/0.921. Checkpoint: `/scratch/cowifornia/nd-rl/ab2/w2/runs/e12_n071_idx/final.pt` (seed 0; path from `nodes/n078/meta.json`); the replicate is the auditor's `rep_n078_s1`. Pending at 14:20: n096 (1311), n092 (1286). My best node: n089 (score 1047, dev 347, frontier 175), `/scratch/cowifornia/nd-rl/ab2/w1/runs/e14_divpick/final.pt`.

**What moved the score**
- Line-verified sampling (my n010, BFS-Prover-style step checking): an incremental Fitch checker rejects a bad line at its `;`, masks it out of the KV cache and resamples it from the stored line-start logits. n010 two-seed 206.5 vs seed 185; with prefix branching (n016) 243 dev / 32.5 frontier, confirmed BEST. Every later elite builds on it (w6's reimplementation n024, w3's cuda-graph version n046/n060).
- Schema-coverage subset (n041): +136 score on n024 (frontier 69.5 -> 105), transplanted into n050/n060.
- From others: cuda-graph lv sampler (more rounds), index-shift cap, length-weighted quotas on the lv base, weight averaging, larger subsets.

**What did not help (my tests):** length-weighted quotas on the plain sampler (n002, -8); one-line backtracking (n027, 4 rounds slower); a lower-sync lv sampler (n021: only 8% faster; its +42 was seed noise); retry heating (n048 +62 within noise; n055 -323); staged temperature on n053 (n063 -81); reductio-first seeding (n070: 8 reductio proofs found, all L6); max_targets 9000 on n060 (n077 -280, slow GPU); strategy-diverse proof selection (n089 +35.5) and weight averaging on it (n097 +11), both inside the margin. n081 crashed: my patch bug.

**Controller steps:** none. Other workers made v10-v14 while I was running; each of my decision lines after five experiments justified the search as healthy.

**Ideas with sources:** arXiv 2502.03438 (n010, n016, n027), 2510.08892 (n048, n055), POLARIS blog (n063), 2507.13266 (n070), 2509.02534 (n089), 2504.10478 (n097); the rest were my own ideas.

**Caveats:** single-seed noise on the lv line is roughly +-150 score (n038 seeds 897/641). GPU speed differs by up to 30% (the power-capped GPUs 0 and 3), which confounds single runs. L14 is still at 0-1 of 100 everywhere.

**Next:** ignite the classical-detour shapes (excluded_middle L9, demorgan_nand_to_or L14). Seeding the first line is not enough, so seed the NEGI/DN ending, or seed from a stronger (champion) checkpoint. Replicate n089. Run a champion on n096/n092 if they confirm.
