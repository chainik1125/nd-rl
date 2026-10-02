# w5 report (GEAR run 2, ab2)

**Frontier at the end (14:20 UTC).** BEST is n078 (w2), two-seed score 1175 (dev 383, frontier 198). n096 (1311) and n092 (1286) wait for a replicate. Score = dev + 4 x frontier.

**My best checkpoint.** n060 = `w5/runs/x_n050_graph_s0/final.pt`. Seed 0: dev 360, frontier 176. The auditor's seed 1: 384 and 187. Two-seed mean: 1098 (dev 372, frontier 181.5), held-out 0.923. It was BEST from about 10:30 to 12:45. Best single seed: n084 at 1132 (364, 192), `w5/runs/idxcap_n076_s0/final.pt`.

**What moved the scores** (single seeds unless marked):
- Line-verified sampling, added to the heated-line-head line (n032): dev 271 vs 191, frontier 54 vs 15.
- The CUDA-graph lv sampler on n050 (n060): +96 score from 26 rounds instead of about 20. Confirmed with two seeds.
- POLARIS-style staged temperature (1.0, then 1.1 at 35%, then 1.2 at 70% of the budget): +52 on n041 (n054) and +109 on n053 (n067), mostly on the frontier.

**What did not help:**
- Schema sibling spreading (n004): -49.
- GEM without line verification: +9 (n017) and -27.5 (n025).
- Length-weighted quotas on n039 (n047, -86) and on n060 (n074, -217).
- Index cap on n076 (n084): +12.
- Heated heads on top of staged temperature (n091): -320, as the accept rate collapsed.
- max_targets 9000 on n090 (n098): -102.
- Prefix hindsight relabelling gave about 1-8 by-products per round. I aborted it twice and recorded no node.
- Line backtracking was no better than more samples in a CPU test, so I never ran it on GPU.

**Controller steps.** Two candidates passed the dry-run and the tests, but my session was not allowed to install either.
- v11 (auditor replicates the highest-metric pending node first): superseded by w6's version, installed at 05:45.
- v18 (a child that lost to its parent1 by more than the loss band cannot enter as DIVERSE): staged as `controller.new.py` and `test_controller.new.py`, waiting for the monitor.

**Sources:**
- GEM: arXiv 2408.16673
- BFS-Prover line verification: arXiv 2502.03438
- POLARIS: hkunlp.github.io/blog/2025/Polaris
- Token-level temperature: arXiv 2510.08892
- APOLLO and HER (for the aborted prefix hindsight): arXiv 2505.05758, arXiv 2112.10664

**Next:**
- A second seed for the 9000-target change, which helps on some bases and hurts on others.
- Staged temperature in the 7,200 s champion.
- Cheaper line verification, since the GPUs are capped at 150 W and throughput is still the lever.
- Install v18.
