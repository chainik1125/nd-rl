### OOD sets without textbook_dev (840), k 16; theorems resampled independently

```
python -m code.thinking.stats table --group T0=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s1/rows_plain_B0.jsonl --group 'T1 B0=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B0.jsonl' --group 'T1 B64=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B64.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B64.jsonl' --group 'T1 B256=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B256.jsonl' --group 'T0 guided=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s0/rows_logical_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s1/rows_logical_B0.jsonl' --group 'T0e (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0e_s0/rows_plain_B0.jsonl' --group 'T1v1 B256 (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1v1_s0/rows_plain_B256.jsonl' --group 'T1 guided B256 (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_logical_T1_s0/rows_logical_B256.jsonl' --pair 'T1 B256 vs T1 B0' --pair 'T1 B256 vs T0 guided' --k 1,5,n --tokens 500,1000,2000,4000 --exclude textbook_dev --title 'OOD sets without textbook_dev (840), k 16; theorems resampled independently' --out code/experiments/current/thinking_traces_20261005/stage1_stats/ood_theorems.md
```

n = 840 theorems (charles_long_reach, ood_and_reorder, ood_case_product, ood_commute, ood_or_reorder); 4000 draws, RNG seed 0; resampled: both. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | pass@16 | tokens / attempt |
|---|---|---|---|---|---|
| T0 | 2 | 1.8% [1.2, 2.6] | 3.7% [2.6, 4.8] | 5.1% [3.7, 6.5] | 140 |
| T1 B0 | 2 | 1.1% [0.7, 1.6] | 2.7% [1.8, 3.7] | 4.3% [3.0, 5.7] | 159 |
| T1 B64 | 2 | 0.8% [0.5, 1.2] | 2.1% [1.2, 3.1] | 3.5% [1.8, 5.4] | 211 |
| T1 B256 | 2 | 1.3% [0.8, 1.9] | 3.2% [2.1, 4.4] | 5.1% [3.2, 7.1] | 370 |
| T0 guided | 2 | 4.1% [3.1, 5.2] | 8.2% [6.6, 9.8] | 11.7% [9.6, 13.7] | 426 |
| T0e (1 seed) | 1 | 1.4% [0.8, 2.0] | 2.9% [2.0, 4.0] | 4.2% [2.9, 5.6] | 139 |
| T1v1 B256 (1 seed) | 1 | 1.6% [1.1, 2.2] | 4.0% [2.9, 5.2] | 6.2% [4.6, 7.9] | 634 |
| T1 guided B256 (1 seed) | 1 | 4.1% [3.2, 5.1] | 9.9% [8.2, 11.6] | 15.5% [13.1, 17.9] | 1328 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T1 B0 − T0 | pass@1 | −0.7 [−1.2, −0.3] | <0.0003 | 22/47 (0.004) | [−1.1, −0.4] (839.0) | 2 vs 2 |
| T1 B0 − T0 | pass@5 | −0.9 [−1.7, −0.2] | 0.016 | 20/49 (6.4e-04) | [−1.5, −0.4] (839.0) | 2 vs 2 |
| T1 B0 − T0 | pass@16 | −0.8 [−2.0, +0.4] | 0.182 | 17/28 (0.135) | [−1.7, +0.1] (171.6) | 2 vs 2 |
| T1 B64 − T0 | pass@1 | −1.0 [−1.5, −0.6] | <0.0003 | 11/56 (2.1e-08) | [−1.4, −0.6] (839.0) | 2 vs 2 |
| T1 B64 − T0 | pass@5 | −1.5 [−2.4, −0.7] | 5.0e-04 | 13/54 (4.5e-07) | [−2.7, −0.4] (4.2) | 2 vs 2 |
| T1 B64 − T0 | pass@16 | −1.7 [−3.3, +0.0] | 0.062 | 10/34 (3.9e-04) | [−7.6, +4.3] (1.5) | 2 vs 2 |
| T1 B256 − T0 | pass@1 | −0.5 [−0.9, −0.2] | 0.003 | 34/47 (0.182) | [−0.8, −0.2] (839.0) | 2 vs 2 |
| T1 B256 − T0 | pass@5 | −0.4 [−1.2, +0.4] | 0.261 | 35/46 (0.266) | [−1.5, +0.7] (3.7) | 2 vs 2 |
| T1 B256 − T0 | pass@16 | −0.1 [−1.8, +1.9] | 0.957 | 28/27 (1.000) | [−6.0, +5.8] (1.6) | 2 vs 2 |
| T0 guided − T0 | pass@1 | +2.3 [+1.7, +2.8] | <0.0003 | 135/4 (4.4e-35) | [+1.8, +2.8] (839.0) | 2 vs 2 (paired) |
| T0 guided − T0 | pass@5 | +4.5 [+3.5, +5.6] | <0.0003 | 129/6 (3.6e-31) | [+3.6, +5.4] (839.0) | 2 vs 2 (paired) |
| T0 guided − T0 | pass@16 | +6.5 [+5.0, +8.2] | <0.0003 | 94/3 (1.9e-24) | [+5.2, +7.9] (839.0) | 2 vs 2 (paired) |
| T0e (1 seed) − T0 | pass@1 | −0.5 [−0.9, −0.1] | 0.009 | 17/46 (3.4e-04) | [−0.8, −0.2] (839.0) | 1 vs 2 |
| T0e (1 seed) − T0 | pass@5 | −0.7 [−1.4, −0.1] | 0.021 | 18/44 (0.001) | [−1.3, −0.2] (839.0) | 1 vs 2 |
| T0e (1 seed) − T0 | pass@16 | −1.0 [−2.1, +0.2] | 0.115 | 12/30 (0.008) | [−1.9, +0.0] (839.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@1 | −0.2 [−0.7, +0.2] | 0.281 | 33/40 (0.483) | [−0.7, +0.2] (839.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@5 | +0.4 [−0.4, +1.1] | 0.349 | 39/36 (0.818) | [−0.3, +1.0] (839.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@16 | +1.1 [−0.2, +2.4] | 0.123 | 28/23 (0.576) | [−0.1, +2.3] (839.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@1 | +2.3 [+1.7, +2.9] | <0.0003 | 114/22 (3.6e-16) | [+1.7, +2.9] (839.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@5 | +6.2 [+4.9, +7.6] | <0.0003 | 116/20 (1.2e-17) | [+4.9, +7.6] (839.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@16 | +10.4 [+8.2, +12.5] | <0.0003 | 103/10 (1.3e-20) | [+8.3, +12.4] (839.0) | 1 vs 2 |
| T1 B256 − T1 B0 | pass@1 | +0.2 [+0.0, +0.5] | 0.042 | 43/24 (0.027) | [+0.0, +0.4] (839.0) | 2 vs 2 (paired) |
| T1 B256 − T1 B0 | pass@5 | +0.5 [−0.0, +1.2] | 0.073 | 44/24 (0.021) | [+0.0, +1.0] (839.0) | 2 vs 2 (paired) |
| T1 B256 − T1 B0 | pass@16 | +0.8 [−0.5, +2.3] | 0.273 | 28/13 (0.028) | [−1.3, +2.8] (2.9) | 2 vs 2 (paired) |
| T1 B256 − T0 guided | pass@1 | −2.8 [−3.5, −2.1] | <0.0003 | 9/129 (2.4e-28) | [−3.4, −2.1] (839.0) | 2 vs 2 |
| T1 B256 − T0 guided | pass@5 | −4.9 [−6.3, −3.6] | <0.0003 | 12/124 (1.3e-24) | [−6.2, −3.7] (21.9) | 2 vs 2 |
| T1 B256 − T0 guided | pass@16 | −6.6 [−8.8, −4.4] | <0.0003 | 9/96 (1.6e-19) | [−10.7, −2.5] (2.7) | 2 vs 2 |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T0 | 3.3% [2.3, 4.4] | 4.1% [2.9, 5.3] | 5.0% [3.6, 6.4] | – |
| T1 B0 | 2.2% [1.4, 3.0] | 3.0% [2.0, 4.1] | 3.9% [2.7, 5.3] | – |
| T1 B64 | 1.4% [0.8, 2.1] | 2.1% [1.2, 3.0] | 2.8% [1.6, 4.1] | – |
| T1 B256 | 1.6% [1.0, 2.3] | 2.4% [1.6, 3.2] | 3.3% [2.2, 4.5] | 4.4% [2.9, 5.9] |
| T0 guided | 4.4% [3.4, 5.6] | 6.0% [4.8, 7.4] | 8.0% [6.4, 9.6] | 10.1% [8.3, 12.0] |
| T0e (1 seed) | 2.6% [1.7, 3.6] | 3.3% [2.3, 4.4] | 4.0% [2.8, 5.4] | – |
| T1v1 B256 (1 seed) | – | 2.2% [1.5, 3.0] | 3.3% [2.3, 4.3] | 4.4% [3.2, 5.7] |
| T1 guided B256 (1 seed) | – | – | 5.4% [4.3, 6.5] | 7.7% [6.3, 9.2] |

| difference (pp) | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T1 B0 − T0 | −1.1 [−1.8, −0.4] p 5.0e-04 | −1.1 [−1.9, −0.2] p 0.014 | −1.0 [−2.1, +0.1] p 0.083 | – |
| T1 B64 − T0 | −1.8 [−2.6, −1.2] p <0.0003 | −2.0 [−2.9, −1.2] p <0.0003 | −2.2 [−3.5, −1.0] p <0.0003 | – |
| T1 B256 − T0 | −1.7 [−2.3, −1.0] p <0.0003 | −1.7 [−2.5, −1.0] p <0.0003 | −1.7 [−2.7, −0.6] p 5.0e-04 | – |
| T0 guided − T0 | +1.2 [+0.6, +1.8] p <0.0003 | +2.0 [+1.3, +2.7] p <0.0003 | +3.0 [+2.0, +4.0] p <0.0003 | – |
| T0e (1 seed) − T0 | −0.7 [−1.3, −0.1] p 0.017 | −0.8 [−1.6, −0.0] p 0.043 | −0.9 [−2.0, +0.2] p 0.089 | – |
| T1v1 B256 (1 seed) − T0 | – | −1.9 [−2.7, −1.1] p <0.0003 | −1.7 [−2.7, −0.8] p <0.0003 | – |
| T1 guided B256 (1 seed) − T0 | – | – | +0.4 [−0.7, +1.5] p 0.454 | – |
| T1 B256 − T1 B0 | −0.6 [−1.0, −0.2] p 0.004 | −0.6 [−1.3, −0.1] p 0.014 | −0.6 [−1.4, +0.1] p 0.079 | – |
| T1 B256 − T0 guided | −2.8 [−3.6, −2.1] p <0.0003 | −3.7 [−4.6, −2.8] p <0.0003 | −4.7 [−5.9, −3.5] p <0.0003 | −5.7 [−7.4, −4.1] p <0.0003 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 658 | 0 | 182 |
| pass@5 | 658 | 0 | 182 |
| pass@16 | 658 | 14 | 168 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 840) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T1 B0 − T0 | pass@1 | 0.0 / 0.0 | 4.0 | 0.2 / 0.2 | 0.5 | 2 | 26 |
| T1 B0 − T0 | pass@5 | 0.0 / 0.0 | 3.6 | 0.4 / 0.3 | 0.8 | 2 | 65 |
| T1 B0 − T0 | pass@16 | 0.2 / 0.0 | 0.0 | 0.6 / 0.5 | 1.4 | 2 | 169 |
| T1 B64 − T0 | pass@1 | 0.0 / 0.0 | 4.8 | 0.2 / 0.2 | 0.6 | 2 | 31 |
| T1 B64 − T0 | pass@5 | 0.4 / 0.0 | 5.3 | 0.4 / 0.4 | 1.2 | 2 | 71 |
| T1 B64 − T0 | pass@16 | 1.3 / 0.0 | 2.0 | 0.9 / 1.0 | 2.8 | 2 | 519 |
| T1 B256 − T0 | pass@1 | 0.0 / 0.0 | 2.3 | 0.2 / 0.2 | 0.4 | 2 | 17 |
| T1 B256 − T0 | pass@5 | 0.4 / 0.0 | 1.9 | 0.4 / 0.4 | 1.1 | 2 | 56 |
| T1 B256 − T0 | pass@16 | 1.3 / 0.0 | 1.0 | 0.9 / 1.1 | 3.0 | 2 | 813 |
| T0 guided − T0 | pass@1 | 0.0 | 5.7 | 0.3 / 0.3 | 0.7 | 2 | 47 |
| T0 guided − T0 | pass@5 | 0.0 | 10.0 | 0.5 / 0.5 | 1.3 | 2 | 162 |
| T0 guided − T0 | pass@16 | 0.0 | 13.1 | 0.8 / 0.7 | 2.0 | 2 | 363 |
| T0e (1 seed) − T0 | pass@1 | – / 0.0 | 3.6 | 0.2 / 0.2 | 0.4 | n/a (1 seed) | 19 |
| T0e (1 seed) − T0 | pass@5 | – / 0.0 | 6.2 | 0.3 / 0.3 | 0.8 | n/a (1 seed) | 62 |
| T0e (1 seed) − T0 | pass@16 | – / 0.0 | 10.9 | 0.6 / 0.5 | 1.4 | n/a (1 seed) | 187 |
| T1v1 B256 (1 seed) − T0 | pass@1 | – / 0.0 | 5.6 | 0.2 / 0.2 | 0.6 | n/a (1 seed) | 34 |
| T1v1 B256 (1 seed) − T0 | pass@5 | – / 0.0 | 8.0 | 0.4 / 0.3 | 0.9 | n/a (1 seed) | 84 |
| T1v1 B256 (1 seed) − T0 | pass@16 | – / 0.0 | 15.0 | 0.7 / 0.6 | 1.7 | n/a (1 seed) | 280 |
| T1 guided B256 (1 seed) − T0 | pass@1 | – / 0.0 | 8.5 | 0.3 / 0.3 | 0.9 | n/a (1 seed) | 71 |
| T1 guided B256 (1 seed) − T0 | pass@5 | – / 0.0 | 19.1 | 0.7 / 0.7 | 1.9 | n/a (1 seed) | 346 |
| T1 guided B256 (1 seed) − T0 | pass@16 | – / 0.0 | 29.3 | 1.1 / 1.1 | 3.0 | n/a (1 seed) | 832 |
| T1 B256 − T1 B0 | pass@1 | 0.0 | 1.0 | 0.1 / 0.1 | 0.3 | 2 | 7 |
| T1 B256 − T1 B0 | pass@5 | 0.0 | 2.3 | 0.3 / 0.2 | 0.7 | 2 | 41 |
| T1 B256 − T1 B0 | pass@16 | 0.7 | 0.0 | 0.7 / 0.6 | 1.8 | 2 | 180 |
| T1 B256 − T0 guided | pass@1 | 0.0 / 0.0 | 8.0 | 0.4 / 0.3 | 1.0 | 2 | 85 |
| T1 B256 − T0 guided | pass@5 | 0.4 / 0.0 | 10.5 | 0.7 / 0.6 | 1.7 | 2 | 224 |
| T1 B256 − T0 guided | pass@16 | 1.3 / 0.0 | 12.2 | 1.1 / 1.2 | 3.4 | 3 | 1970 |

