### OOD sets without textbook_dev (840), k 16; generated (family, knob) cells resampled as clusters

```
python -m code.thinking.stats table --group T0=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s1/rows_plain_B0.jsonl --group 'T1 B0=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B0.jsonl' --group 'T1 B64=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B64.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B64.jsonl' --group 'T1 B256=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B256.jsonl' --group 'T0 guided=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s0/rows_logical_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s1/rows_logical_B0.jsonl' --group 'T0e (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0e_s0/rows_plain_B0.jsonl' --group 'T1v1 B256 (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1v1_s0/rows_plain_B256.jsonl' --group 'T1 guided B256 (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_logical_T1_s0/rows_logical_B256.jsonl' --pair 'T1 B256 vs T1 B0' --pair 'T1 B256 vs T0 guided' --k 1,5,n --tokens 500,1000,2000,4000 --exclude textbook_dev --clusters /scratch/cowifornia/nd-rl/think/ood_check/sets/ood_or_reorder.jsonl,/scratch/cowifornia/nd-rl/think/ood_check/sets/ood_and_reorder.jsonl,/scratch/cowifornia/nd-rl/think/ood_check/sets/ood_commute.jsonl,/scratch/cowifornia/nd-rl/think/ood_check/sets/ood_case_product.jsonl --cluster-by family,knob --title 'OOD sets without textbook_dev (840), k 16; generated (family, knob) cells resampled as clusters' --out code/experiments/current/thinking_traces_20261005/stage1_stats/ood_clustered.md --json code/experiments/current/thinking_traces_20261005/stage1_stats/ood_clustered.json
```

n = 840 theorems (charles_long_reach, ood_and_reorder, ood_case_product, ood_commute, ood_or_reorder); 4000 draws, RNG seed 0; resampled: both, theorems in 99 clusters. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | pass@16 | tokens / attempt |
|---|---|---|---|---|---|
| T0 | 2 | 1.8% [0.6, 3.4] | 3.7% [1.4, 6.8] | 5.1% [2.0, 9.4] | 140 |
| T1 B0 | 2 | 1.1% [0.2, 2.3] | 2.7% [0.7, 5.5] | 4.3% [1.2, 8.5] | 159 |
| T1 B64 | 2 | 0.8% [0.1, 1.7] | 2.1% [0.5, 4.3] | 3.5% [1.1, 6.8] | 211 |
| T1 B256 | 2 | 1.3% [0.3, 2.7] | 3.2% [1.0, 6.2] | 5.1% [1.7, 9.8] | 370 |
| T0 guided | 2 | 4.1% [1.7, 7.3] | 8.2% [3.5, 14.2] | 11.7% [5.3, 19.9] | 426 |
| T0e (1 seed) | 1 | 1.4% [0.4, 2.7] | 2.9% [0.9, 5.7] | 4.2% [1.1, 8.3] | 139 |
| T1v1 B256 (1 seed) | 1 | 1.6% [0.4, 3.1] | 4.0% [1.4, 7.5] | 6.2% [2.5, 11.2] | 634 |
| T1 guided B256 (1 seed) | 1 | 4.1% [1.4, 7.7] | 9.9% [3.8, 17.9] | 15.5% [6.9, 26.8] | 1328 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T1 B0 − T0 | pass@1 | −0.7 [−1.4, −0.2] | <0.0003 | 22/47 (0.004) | [−1.3, −0.2] (98.0) | 2 vs 2 |
| T1 B0 − T0 | pass@5 | −0.9 [−1.8, −0.2] | 0.015 | 20/49 (6.4e-04) | [−1.5, −0.3] (98.0) | 2 vs 2 |
| T1 B0 − T0 | pass@16 | −0.8 [−2.2, +0.4] | 0.158 | 17/28 (0.135) | [−1.7, +0.0] (70.3) | 2 vs 2 |
| T1 B64 − T0 | pass@1 | −1.0 [−1.8, −0.4] | <0.0003 | 11/56 (2.1e-08) | [−1.7, −0.3] (98.0) | 2 vs 2 |
| T1 B64 − T0 | pass@5 | −1.5 [−2.8, −0.5] | 5.0e-04 | 13/54 (4.5e-07) | [−2.7, −0.3] (13.2) | 2 vs 2 |
| T1 B64 − T0 | pass@16 | −1.7 [−4.0, +0.2] | 0.095 | 10/34 (3.9e-04) | [−5.6, +2.3] (2.7) | 2 vs 2 |
| T1 B256 − T0 | pass@1 | −0.5 [−1.1, −0.1] | 0.013 | 34/47 (0.182) | [−0.9, −0.1] (98.0) | 2 vs 2 |
| T1 B256 − T0 | pass@5 | −0.4 [−1.2, +0.2] | 0.190 | 35/46 (0.266) | [−1.7, +0.8] (2.4) | 2 vs 2 |
| T1 B256 − T0 | pass@16 | −0.1 [−1.8, +1.8] | 0.944 | 28/27 (1.000) | [−8.3, +8.2] (1.2) | 2 vs 2 |
| T0 guided − T0 | pass@1 | +2.3 [+1.0, +3.9] | <0.0003 | 135/4 (4.4e-35) | [+0.8, +3.7] (98.0) | 2 vs 2 (paired) |
| T0 guided − T0 | pass@5 | +4.5 [+2.0, +7.8] | <0.0003 | 129/6 (3.6e-31) | [+1.7, +7.3] (98.0) | 2 vs 2 (paired) |
| T0 guided − T0 | pass@16 | +6.5 [+3.1, +11.0] | <0.0003 | 94/3 (1.9e-24) | [+2.7, +10.4] (98.0) | 2 vs 2 (paired) |
| T0e (1 seed) − T0 | pass@1 | −0.5 [−0.9, −0.1] | 0.003 | 17/46 (3.4e-04) | [−0.8, −0.1] (98.0) | 1 vs 2 |
| T0e (1 seed) − T0 | pass@5 | −0.7 [−1.6, −0.0] | 0.045 | 18/44 (0.001) | [−1.5, −0.0] (98.0) | 1 vs 2 |
| T0e (1 seed) − T0 | pass@16 | −1.0 [−2.3, +0.4] | 0.180 | 12/30 (0.008) | [−2.2, +0.3] (98.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@1 | −0.2 [−0.8, +0.3] | 0.378 | 33/40 (0.483) | [−0.8, +0.3] (98.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@5 | +0.4 [−0.4, +1.3] | 0.399 | 39/36 (0.818) | [−0.5, +1.2] (98.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@16 | +1.1 [−0.1, +2.5] | 0.095 | 28/23 (0.576) | [−0.2, +2.3] (98.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@1 | +2.3 [+0.7, +4.4] | <0.0003 | 114/22 (3.6e-16) | [+0.5, +4.1] (98.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@5 | +6.2 [+2.3, +11.4] | <0.0003 | 116/20 (1.2e-17) | [+1.6, +10.8] (98.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@16 | +10.4 [+4.9, +17.7] | <0.0003 | 103/10 (1.3e-20) | [+3.8, +16.9] (98.0) | 1 vs 2 |
| T1 B256 − T1 B0 | pass@1 | +0.2 [+0.0, +0.5] | 0.016 | 43/24 (0.027) | [+0.0, +0.4] (98.0) | 2 vs 2 (paired) |
| T1 B256 − T1 B0 | pass@5 | +0.5 [+0.0, +1.1] | 0.028 | 44/24 (0.021) | [+0.1, +0.9] (98.0) | 2 vs 2 (paired) |
| T1 B256 − T1 B0 | pass@16 | +0.8 [−0.4, +2.3] | 0.259 | 28/13 (0.028) | [−1.2, +2.7] (3.3) | 2 vs 2 (paired) |
| T1 B256 − T0 guided | pass@1 | −2.8 [−4.8, −1.2] | <0.0003 | 9/129 (2.4e-28) | [−4.5, −1.0] (98.0) | 2 vs 2 |
| T1 B256 − T0 guided | pass@5 | −4.9 [−8.7, −2.2] | <0.0003 | 12/124 (1.3e-24) | [−8.0, −1.8] (94.7) | 2 vs 2 |
| T1 B256 − T0 guided | pass@16 | −6.6 [−11.7, −2.9] | <0.0003 | 9/96 (1.6e-19) | [−11.1, −2.1] (24.0) | 2 vs 2 |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T0 | 3.3% [1.2, 6.2] | 4.1% [1.5, 7.7] | 5.0% [1.9, 9.3] | – |
| T1 B0 | 2.2% [0.5, 4.5] | 3.0% [0.8, 6.2] | 3.9% [1.1, 8.0] | – |
| T1 B64 | 1.4% [0.3, 3.0] | 2.1% [0.5, 4.1] | 2.8% [0.8, 5.5] | – |
| T1 B256 | 1.6% [0.4, 3.3] | 2.4% [0.7, 4.6] | 3.3% [1.1, 6.3] | 4.4% [1.5, 8.4] |
| T0 guided | 4.4% [1.7, 8.2] | 6.0% [2.5, 11.0] | 8.0% [3.4, 14.3] | 10.1% [4.4, 17.8] |
| T0e (1 seed) | 2.6% [0.7, 5.2] | 3.3% [0.9, 6.6] | 4.0% [1.0, 8.2] | – |
| T1v1 B256 (1 seed) | – | 2.2% [0.7, 4.3] | 3.3% [1.1, 6.2] | 4.4% [1.6, 8.2] |
| T1 guided B256 (1 seed) | – | – | 5.4% [1.8, 10.4] | 7.7% [2.8, 14.8] |

| difference (pp) | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T1 B0 − T0 | −1.1 [−2.0, −0.4] p 0.001 | −1.1 [−2.1, −0.3] p 0.011 | −1.0 [−2.3, +0.1] p 0.060 | – |
| T1 B64 − T0 | −1.8 [−3.3, −0.7] p <0.0003 | −2.0 [−3.7, −0.8] p 5.0e-04 | −2.2 [−4.4, −0.6] p 0.004 | – |
| T1 B256 − T0 | −1.7 [−3.0, −0.6] p <0.0003 | −1.7 [−3.1, −0.7] p <0.0003 | −1.7 [−3.3, −0.5] p 0.002 | – |
| T0 guided − T0 | +1.2 [+0.4, +2.3] p <0.0003 | +2.0 [+0.8, +3.6] p <0.0003 | +3.0 [+1.3, +5.4] p <0.0003 | – |
| T0e (1 seed) − T0 | −0.7 [−1.5, −0.1] p 0.021 | −0.8 [−1.8, +0.1] p 0.072 | −0.9 [−2.2, +0.4] p 0.155 | – |
| T1v1 B256 (1 seed) − T0 | – | −1.9 [−3.4, −0.7] p 5.0e-04 | −1.7 [−3.3, −0.5] p 0.003 | – |
| T1 guided B256 (1 seed) − T0 | – | – | +0.4 [−0.9, +2.2] p 0.605 | – |
| T1 B256 − T1 B0 | −0.6 [−1.4, −0.1] p 0.026 | −0.6 [−1.7, +0.0] p 0.057 | −0.6 [−1.9, +0.3] p 0.182 | – |
| T1 B256 − T0 guided | −2.8 [−5.2, −1.2] p <0.0003 | −3.7 [−6.7, −1.6] p <0.0003 | −4.7 [−8.5, −2.0] p <0.0003 | −5.7 [−10.5, −2.4] p <0.0003 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 658 | 0 | 182 |
| pass@5 | 658 | 0 | 182 |
| pass@16 | 658 | 14 | 168 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 840) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts (theorem terms are cluster-robust).

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T1 B0 − T0 | pass@1 | 0.0 / 0.0 | 4.0 | 0.3 / 0.3 | 0.8 | 2 | 59 |
| T1 B0 − T0 | pass@5 | 0.0 / 0.0 | 3.6 | 0.4 / 0.3 | 0.8 | 2 | 65 |
| T1 B0 − T0 | pass@16 | 0.2 / 0.0 | 0.0 | 0.6 / 0.4 | 1.3 | 2 | 157 |
| T1 B64 − T0 | pass@1 | 0.0 / 0.0 | 4.8 | 0.4 / 0.4 | 1.0 | 2 | 98 |
| T1 B64 − T0 | pass@5 | 0.4 / 0.0 | 5.3 | 0.6 / 0.6 | 1.6 | 2 | 187 |
| T1 B64 − T0 | pass@16 | 1.3 / 0.0 | 2.0 | 1.1 / 1.2 | 3.3 | 3 | 1446 |
| T1 B256 − T0 | pass@1 | 0.0 / 0.0 | 2.3 | 0.2 / 0.2 | 0.6 | 2 | 36 |
| T1 B256 − T0 | pass@5 | 0.4 / 0.0 | 1.9 | 0.4 / 0.3 | 1.0 | 2 | 34 |
| T1 B256 − T0 | pass@16 | 1.3 / 0.0 | 1.0 | 0.9 / 1.0 | 2.8 | 2 | 334 |
| T0 guided − T0 | pass@1 | 0.0 | 5.7 | 0.8 / 0.7 | 2.1 | 2 | 397 |
| T0 guided − T0 | pass@5 | 0.0 | 10.0 | 1.5 / 1.4 | 4.0 | 43 | 1495 |
| T0 guided − T0 | pass@16 | 0.0 | 13.1 | 2.0 / 1.9 | 5.5 | >10k | 2785 |
| T0e (1 seed) − T0 | pass@1 | – / 0.0 | 3.6 | 0.2 / 0.2 | 0.5 | n/a (1 seed) | 24 |
| T0e (1 seed) − T0 | pass@5 | – / 0.0 | 6.2 | 0.4 / 0.4 | 1.0 | n/a (1 seed) | 103 |
| T0e (1 seed) − T0 | pass@16 | – / 0.0 | 10.9 | 0.7 / 0.6 | 1.7 | n/a (1 seed) | 281 |
| T1v1 B256 (1 seed) − T0 | pass@1 | – / 0.0 | 5.6 | 0.3 / 0.3 | 0.8 | n/a (1 seed) | 53 |
| T1v1 B256 (1 seed) − T0 | pass@5 | – / 0.0 | 8.0 | 0.4 / 0.4 | 1.2 | n/a (1 seed) | 128 |
| T1v1 B256 (1 seed) − T0 | pass@16 | – / 0.0 | 15.0 | 0.7 / 0.6 | 1.7 | n/a (1 seed) | 281 |
| T1 guided B256 (1 seed) − T0 | pass@1 | – / 0.0 | 8.5 | 0.9 / 0.9 | 2.6 | n/a (1 seed) | 612 |
| T1 guided B256 (1 seed) − T0 | pass@5 | – / 0.0 | 19.1 | 2.4 / 2.3 | 6.5 | n/a (1 seed) | 3899 |
| T1 guided B256 (1 seed) − T0 | pass@16 | – / 0.0 | 29.3 | 3.3 / 3.3 | 9.2 | n/a (1 seed) | 7880 |
| T1 B256 − T1 B0 | pass@1 | 0.0 | 1.0 | 0.1 / 0.1 | 0.3 | 2 | 7 |
| T1 B256 − T1 B0 | pass@5 | 0.0 | 2.3 | 0.3 / 0.2 | 0.6 | 2 | 33 |
| T1 B256 − T1 B0 | pass@16 | 0.7 | 0.0 | 0.7 / 0.7 | 1.9 | 2 | 208 |
| T1 B256 − T0 guided | pass@1 | 0.0 / 0.0 | 8.0 | 0.9 / 0.9 | 2.5 | 2 | 561 |
| T1 B256 − T0 guided | pass@5 | 0.4 / 0.0 | 10.5 | 1.6 / 1.6 | 4.4 | 49 | 1859 |
| T1 B256 − T0 guided | pass@16 | 1.3 / 0.0 | 12.2 | 2.2 / 2.2 | 6.1 | >10k | 13390 |

