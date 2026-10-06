### textbook_dev (58) inside the OOD reads, k 16

```
python -m code.thinking.stats table --group T0=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s1/rows_plain_B0.jsonl --group 'T1 B0=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B0.jsonl' --group 'T1 B64=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B64.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B64.jsonl' --group 'T1 B256=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B256.jsonl' --group 'T0 guided=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s0/rows_logical_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s1/rows_logical_B0.jsonl' --group 'T0e (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0e_s0/rows_plain_B0.jsonl' --group 'T1v1 B256 (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1v1_s0/rows_plain_B256.jsonl' --group 'T1 guided B256 (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_logical_T1_s0/rows_logical_B256.jsonl' --pair 'T1 B256 vs T1 B0' --pair 'T1 B256 vs T0 guided' --k 1,5,n --tokens 500,1000,2000,4000 --sets textbook_dev --title 'textbook_dev (58) inside the OOD reads, k 16' --out code/experiments/current/thinking_traces_20261005/stage1_stats/ood_textbook_dev.md
```

n = 58 theorems (textbook_dev); 4000 draws, RNG seed 0; resampled: both. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | pass@16 | tokens / attempt |
|---|---|---|---|---|---|
| T0 | 2 | 16.3% [9.3, 24.8] | 25.4% [16.1, 35.7] | 31.9% [20.7, 43.1] | 117 |
| T1 B0 | 2 | 10.8% [4.5, 18.1] | 17.7% [8.8, 27.6] | 25.0% [13.8, 37.9] | 130 |
| T1 B64 | 2 | 12.1% [5.9, 18.8] | 24.7% [14.2, 35.8] | 35.3% [20.7, 50.0] | 212 |
| T1 B256 | 2 | 13.7% [6.8, 21.1] | 26.6% [15.4, 38.7] | 37.1% [24.1, 50.0] | 329 |
| T0 guided | 2 | 26.0% [17.6, 35.1] | 38.8% [27.7, 50.0] | 43.1% [31.0, 55.2] | 251 |
| T0e (1 seed) | 1 | 13.4% [6.4, 21.7] | 19.7% [11.1, 29.6] | 25.9% [15.5, 37.9] | 111 |
| T1v1 B256 (1 seed) | 1 | 17.7% [10.7, 25.2] | 33.2% [23.1, 43.7] | 44.8% [32.8, 56.9] | 541 |
| T1 guided B256 (1 seed) | 1 | 30.5% [21.1, 40.0] | 46.4% [34.7, 57.8] | 55.2% [43.1, 67.2] | 873 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T1 B0 − T0 | pass@1 | −5.5 [−10.1, −1.7] | 0.004 | 3/22 (1.6e-04) | [−9.7, −1.4] (7.5) | 2 vs 2 |
| T1 B0 − T0 | pass@5 | −7.7 [−15.4, −0.3] | 0.041 | 4/18 (0.004) | [−17.9, +2.4] (3.3) | 2 vs 2 |
| T1 B0 − T0 | pass@16 | −6.9 [−17.2, +3.4] | 0.240 | 4/11 (0.118) | [−20.6, +6.8] (4.1) | 2 vs 2 |
| T1 B64 − T0 | pass@1 | −4.3 [−8.9, +0.1] | 0.057 | 8/20 (0.036) | [−8.9, +0.4] (10.2) | 2 vs 2 |
| T1 B64 − T0 | pass@5 | −0.7 [−9.8, +8.1] | 0.856 | 9/17 (0.169) | [−20.2, +18.7] (2.2) | 2 vs 2 |
| T1 B64 − T0 | pass@16 | +3.4 [−9.5, +16.4] | 0.650 | 8/5 (0.581) | [−22.0, +28.9] (2.4) | 2 vs 2 |
| T1 B256 − T0 | pass@1 | −2.6 [−7.2, +1.7] | 0.232 | 9/19 (0.087) | [−7.7, +2.5] (6.0) | 2 vs 2 |
| T1 B256 − T0 | pass@5 | +1.2 [−7.9, +11.0] | 0.795 | 10/14 (0.541) | [−27.0, +29.4] (1.7) | 2 vs 2 |
| T1 B256 − T0 | pass@16 | +5.2 [−6.0, +17.2] | 0.411 | 10/6 (0.454) | [−9.2, +19.5] (4.7) | 2 vs 2 |
| T0 guided − T0 | pass@1 | +9.6 [+5.5, +14.4] | <0.0003 | 25/1 (8.0e-07) | [+5.2, +14.1] (16.2) | 2 vs 2 (paired) |
| T0 guided − T0 | pass@5 | +13.3 [+6.8, +21.0] | <0.0003 | 24/1 (1.5e-06) | [+7.3, +19.4] (57.0) | 2 vs 2 (paired) |
| T0 guided − T0 | pass@16 | +11.2 [+3.4, +20.7] | 0.013 | 12/1 (0.003) | [+4.2, +18.2] (57.0) | 2 vs 2 (paired) |
| T0e (1 seed) − T0 | pass@1 | −3.0 [−8.1, +1.3] | 0.195 | 6/18 (0.023) | [−7.3, +1.4] (57.0) | 1 vs 2 |
| T0e (1 seed) − T0 | pass@5 | −5.7 [−12.2, −0.3] | 0.043 | 6/15 (0.078) | [−10.6, −0.8] (57.0) | 1 vs 2 |
| T0e (1 seed) − T0 | pass@16 | −6.0 [−13.8, +2.6] | 0.212 | 3/11 (0.057) | [−13.0, +1.0] (57.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@1 | +1.3 [−3.2, +5.8] | 0.560 | 17/15 (0.860) | [−2.8, +5.5] (57.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@5 | +7.8 [−1.0, +16.9] | 0.093 | 19/9 (0.087) | [−0.8, +16.4] (57.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@16 | +12.9 [+0.0, +25.9] | 0.053 | 15/7 (0.134) | [+1.2, +24.6] (57.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@1 | +14.2 [+7.0, +21.3] | 5.0e-04 | 26/4 (5.9e-05) | [+7.0, +21.3] (57.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@5 | +21.0 [+11.6, +30.8] | <0.0003 | 24/4 (1.8e-04) | [+11.7, +30.2] (57.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@16 | +23.3 [+12.9, +34.5] | <0.0003 | 20/1 (2.1e-05) | [+13.4, +33.2] (57.0) | 1 vs 2 |
| T1 B256 − T1 B0 | pass@1 | +3.0 [+0.5, +5.6] | 0.021 | 19/6 (0.015) | [+0.7, +5.3] (57.0) | 2 vs 2 (paired) |
| T1 B256 − T1 B0 | pass@5 | +8.9 [+2.9, +16.2] | 0.003 | 18/5 (0.011) | [+3.5, +14.3] (57.0) | 2 vs 2 (paired) |
| T1 B256 − T1 B0 | pass@16 | +12.1 [+1.7, +22.4] | 0.021 | 13/3 (0.021) | [+3.4, +20.7] (57.0) | 2 vs 2 (paired) |
| T1 B256 − T0 guided | pass@1 | −12.2 [−18.6, −6.7] | <0.0003 | 3/25 (2.7e-05) | [−18.3, −6.1] (18.5) | 2 vs 2 |
| T1 B256 − T0 guided | pass@5 | −12.2 [−23.1, −1.8] | 0.019 | 4/22 (5.3e-04) | [−34.5, +10.2] (2.3) | 2 vs 2 |
| T1 B256 − T0 guided | pass@16 | −6.0 [−17.2, +5.2] | 0.321 | 3/9 (0.146) | [−20.4, +8.4] (4.4) | 2 vs 2 |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T0 | 24.6% [15.3, 35.0] | 28.3% [18.3, 39.4] | – | – |
| T1 B0 | 16.5% [7.9, 26.2] | 20.1% [10.3, 30.9] | 24.7% [13.3, 37.4]† | – |
| T1 B64 | 18.3% [9.9, 27.6] | 24.2% [13.8, 35.4] | 30.5% [18.2, 43.6] | – |
| T1 B256 | 16.6% [8.6, 25.5] | 22.0% [12.0, 32.9] | 28.5% [16.6, 41.6] | 35.2% [22.2, 49.0] |
| T0 guided | 32.4% [22.1, 43.4] | 37.5% [26.4, 49.0] | 40.8% [29.3, 52.6] | 43.1% [31.0, 55.2]† |
| T0e (1 seed) | 19.3% [10.6, 29.3] | 22.5% [13.0, 33.3] | – | – |
| T1v1 B256 (1 seed) | – | 23.4% [14.8, 33.0] | 30.2% [20.2, 40.8] | 37.2% [26.0, 48.7] |
| T1 guided B256 (1 seed) | – | 32.0% [21.6, 43.0] | 39.6% [28.1, 51.2] | 45.8% [33.7, 57.7] |

| difference (pp) | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T1 B0 − T0 | −8.1 [−15.5, −1.1] p 0.022 | −8.2 [−17.1, +0.7] p 0.071 | – | – |
| T1 B64 − T0 | −6.3 [−13.8, +0.8] p 0.085 | −4.1 [−13.5, +4.9] p 0.363 | – | – |
| T1 B256 − T0 | −8.0 [−14.9, −1.2] p 0.025 | −6.3 [−15.0, +2.6] p 0.158 | – | – |
| T0 guided − T0 | +7.8 [+3.1, +13.6] p 5.0e-04 | +9.2 [+3.4, +16.3] p 5.0e-04 | – | – |
| T0e (1 seed) − T0 | −5.3 [−11.8, +0.1] p 0.053 | −5.8 [−12.8, +0.5] p 0.074 | – | – |
| T1v1 B256 (1 seed) − T0 | – | −4.9 [−13.2, +3.1] p 0.241 | – | – |
| T1 guided B256 (1 seed) − T0 | – | +3.7 [−4.7, +12.6] p 0.363 | – | – |
| T1 B256 − T1 B0 | +0.1 [−3.8, +3.8] p 0.925 | +1.8 [−3.5, +7.5] p 0.481 | +3.8 [−4.4, +12.3]† p 0.350 | – |
| T1 B256 − T0 guided | −15.8 [−24.1, −8.1] p <0.0003 | −15.6 [−25.6, −6.1] p 0.001 | −12.3 [−23.8, −1.5] p 0.025 | −7.9 [−19.7, +3.0]† p 0.164 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 22 | 0 | 36 |
| pass@5 | 22 | 2 | 34 |
| pass@16 | 22 | 7 | 29 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 58) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T1 B0 − T0 | pass@1 | 1.5 / 0.0 | 0.0 | 2.1 / 1.8 | 5.5 | 7 | 5282 |
| T1 B0 − T0 | pass@5 | 3.5 / 0.0 | 0.0 | 3.8 / 3.4 | 10.7 | 26 | never |
| T1 B0 − T0 | pass@16 | 4.9 / 0.0 | 0.0 | 5.4 / 5.0 | 14.9 | 50 | never |
| T1 B64 − T0 | pass@1 | 1.6 / 0.0 | 7.3 | 2.3 / 2.1 | 5.9 | 30 | never |
| T1 B64 − T0 | pass@5 | 5.7 / 0.0 | 6.8 | 4.5 / 4.9 | 13.7 | 138 | never |
| T1 B64 − T0 | pass@16 | 7.8 / 0.0 | 5.1 | 6.5 / 6.8 | 19.1 | 132 | never |
| T1 B256 − T0 | pass@1 | 1.9 / 0.0 | 4.7 | 2.2 / 2.1 | 5.9 | 11 | never |
| T1 B256 − T0 | pass@5 | 6.7 / 0.0 | 0.0 | 4.8 / 5.4 | 15.4 | 53 | never |
| T1 B256 − T0 | pass@16 | 5.2 / 0.0 | 9.0 | 5.7 / 5.5 | 15.3 | >10k | never |
| T0 guided − T0 | pass@1 | 1.4 | 10.8 | 2.3 / 2.1 | 5.9 | >10k | 1466 |
| T0 guided − T0 | pass@5 | 0.0 | 16.4 | 3.6 / 3.0 | 8.5 | >10k | 468 |
| T0 guided − T0 | pass@16 | 0.0 | 15.2 | 4.4 / 3.5 | 9.8 | >10k | 616 |
| T0e (1 seed) − T0 | pass@1 | – / 0.0 | 13.5 | 2.4 / 2.2 | 6.1 | n/a (1 seed) | 238 |
| T0e (1 seed) − T0 | pass@5 | – / 0.0 | 7.2 | 3.0 / 2.4 | 6.8 | n/a (1 seed) | 301 |
| T0e (1 seed) − T0 | pass@16 | – / 0.0 | 11.8 | 4.3 / 3.5 | 9.8 | n/a (1 seed) | 618 |
| T1v1 B256 (1 seed) − T0 | pass@1 | – / 0.0 | 12.6 | 2.3 / 2.1 | 5.8 | n/a (1 seed) | 219 |
| T1v1 B256 (1 seed) − T0 | pass@5 | – / 0.0 | 28.0 | 4.6 / 4.3 | 12.1 | n/a (1 seed) | 937 |
| T1v1 B256 (1 seed) − T0 | pass@16 | – / 0.0 | 37.5 | 6.3 / 5.8 | 16.4 | n/a (1 seed) | 1726 |
| T1 guided B256 (1 seed) − T0 | pass@1 | – / 0.0 | 25.5 | 3.7 / 3.6 | 10.0 | n/a (1 seed) | 645 |
| T1 guided B256 (1 seed) − T0 | pass@5 | – / 0.0 | 30.7 | 4.9 / 4.6 | 12.9 | n/a (1 seed) | 1080 |
| T1 guided B256 (1 seed) − T0 | pass@16 | – / 0.0 | 29.2 | 5.5 / 5.0 | 13.9 | n/a (1 seed) | 1241 |
| T1 B256 − T1 B0 | pass@1 | 0.0 | 5.5 | 1.3 / 1.1 | 3.2 | 3 | 67 |
| T1 B256 − T1 B0 | pass@5 | 0.0 | 13.5 | 3.3 / 2.7 | 7.5 | >10k | 366 |
| T1 B256 − T1 B0 | pass@16 | 0.0 | 19.4 | 4.9 / 4.3 | 12.1 | >10k | 942 |
| T1 B256 − T0 guided | pass@1 | 1.9 / 0.0 | 14.2 | 3.1 / 2.9 | 8.2 | >10k | never |
| T1 B256 − T0 guided | pass@5 | 6.7 / 0.0 | 16.3 | 5.4 / 5.9 | 16.5 | >10k | never |
| T1 B256 − T0 guided | pass@16 | 5.2 / 0.0 | 15.7 | 5.6 / 5.4 | 15.0 | >10k | never |

