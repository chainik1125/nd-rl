### pass@k with hierarchical-bootstrap intervals

```
python -m code.thinking.stats table --group T0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0_s2/rows_plain_B0.jsonl --group T0g=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0g_s0/rows_plain_B0.jsonl --group 'T1v1 B0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s2/rows_plain_B0.jsonl' --group 'T1v1g B0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1g_s0/rows_plain_B0.jsonl' --group 'T1v1 B256=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s0/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s1/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s2/rows_plain_B256.jsonl' --group 'T1v1g B256=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1g_s0/rows_plain_B256.jsonl' --group 'T1v1s0 B256=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s0/rows_plain_B256.jsonl' --group 'T1v1s0 B0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s0/rows_plain_B0.jsonl' --group T0s0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0_s0/rows_plain_B0.jsonl --k 1,5 --tokens 1000,3000 --pair 'T1v1g B256 vs T1v1 B256' --pair 'T1v1g B256 vs T1v1s0 B256' --pair 'T1v1g B0 vs T1v1 B0' --pair 'T0g vs T0' --pair 'T0g vs T0s0' --pair 'T1v1g B256 vs T1v1g B0' --out /home/cowifornia/.claude/jobs/99f17bd8/tmp/goalstack_tb72.md
```

n = 72 theorems (textbook_dev, textbook_train); 4000 draws, RNG seed 0; resampled: both. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | tokens / attempt |
|---|---|---|---|---|
| T0 | 3 | 17.2% [10.4, 25.1] | 26.0% [17.6, 35.2] | 116 |
| T0g | 1 | 15.3% [8.8, 22.7] | 22.9% [14.2, 32.7] | 126 |
| T1v1 B0 | 3 | 13.1% [7.0, 20.2] | 20.2% [12.5, 28.9] | 125 |
| T1v1g B0 | 1 | 14.2% [7.6, 21.8] | 19.4% [11.4, 28.5] | 105 |
| T1v1 B256 | 3 | 16.8% [10.2, 24.1] | 29.3% [20.2, 38.7] | 420 |
| T1v1g B256 | 1 | 17.1% [10.2, 24.9] | 25.8% [17.2, 35.1] | 293 |
| T1v1s0 B256 | 1 | 19.1% [12.4, 26.4] | 33.3% [24.2, 42.4] | 508 |
| T1v1s0 B0 | 1 | 12.6% [6.7, 19.4] | 19.8% [12.2, 28.4] | 134 |
| T0s0 | 1 | 18.6% [11.3, 27.0] | 27.0% [17.8, 37.0] | 116 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T0g − T0 | pass@1 | −1.9 [−5.5, +1.4] | 0.240 | 10/30 (0.002) | [−4.7, +0.8] (73.0) | 1 vs 3 |
| T0g − T0 | pass@5 | −3.0 [−8.2, +2.1] | 0.242 | 14/26 (0.081) | [−7.7, +1.6] (71.0) | 1 vs 3 |
| T1v1 B0 − T0 | pass@1 | −4.1 [−7.7, −0.9] | 0.015 | 7/31 (1.2e-04) | [−6.7, −1.5] (72.9) | 3 vs 3 |
| T1v1 B0 − T0 | pass@5 | −5.8 [−10.4, −1.7] | 0.005 | 8/31 (2.9e-04) | [−9.2, −2.4] (71.0) | 3 vs 3 |
| T1v1g B0 − T0 | pass@1 | −3.0 [−6.4, +0.6] | 0.101 | 11/28 (0.009) | [−6.1, +0.1] (72.9) | 1 vs 3 |
| T1v1g B0 − T0 | pass@5 | −6.6 [−10.8, −2.5] | 0.001 | 11/28 (0.009) | [−10.4, −2.8] (71.0) | 1 vs 3 |
| T1v1 B256 − T0 | pass@1 | −0.4 [−4.0, +3.3] | 0.810 | 26/22 (0.665) | [−3.9, +3.1] (8.1) | 3 vs 3 |
| T1v1 B256 − T0 | pass@5 | +3.4 [−2.1, +9.2] | 0.229 | 31/17 (0.059) | [−2.4, +9.2] (9.1) | 3 vs 3 |
| T1v1g B256 − T0 | pass@1 | −0.1 [−3.5, +3.4] | 0.974 | 15/24 (0.200) | [−3.0, +2.9] (73.0) | 1 vs 3 |
| T1v1g B256 − T0 | pass@5 | −0.1 [−4.4, +4.5] | 0.927 | 21/19 (0.875) | [−4.1, +3.8] (71.0) | 1 vs 3 |
| T1v1s0 B256 − T0 | pass@1 | +1.9 [−1.7, +5.3] | 0.288 | 30/14 (0.023) | [−1.1, +4.8] (73.0) | 1 vs 3 |
| T1v1s0 B256 − T0 | pass@5 | +7.4 [+1.5, +13.0] | 0.011 | 33/11 (0.001) | [+2.0, +12.7] (71.0) | 1 vs 3 |
| T1v1s0 B0 − T0 | pass@1 | −4.6 [−8.6, −1.3] | 0.009 | 8/31 (2.9e-04) | [−7.9, −1.4] (72.9) | 1 vs 3 |
| T1v1s0 B0 − T0 | pass@5 | −6.2 [−11.3, −1.7] | 0.007 | 10/29 (0.003) | [−10.8, −1.6] (71.0) | 1 vs 3 |
| T0s0 − T0 | pass@1 | +1.4 [−1.1, +4.9] | 0.370 | 18/21 (0.749) | [−0.9, +3.7] (72.6) | 1 vs 3 |
| T0s0 − T0 | pass@5 | +1.1 [−2.3, +5.3] | 0.564 | 22/17 (0.522) | [−2.0, +4.1] (71.0) | 1 vs 3 |
| T1v1g B256 − T1v1 B256 | pass@1 | +0.4 [−3.4, +4.0] | 0.851 | 16/31 (0.040) | [−3.6, +4.3] (15.6) | 1 vs 3 |
| T1v1g B256 − T1v1 B256 | pass@5 | −3.5 [−9.5, +2.3] | 0.229 | 19/29 (0.193) | [−9.8, +2.7] (14.3) | 1 vs 3 |
| T1v1g B256 − T1v1s0 B256 | pass@1 | −1.9 [−5.5, +2.0] | 0.306 | 16/28 (0.096) | [−5.6, +1.8] (71.0) | 1 vs 1 |
| T1v1g B256 − T1v1s0 B256 | pass@5 | −7.5 [−13.2, −1.4] | 0.013 | 16/28 (0.096) | [−13.5, −1.5] (71.0) | 1 vs 1 |
| T1v1g B0 − T1v1 B0 | pass@1 | +1.1 [−0.8, +3.6] | 0.295 | 14/20 (0.392) | [−0.9, +3.1] (71.0) | 1 vs 3 |
| T1v1g B0 − T1v1 B0 | pass@5 | −0.8 [−4.2, +2.7] | 0.633 | 15/19 (0.608) | [−3.6, +2.0] (71.0) | 1 vs 3 |
| T0g − T0 | pass@1 | −1.9 [−5.5, +1.4] | 0.240 | 10/30 (0.002) | [−4.7, +0.8] (73.0) | 1 vs 3 |
| T0g − T0 | pass@5 | −3.0 [−8.2, +2.1] | 0.242 | 14/26 (0.081) | [−7.7, +1.6] (71.0) | 1 vs 3 |
| T0g − T0s0 | pass@1 | −3.3 [−7.4, +0.1] | 0.067 | 13/24 (0.099) | [−7.0, +0.4] (71.0) | 1 vs 1 |
| T0g − T0s0 | pass@5 | −4.1 [−10.0, +1.6] | 0.154 | 13/24 (0.099) | [−10.0, +1.8] (71.0) | 1 vs 1 |
| T1v1g B256 − T1v1g B0 | pass@1 | +3.0 [+0.8, +5.7] | 0.004 | 18/12 (0.362) | [+0.4, +5.5] (71.0) | 1 vs 1 (paired) |
| T1v1g B256 − T1v1g B0 | pass@5 | +6.5 [+2.3, +11.4] | <0.0003 | 18/12 (0.362) | [+1.7, +11.2] (71.0) | 1 vs 1 (paired) |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 1k | 3k |
|---|---|---|
| T0 | 28.8% [19.9, 38.4] | 34.4% [24.7, 44.8] |
| T0g | 24.5% [15.5, 34.4] | 28.9% [19.4, 39.0] |
| T1v1 B0 | 22.3% [14.2, 31.5] | 27.2% [18.2, 37.0] |
| T1v1g B0 | 21.2% [12.7, 30.9] | 23.8% [14.9, 33.8] |
| T1v1 B256 | 23.8% [15.4, 32.9] | 32.6% [22.6, 42.7] |
| T1v1g B256 | 23.8% [15.3, 33.2] | 29.5% [19.9, 39.6] |
| T1v1s0 B256 | 25.0% [16.7, 33.9] | 34.8% [25.2, 44.5] |
| T1v1s0 B0 | 21.6% [13.6, 30.8] | 27.0% [17.7, 37.2] |
| T0s0 | 29.5% [19.8, 40.0] | 33.7% [23.4, 44.4] |

| difference (pp) | 1k | 3k |
|---|---|---|
| T0g − T0 | −4.3 [−9.8, +1.0] p 0.115 | −5.6 [−11.4, +0.1] p 0.055 |
| T1v1 B0 − T0 | −6.5 [−11.3, −2.2] p 0.001 | −7.2 [−12.2, −2.4] p 0.004 |
| T1v1g B0 − T0 | −7.6 [−12.3, −3.1] p 0.002 | −10.6 [−16.3, −5.2] p <0.0003 |
| T1v1 B256 − T0 | −5.0 [−10.2, −0.3] p 0.038 | −1.8 [−8.1, +3.6] p 0.527 |
| T1v1g B256 − T0 | −4.9 [−9.7, −0.2] p 0.041 | −5.0 [−10.2, +0.4] p 0.070 |
| T1v1s0 B256 − T0 | −3.7 [−9.1, +1.1] p 0.137 | +0.4 [−5.4, +5.8] p 0.909 |
| T1v1s0 B0 − T0 | −7.2 [−12.8, −2.4] p 0.002 | −7.4 [−13.5, −1.9] p 0.008 |
| T0s0 − T0 | +0.7 [−3.1, +5.1] p 0.755 | −0.8 [−6.0, +3.7] p 0.779 |
| T1v1g B256 − T1v1 B256 | +0.1 [−4.5, +5.2] p 0.987 | −3.1 [−9.1, +2.9] p 0.319 |
| T1v1g B256 − T1v1s0 B256 | −1.2 [−6.1, +4.1] p 0.632 | −5.3 [−11.2, +0.8] p 0.091 |
| T1v1g B0 − T1v1 B0 | −1.1 [−5.0, +2.8] p 0.572 | −3.4 [−8.0, +0.6] p 0.102 |
| T0g − T0 | −4.3 [−9.8, +1.0] p 0.115 | −5.6 [−11.4, +0.1] p 0.055 |
| T0g − T0s0 | −5.0 [−11.4, +1.0] p 0.114 | −4.8 [−11.6, +1.6] p 0.155 |
| T1v1g B256 − T1v1g B0 | +2.7 [−1.5, +7.3] p 0.225 | +5.7 [+1.2, +10.5] p 0.009 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 24 | 0 | 48 |
| pass@5 | 24 | 0 | 48 |
| pass@256 | 24 | 16 | 32 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 72) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T0g − T0 | pass@1 | – / 0.4 | 8.7 | 1.7 / 1.4 | 3.9 | n/a (1 seed) | 123 |
| T0g − T0 | pass@5 | – / 0.0 | 17.0 | 2.6 / 2.3 | 6.5 | n/a (1 seed) | 344 |
| T1v1 B0 − T0 | pass@1 | 0.0 / 0.4 | 6.1 | 1.7 / 1.3 | 3.6 | 6 | 106 |
| T1v1 B0 − T0 | pass@5 | 0.0 / 0.0 | 6.5 | 2.2 / 1.7 | 4.8 | 13 | 185 |
| T1v1g B0 − T0 | pass@1 | – / 0.4 | 10.4 | 1.8 / 1.5 | 4.3 | n/a (1 seed) | 152 |
| T1v1g B0 − T0 | pass@5 | – / 0.0 | 12.4 | 2.1 / 1.9 | 5.3 | n/a (1 seed) | 226 |
| T1v1 B256 − T0 | pass@1 | 1.9 / 0.4 | 1.0 | 1.9 / 1.5 | 4.3 | 7 | never |
| T1v1 B256 − T0 | pass@5 | 3.0 / 0.0 | 9.1 | 2.8 / 2.6 | 7.2 | >10k | never |
| T1v1g B256 − T0 | pass@1 | – / 0.4 | 9.7 | 1.8 / 1.5 | 4.1 | n/a (1 seed) | 140 |
| T1v1g B256 − T0 | pass@5 | – / 0.0 | 13.5 | 2.2 / 2.0 | 5.6 | n/a (1 seed) | 252 |
| T1v1s0 B256 − T0 | pass@1 | – / 0.4 | 9.7 | 1.8 / 1.5 | 4.1 | n/a (1 seed) | 139 |
| T1v1s0 B256 − T0 | pass@5 | – / 0.0 | 20.4 | 2.9 / 2.7 | 7.5 | n/a (1 seed) | 455 |
| T1v1s0 B0 − T0 | pass@1 | – / 0.4 | 11.4 | 1.9 / 1.6 | 4.6 | n/a (1 seed) | 171 |
| T1v1s0 B0 − T0 | pass@5 | – / 0.0 | 16.5 | 2.5 / 2.3 | 6.4 | n/a (1 seed) | 331 |
| T0s0 − T0 | pass@1 | – / 0.4 | 5.9 | 1.5 / 1.2 | 3.2 | n/a (1 seed) | 85 |
| T0s0 − T0 | pass@5 | – / 0.0 | 8.0 | 1.9 / 1.5 | 4.3 | n/a (1 seed) | 147 |
| T1v1g B256 − T1v1 B256 | pass@1 | – / 1.9 | 11.7 | 1.9 / 1.8 | 5.2 | n/a (1 seed) | never |
| T1v1g B256 − T1v1 B256 | pass@5 | – / 3.0 | 18.0 | 3.0 / 2.9 | 8.2 | n/a (1 seed) | never |
| T1v1g B256 − T1v1s0 B256 | pass@1 | – / – | 15.8 | 1.9 / 1.9 | 5.2 | n/a (1 seed) | 217 |
| T1v1g B256 − T1v1s0 B256 | pass@5 | – / – | 25.4 | 3.0 / 3.0 | 8.4 | n/a (1 seed) | 565 |
| T1v1g B0 − T1v1 B0 | pass@1 | – / 0.0 | 7.0 | 1.1 / 1.0 | 2.8 | n/a (1 seed) | 62 |
| T1v1g B0 − T1v1 B0 | pass@5 | – / 0.0 | 9.0 | 1.7 / 1.4 | 4.0 | n/a (1 seed) | 127 |
| T0g − T0 | pass@1 | – / 0.4 | 8.7 | 1.7 / 1.4 | 3.9 | n/a (1 seed) | 123 |
| T0g − T0 | pass@5 | – / 0.0 | 17.0 | 2.6 / 2.3 | 6.5 | n/a (1 seed) | 344 |
| T0g − T0s0 | pass@1 | – / – | 15.8 | 1.9 / 1.9 | 5.2 | n/a (1 seed) | 219 |
| T0g − T0s0 | pass@5 | – / – | 25.1 | 3.0 / 3.0 | 8.3 | n/a (1 seed) | 549 |
| T1v1g B256 − T1v1g B0 | pass@1 | – | 10.9 | 1.3 / 1.3 | 3.6 | n/a (1 seed) | 104 |
| T1v1g B256 − T1v1g B0 | pass@5 | – | 20.0 | 2.3 / 2.4 | 6.6 | n/a (1 seed) | 351 |

