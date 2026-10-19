### Charles's long problems (Tier A, 80), k 16

```
python -m code.thinking.stats table --group T0=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s1/rows_plain_B0.jsonl --group 'T1 B0=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B0.jsonl' --group 'T1 B64=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B64.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B64.jsonl' --group 'T1 B256=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s0/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1_s1/rows_plain_B256.jsonl' --group 'T0 guided=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s0/rows_logical_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0_s1/rows_logical_B0.jsonl' --group 'T0e (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T0e_s0/rows_plain_B0.jsonl' --group 'T1v1 B256 (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_T1v1_s0/rows_plain_B256.jsonl' --group 'T1 guided B256 (1 seed)=/scratch/cowifornia/nd-rl/think/runs/eval/ood_logical_T1_s0/rows_logical_B256.jsonl' --pair 'T1 B256 vs T1 B0' --pair 'T1 B256 vs T0 guided' --k 1,5,n --tokens 500,1000,2000,4000 --sets charles_long_reach --title 'Charles'"'"'s long problems (Tier A, 80), k 16' --out code/experiments/current/thinking_traces_20261005/stage1_stats/ood_charles.md --json code/experiments/current/thinking_traces_20261005/stage1_stats/ood_charles.json
```

n = 80 theorems (charles_long_reach); 4000 draws, RNG seed 0; resampled: both. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | pass@16 | tokens / attempt |
|---|---|---|---|---|---|
| T0 | 2 | 1.4% [0.1, 3.3] | 3.5% [0.6, 7.2] | 5.6% [1.2, 11.2] | 149 |
| T1 B0 | 2 | 0.2% [0.0, 0.8] | 1.0% [0.0, 3.0] | 1.9% [0.0, 5.0] | 165 |
| T1 B64 | 2 | 0.2% [0.0, 0.6] | 1.1% [0.0, 2.8] | 2.5% [0.0, 6.2] | 239 |
| T1 B256 | 2 | 0.5% [0.0, 1.5] | 2.0% [0.0, 5.2] | 3.8% [0.0, 10.0] | 392 |
| T0 guided | 2 | 4.9% [1.8, 8.6] | 10.6% [4.9, 17.1] | 16.2% [7.5, 26.2] | 423 |
| T0e (1 seed) | 1 | 1.1% [0.0, 2.9] | 2.5% [0.0, 5.9] | 3.8% [0.0, 8.8] | 135 |
| T1v1 B256 (1 seed) | 1 | 1.4% [0.5, 2.6] | 5.3% [1.9, 9.5] | 10.0% [3.8, 17.5] | 628 |
| T1 guided B256 (1 seed) | 1 | 4.6% [2.3, 7.4] | 14.1% [8.2, 20.3] | 23.8% [15.0, 33.8] | 1221 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T1 B0 − T0 | pass@1 | −1.1 [−2.9, −0.1] | 0.010 | 0/7 (0.016) | [−2.5, +0.2] (53.2) | 2 vs 2 |
| T1 B0 − T0 | pass@5 | −2.5 [−5.2, −0.4] | 0.010 | 0/7 (0.016) | [−4.9, −0.1] (78.3) | 2 vs 2 |
| T1 B0 − T0 | pass@16 | −3.8 [−8.1, −0.6] | 0.043 | 0/6 (0.031) | [−6.7, −0.8] (79.0) | 2 vs 2 |
| T1 B64 − T0 | pass@1 | −1.1 [−3.0, +0.0] | 0.061 | 1/7 (0.070) | [−2.6, +0.3] (58.1) | 2 vs 2 |
| T1 B64 − T0 | pass@5 | −2.4 [−5.5, +0.0] | 0.059 | 1/7 (0.070) | [−5.0, +0.2] (79.0) | 2 vs 2 |
| T1 B64 − T0 | pass@16 | −3.1 [−7.5, +1.2] | 0.194 | 1/6 (0.125) | [−7.0, +0.8] (16.5) | 2 vs 2 |
| T1 B256 − T0 | pass@1 | −0.8 [−2.5, +0.2] | 0.172 | 1/5 (0.219) | [−2.2, +0.5] (35.8) | 2 vs 2 |
| T1 B256 − T0 | pass@5 | −1.5 [−4.5, +1.0] | 0.274 | 1/5 (0.219) | [−4.8, +1.9] (7.8) | 2 vs 2 |
| T1 B256 − T0 | pass@16 | −1.9 [−7.5, +3.7] | 0.526 | 1/4 (0.375) | [−13.1, +9.4] (2.0) | 2 vs 2 |
| T0 guided − T0 | pass@1 | +3.5 [+1.2, +6.5] | <0.0003 | 19/0 (3.8e-06) | [+1.4, +5.6] (79.4) | 2 vs 2 (paired) |
| T0 guided − T0 | pass@5 | +7.2 [+2.6, +12.4] | <0.0003 | 19/0 (3.8e-06) | [+1.7, +12.6] (9.0) | 2 vs 2 (paired) |
| T0 guided − T0 | pass@16 | +10.6 [+2.5, +20.0] | 0.002 | 14/0 (1.2e-04) | [−7.7, +28.9] (2.3) | 2 vs 2 (paired) |
| T0e (1 seed) − T0 | pass@1 | −0.3 [−1.3, +0.9] | 0.597 | 2/5 (0.453) | [−1.3, +0.7] (23.7) | 1 vs 2 |
| T0e (1 seed) − T0 | pass@5 | −0.9 [−2.4, +0.4] | 0.190 | 2/5 (0.453) | [−2.2, +0.3] (79.0) | 1 vs 2 |
| T0e (1 seed) − T0 | pass@16 | −1.9 [−5.6, +1.2] | 0.391 | 1/4 (0.375) | [−4.6, +0.9] (79.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@1 | +0.0 [−1.4, +1.2] | 0.900 | 6/5 (1.000) | [−1.3, +1.3] (47.6) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@5 | +1.9 [−1.0, +5.2] | 0.220 | 6/5 (1.000) | [−1.3, +5.1] (79.0) | 1 vs 2 |
| T1v1 B256 (1 seed) − T0 | pass@16 | +4.4 [−1.2, +10.6] | 0.172 | 6/3 (0.508) | [−1.3, +10.0] (79.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@1 | +3.2 [+1.5, +5.5] | <0.0003 | 18/1 (7.6e-05) | [+1.3, +5.2] (74.3) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@5 | +10.6 [+6.0, +15.8] | <0.0003 | 18/1 (7.6e-05) | [+5.6, +15.6] (79.0) | 1 vs 2 |
| T1 guided B256 (1 seed) − T0 | pass@16 | +18.1 [+10.0, +26.9] | <0.0003 | 17/0 (1.5e-05) | [+10.0, +26.3] (79.0) | 1 vs 2 |
| T1 B256 − T1 B0 | pass@1 | +0.3 [−0.2, +1.0] | 0.352 | 4/0 (0.125) | [−1.1, +1.7] (2.0) | 2 vs 2 (paired) |
| T1 B256 − T1 B0 | pass@5 | +1.0 [−0.8, +3.6] | 0.393 | 4/0 (0.125) | [−6.7, +8.7] (1.5) | 2 vs 2 (paired) |
| T1 B256 − T1 B0 | pass@16 | +1.9 [−2.5, +8.8] | 0.569 | 3/0 (0.250) | [−18.5, +22.2] (1.3) | 2 vs 2 (paired) |
| T1 B256 − T0 guided | pass@1 | −4.3 [−7.6, −1.6] | <0.0003 | 0/19 (3.8e-06) | [−7.1, −1.6] (79.4) | 2 vs 2 |
| T1 B256 − T0 guided | pass@5 | −8.7 [−14.6, −3.4] | 5.0e-04 | 0/19 (3.8e-06) | [−13.9, −3.4] (31.3) | 2 vs 2 |
| T1 B256 − T0 guided | pass@16 | −12.5 [−22.5, −3.8] | 0.004 | 0/16 (3.1e-05) | [−25.1, +0.1] (4.8) | 2 vs 2 |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T0 | 2.9% [0.4, 6.3] | 3.9% [0.8, 7.9] | 5.2% [1.2, 10.1] | – |
| T1 B0 | 0.6% [0.0, 2.0] | 1.1% [0.0, 3.4] | 1.7% [0.0, 4.8] | – |
| T1 B64 | 0.5% [0.0, 1.2] | 0.9% [0.0, 2.4] | 1.6% [0.0, 4.2] | – |
| T1 B256 | 0.7% [0.0, 1.9] | 1.2% [0.0, 3.2] | 2.0% [0.0, 5.2] | 3.0% [0.0, 7.5] |
| T0 guided | 5.4% [2.1, 9.7]† | 7.7% [3.3, 13.2] | 10.4% [4.7, 17.0] | 13.5% [6.4, 22.2] |
| T0e (1 seed) | 2.2% [0.0, 5.5] | 2.9% [0.0, 6.7] | 3.7% [0.0, 8.4]† | – |
| T1v1 B256 (1 seed) | – | 2.2% [0.7, 4.1] | 3.8% [1.3, 7.0] | 6.3% [2.4, 11.0] |
| T1 guided B256 (1 seed) | – | – | 6.9% [3.6, 10.9] | 11.0% [6.0, 16.5] |

| difference (pp) | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T1 B0 − T0 | −2.3 [−5.0, −0.4] p 0.010 | −2.8 [−5.7, −0.5] p 0.010 | −3.5 [−7.4, −0.6] p 0.010 | – |
| T1 B64 − T0 | −2.4 [−5.6, −0.2] p 0.028 | −3.0 [−6.4, −0.3] p 0.033 | −3.6 [−7.7, −0.0] p 0.045 | – |
| T1 B256 − T0 | −2.2 [−5.2, −0.2] p 0.022 | −2.7 [−5.9, −0.2] p 0.025 | −3.2 [−7.3, +0.0] p 0.057 | – |
| T0 guided − T0 | +2.5 [+0.1, +5.4]† p 0.040 | +3.8 [+0.7, +7.5] p 0.020 | +5.2 [+0.4, +10.5] p 0.032 | – |
| T0e (1 seed) − T0 | −0.7 [−2.0, +0.7] p 0.312 | −1.0 [−2.9, +0.6] p 0.214 | −1.6 [−4.9, +1.2]† p 0.346 | – |
| T1v1 B256 (1 seed) − T0 | – | −1.7 [−4.7, +1.0] p 0.246 | −1.4 [−5.2, +2.2] p 0.472 | – |
| T1 guided B256 (1 seed) − T0 | – | – | +1.7 [−1.5, +5.0] p 0.270 | – |
| T1 B256 − T1 B0 | +0.1 [−0.9, +0.9] p 0.898 | +0.1 [−1.5, +1.6] p 0.895 | +0.3 [−2.4, +2.9] p 0.815 | – |
| T1 B256 − T0 guided | −4.7 [−8.4, −1.8]† p <0.0003 | −6.5 [−11.2, −2.6] p <0.0003 | −8.4 [−14.4, −3.2] p 5.0e-04 | −10.5 [−18.8, −3.7] p 0.003 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 55 | 0 | 25 |
| pass@5 | 55 | 0 | 25 |
| pass@16 | 55 | 1 | 24 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 80) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T1 B0 − T0 | pass@1 | 0.1 / 0.3 | 5.2 | 0.7 / 0.7 | 1.9 | 2 | 31 |
| T1 B0 − T0 | pass@5 | 0.3 / 0.0 | 9.1 | 1.2 / 1.2 | 3.3 | 7 | 99 |
| T1 B0 − T0 | pass@16 | 0.0 / 0.0 | 0.0 | 1.9 / 1.5 | 4.3 | 5 | 165 |
| T1 B64 − T0 | pass@1 | 0.0 / 0.3 | 5.7 | 0.8 / 0.7 | 2.0 | 2 | 36 |
| T1 B64 − T0 | pass@5 | 0.0 / 0.0 | 10.1 | 1.4 / 1.3 | 3.7 | >10k | 121 |
| T1 B64 − T0 | pass@16 | 1.3 / 0.0 | 0.0 | 2.2 / 1.9 | 5.3 | 7 | 618 |
| T1 B256 − T0 | pass@1 | 0.3 / 0.3 | 4.5 | 0.7 / 0.7 | 1.8 | 2 | 27 |
| T1 B256 − T0 | pass@5 | 1.2 / 0.0 | 8.1 | 1.4 / 1.4 | 4.0 | 8 | 251 |
| T1 B256 − T0 | pass@16 | 3.1 / 0.0 | 0.0 | 2.6 / 2.6 | 8.0 | 15 | never |
| T0 guided − T0 | pass@1 | 0.1 | 6.6 | 1.3 / 1.1 | 3.0 | 3 | 81 |
| T0 guided − T0 | pass@5 | 1.9 | 14.2 | 2.5 / 2.4 | 6.8 | >10k | never |
| T0 guided − T0 | pass@16 | 5.5 | 16.9 | 4.5 / 4.8 | 13.3 | >10k | never |
| T0e (1 seed) − T0 | pass@1 | – / 0.3 | 3.0 | 0.5 / 0.5 | 1.4 | n/a (1 seed) | 14 |
| T0e (1 seed) − T0 | pass@5 | – / 0.0 | 2.8 | 0.7 / 0.6 | 1.7 | n/a (1 seed) | 27 |
| T0e (1 seed) − T0 | pass@16 | – / 0.0 | 0.0 | 1.8 / 1.4 | 3.9 | n/a (1 seed) | 138 |
| T1v1 B256 (1 seed) − T0 | pass@1 | – / 0.3 | 4.8 | 0.7 / 0.6 | 1.8 | n/a (1 seed) | 27 |
| T1v1 B256 (1 seed) − T0 | pass@5 | – / 0.0 | 13.4 | 1.6 / 1.6 | 4.5 | n/a (1 seed) | 178 |
| T1v1 B256 (1 seed) − T0 | pass@16 | – / 0.0 | 22.1 | 3.0 / 2.8 | 8.0 | n/a (1 seed) | 563 |
| T1 guided B256 (1 seed) − T0 | pass@1 | – / 0.3 | 8.4 | 1.0 / 1.0 | 2.8 | n/a (1 seed) | 70 |
| T1 guided B256 (1 seed) − T0 | pass@5 | – / 0.0 | 22.0 | 2.5 / 2.5 | 7.1 | n/a (1 seed) | 444 |
| T1 guided B256 (1 seed) − T0 | pass@16 | – / 0.0 | 34.4 | 4.2 / 4.1 | 11.5 | n/a (1 seed) | 1173 |
| T1 B256 − T1 B0 | pass@1 | 0.4 | 1.0 | 0.3 / 0.3 | 0.9 | 2 | 3 |
| T1 B256 − T1 B0 | pass@5 | 1.6 | 0.0 | 1.1 / 1.2 | 3.6 | 3 | never |
| T1 B256 − T1 B0 | pass@16 | 3.8 | 0.0 | 2.7 / 2.9 | 8.8 | 18 | never |
| T1 B256 − T0 guided | pass@1 | 0.3 / 0.0 | 10.2 | 1.5 / 1.4 | 3.9 | >10k | 139 |
| T1 B256 − T0 guided | pass@5 | 1.2 / 1.2 | 15.7 | 2.8 / 2.6 | 7.2 | >10k | never |
| T1 B256 − T0 guided | pass@16 | 3.1 / 4.4 | 15.6 | 4.9 / 4.8 | 13.5 | >10k | never |

