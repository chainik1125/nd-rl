### textbook72, seeds resampled only (conditional on these 72 theorems)

```
python -m code.thinking.stats table --group T0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0_s0,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0_s1,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0_s2 --group T0e=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0e_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0e_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0e_s2/rows_plain_B0.jsonl --group 'T1 B0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1_s2/rows_plain_B0.jsonl' --group 'T1 B256=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1_s0/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1_s1/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1_s2/rows_plain_B256.jsonl' --group 'T1v1 B0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s2/rows_plain_B0.jsonl' --group 'T1v1 B256=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s0/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s1/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s2/rows_plain_B256.jsonl' --pair 'T1 B256 vs T1 B0' --pair 'T1v1 B256 vs T1v1 B0' --pair 'T1v1 B256 vs T1 B256' --k 1,5,n --levels seeds --title 'textbook72, seeds resampled only (conditional on these 72 theorems)' --out code/experiments/current/thinking_traces_20261005/stage1_stats/tb72_seeds_only.md
```

n = 72 theorems (textbook_dev, textbook_train); 4000 draws, RNG seed 0; resampled: seeds. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | pass@256 | tokens / attempt |
|---|---|---|---|---|---|
| T0 | 3 | 17.2% [15.4, 18.6] | 26.0% [25.2, 27.0] | 45.4% [43.1, 47.2] | 116 |
| T0e | 3 | 13.6% [10.9, 15.0] | 20.7% [18.7, 22.5] | 39.4% [37.5, 43.1] | 113 |
| T1 B0 | 3 | 13.1% [11.7, 15.4] | 20.7% [17.1, 25.0] | 38.9% [36.1, 41.7] | 129 |
| T1 B256 | 3 | 15.6% [14.5, 17.1] | 27.9% [24.7, 31.1] | 51.4% [47.2, 54.2] | 320 |
| T1v1 B0 | 3 | 13.1% [12.6, 13.5] | 20.2% [19.3, 21.5] | 36.1% [34.7, 38.9] | 125 |
| T1v1 B256 | 3 | 16.8% [14.9, 19.1] | 29.3% [27.3, 33.3] | 56.0% [51.4, 59.7] | 420 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T0e − T0 | pass@1 | −3.6 [−6.3, −1.2] | <0.0003 | 13/26 (0.053) | [−8.3, +1.1] (3.6) | 3 vs 3 |
| T0e − T0 | pass@5 | −5.3 [−7.3, −3.4] | <0.0003 | 14/25 (0.108) | [−9.3, −1.3] (2.9) | 3 vs 3 |
| T0e − T0 | pass@256 | −6.0 [−9.3, −2.3] | 0.002 | 6/13 (0.167) | [−12.6, +0.5] (3.5) | 3 vs 3 |
| T1 B0 − T0 | pass@1 | −4.1 [−6.4, −1.8] | <0.0003 | 9/30 (0.001) | [−8.2, +0.1] (3.8) | 3 vs 3 |
| T1 B0 − T0 | pass@5 | −5.2 [−8.8, −1.1] | <0.0003 | 11/28 (0.009) | [−14.5, +4.1] (2.2) | 3 vs 3 |
| T1 B0 − T0 | pass@256 | −6.5 [−9.7, −3.2] | <0.0003 | 2/12 (0.013) | [−12.2, −0.7] (3.7) | 3 vs 3 |
| T1 B256 − T0 | pass@1 | −1.6 [−3.5, +0.3] | 0.141 | 18/23 (0.533) | [−5.0, +1.8] (3.9) | 3 vs 3 |
| T1 B256 − T0 | pass@5 | +1.9 [−1.2, +4.9] | 0.257 | 19/23 (0.644) | [−5.2, +9.1] (2.4) | 3 vs 3 |
| T1 B256 − T0 | pass@256 | +6.0 [+1.9, +9.7] | 0.002 | 12/4 (0.077) | [−1.5, +13.5] (3.2) | 3 vs 3 |
| T1v1 B0 − T0 | pass@1 | −4.1 [−5.4, −2.6] | <0.0003 | 7/31 (1.2e-04) | [−7.7, −0.5] (2.4) | 3 vs 3 |
| T1v1 B0 − T0 | pass@5 | −5.8 [−7.1, −4.5] | <0.0003 | 8/31 (2.9e-04) | [−8.2, −3.4] (3.9) | 3 vs 3 |
| T1v1 B0 − T0 | pass@256 | −9.3 [−12.0, −6.5] | <0.0003 | 2/15 (0.002) | [−14.4, −4.1] (3.9) | 3 vs 3 |
| T1v1 B256 − T0 | pass@1 | −0.4 [−2.8, +2.0] | 0.731 | 26/22 (0.665) | [−4.8, +4.0] (3.7) | 3 vs 3 |
| T1v1 B256 − T0 | pass@5 | +3.4 [+0.8, +7.2] | <0.0003 | 31/17 (0.059) | [−4.5, +11.3] (2.3) | 3 vs 3 |
| T1v1 B256 − T0 | pass@256 | +10.6 [+6.0, +14.8] | <0.0003 | 21/1 (1.1e-05) | [+1.8, +19.5] (2.9) | 3 vs 3 |
| T1 B256 − T1 B0 | pass@1 | +2.5 [+1.8, +2.9] | <0.0003 | 28/13 (0.028) | [+0.9, +4.1] (2.0) | 3 vs 3 (paired) |
| T1 B256 − T1 B0 | pass@5 | +7.1 [+6.0, +7.8] | <0.0003 | 32/9 (4.3e-04) | [+4.7, +9.5] (2.0) | 3 vs 3 (paired) |
| T1 B256 − T1 B0 | pass@256 | +12.5 [+11.1, +13.9] | <0.0003 | 15/1 (5.2e-04) | [+9.0, +16.0] (2.0) | 3 vs 3 (paired) |
| T1v1 B256 − T1v1 B0 | pass@1 | +3.7 [+1.6, +6.5] | <0.0003 | 41/7 (6.2e-07) | [−2.6, +9.9] (2.0) | 3 vs 3 (paired) |
| T1v1 B256 − T1v1 B0 | pass@5 | +9.2 [+5.9, +13.6] | <0.0003 | 40/8 (3.3e-06) | [−0.6, +19.0] (2.0) | 3 vs 3 (paired) |
| T1v1 B256 − T1v1 B0 | pass@256 | +19.9 [+16.7, +25.0] | <0.0003 | 28/0 (7.5e-09) | [+8.8, +31.0] (2.0) | 3 vs 3 (paired) |
| T1v1 B256 − T1 B256 | pass@1 | +1.2 [−1.1, +3.4] | 0.316 | 33/15 (0.013) | [−3.2, +5.5] (3.4) | 3 vs 3 |
| T1v1 B256 − T1 B256 | pass@5 | +1.5 [−2.7, +5.6] | 0.513 | 32/16 (0.029) | [−6.1, +9.0] (4.0) | 3 vs 3 |
| T1v1 B256 − T1 B256 | pass@256 | +4.6 [−0.5, +9.7] | 0.117 | 13/2 (0.007) | [−4.4, +13.7] (3.9) | 3 vs 3 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 24 | 0 | 48 |
| pass@5 | 24 | 0 | 48 |
| pass@256 | 24 | 14 | 34 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 72) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T0e − T0 | pass@1 | 1.6 / 0.4 | 7.5 | 1.3 / 1.6 | 5.2 | 22 | 802 |
| T0e − T0 | pass@5 | 0.0 / 0.0 | 9.6 | 1.0 / 1.2 | 5.7 | >10k | 261 |
| T0e − T0 | pass@256 | 0.0 / 0.0 | 7.9 | 1.8 / 2.2 | 7.9 | 76 | 501 |
| T1 B0 − T0 | pass@1 | 1.6 / 0.4 | 4.2 | 1.2 / 1.5 | 4.3 | 8 | 558 |
| T1 B0 − T0 | pass@5 | 3.6 / 0.0 | 6.6 | 1.9 / 2.4 | 7.6 | 38 | never |
| T1 B0 − T0 | pass@256 | 1.8 / 0.0 | 12.8 | 1.7 / 2.0 | 7.9 | >10k | 8173 |
| T1 B256 − T0 | pass@1 | 1.1 / 0.4 | 0.9 | 1.0 / 1.2 | 3.4 | 4 | 106 |
| T1 B256 − T0 | pass@5 | 2.9 / 0.0 | 7.9 | 1.6 / 1.9 | 6.7 | 51 | never |
| T1 B256 − T0 | pass@256 | 2.8 / 0.0 | 11.2 | 2.0 / 2.4 | 8.5 | >10k | never |
| T1v1 B0 − T0 | pass@1 | 0.0 / 0.4 | 6.1 | 0.8 / 1.0 | 3.6 | 6 | 106 |
| T1v1 B0 − T0 | pass@5 | 0.0 / 0.0 | 6.5 | 0.7 / 0.9 | 4.8 | 13 | 185 |
| T1v1 B0 − T0 | pass@256 | 0.0 / 0.0 | 8.5 | 1.5 / 1.9 | 7.4 | 137 | 443 |
| T1v1 B256 − T0 | pass@1 | 1.9 / 0.4 | 1.0 | 1.3 / 1.5 | 4.3 | 7 | never |
| T1v1 B256 − T0 | pass@5 | 3.0 / 0.0 | 9.1 | 1.7 / 2.1 | 7.2 | >10k | never |
| T1v1 B256 − T0 | pass@256 | 3.1 / 0.0 | 0.0 | 2.3 / 2.7 | 8.4 | 24 | never |
| T1 B256 − T1 B0 | pass@1 | 0.0 | 5.2 | 0.3 / 0.4 | 2.2 | 2 | 38 |
| T1 B256 − T1 B0 | pass@5 | 0.0 | 11.8 | 0.5 / 0.6 | 4.7 | >10k | 179 |
| T1 B256 − T1 B0 | pass@256 | 0.0 | 24.4 | 0.7 / 0.8 | 9.5 | >10k | 722 |
| T1v1 B256 − T1v1 B0 | pass@1 | 2.3 | 6.7 | 1.2 / 1.5 | 4.6 | 13 | never |
| T1v1 B256 − T1v1 B0 | pass@5 | 3.5 | 13.4 | 1.9 / 2.3 | 7.8 | >10k | never |
| T1v1 B256 − T1v1 B0 | pass@256 | 2.1 | 22.3 | 2.1 / 2.6 | 10.3 | >10k | never |
| T1v1 B256 − T1 B256 | pass@1 | 1.9 / 1.1 | 0.0 | 1.2 / 1.5 | 4.1 | 6 | never |
| T1v1 B256 − T1 B256 | pass@5 | 3.0 / 2.9 | 5.1 | 2.2 / 2.7 | 7.8 | 29 | never |
| T1v1 B256 − T1 B256 | pass@256 | 3.1 / 2.8 | 4.5 | 2.6 / 3.2 | 9.2 | 37 | never |

