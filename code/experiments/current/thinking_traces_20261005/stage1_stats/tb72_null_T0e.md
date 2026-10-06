### Null check: T0e (never thinks) read twice, B 0 and B 256, seeds 1-2

```
python -m code.thinking.stats table --group 'T0e B0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0e_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0e_s2/rows_plain_B0.jsonl' --group 'T0e B256=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0e_s1/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0e_s2/rows_plain_B256.jsonl' --ref 'T0e B0' --k 1,5,n --tokens 1000,3000,10000 --title 'Null check: T0e (never thinks) read twice, B 0 and B 256, seeds 1-2' --out code/experiments/current/thinking_traces_20261005/stage1_stats/tb72_null_T0e.md
```

n = 72 theorems (textbook_dev, textbook_train); 4000 draws, RNG seed 0; resampled: both. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | pass@256 | tokens / attempt |
|---|---|---|---|---|---|
| T0e B0 | 2 | 12.9% [6.7, 20.3] | 20.6% [12.7, 29.3] | 40.3% [29.2, 51.4] | 116 |
| T0e B256 | 2 | 12.7% [6.5, 20.2] | 20.0% [12.1, 28.6] | 42.4% [31.9, 53.5] | 124 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T0e B256 − T0e B0 | pass@1 | −0.2 [−0.5, +0.1] | 0.159 | 16/19 (0.736) | [−0.5, +0.1] (71.0) | 2 vs 2 (paired) |
| T0e B256 − T0e B0 | pass@5 | −0.6 [−1.3, +0.1] | 0.075 | 16/22 (0.418) | [−1.2, −0.0] (71.0) | 2 vs 2 (paired) |
| T0e B256 − T0e B0 | pass@256 | +2.1 [−5.6, +8.3] | 0.521 | 8/5 (0.581) | [−2.9, +7.1] (71.0) | 2 vs 2 (paired) |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 1k | 3k | 10k |
|---|---|---|---|
| T0e B0 | 23.7% [15.4, 32.8] | 30.1% [20.5, 39.9] | 35.7% [25.2, 46.0] |
| T0e B256 | 22.6% [14.5, 31.6] | 29.4% [20.3, 39.1] | 35.9% [26.0, 46.1] |

| difference (pp) | 1k | 3k | 10k |
|---|---|---|---|
| T0e B256 − T0e B0 | −1.1 [−2.2, −0.1] p 0.039 | −0.7 [−2.6, +0.9] p 0.370 | +0.2 [−2.8, +2.6] p 0.786 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 31 | 0 | 41 |
| pass@5 | 31 | 0 | 41 |
| pass@256 | 31 | 19 | 22 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 72) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T0e B256 − T0e B0 | pass@1 | 0.0 | 0.7 | 0.1 / 0.1 | 0.4 | 2 | 2 |
| T0e B256 − T0e B0 | pass@5 | 0.0 | 0.7 | 0.4 / 0.3 | 0.8 | 2 | 6 |
| T0e B256 − T0e B0 | pass@256 | 0.0 | 0.0 | 3.1 / 2.5 | 7.1 | 12 | 399 |

