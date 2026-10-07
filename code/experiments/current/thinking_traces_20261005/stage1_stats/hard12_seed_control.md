### hard12_test_1k (sealed, 1,000 items, 7-12 lines), k 32

```
python -m code.thinking.stats table --group 'T0 s1-2=/scratch/cowifornia/nd-rl/think/runs/eval/h12_T0_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/h12_T0_s2/rows_plain_B0.jsonl' --group 'T0 s0 (filter model)=/scratch/cowifornia/nd-rl/think/runs/eval/h12_T0_s0/rows_plain_B0.jsonl' --group 'T1v1 B0=/scratch/cowifornia/nd-rl/think/runs/eval/h12_T1v1_s0/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/h12_T1v1_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/h12_T1v1_s2/rows_plain_B0.jsonl' --group 'T1v1 B256=/scratch/cowifornia/nd-rl/think/runs/eval/h12_T1v1_s0/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/h12_T1v1_s1/rows_plain_B256.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/h12_T1v1_s2/rows_plain_B256.jsonl' --k 1,5 --tokens 1000,3000 --pair 'T1v1 B256 vs T1v1 B0' --pair 'T1v1 B256 vs T0 s1-2' --pair 'T1v1 B0 vs T0 s1-2' --pair 'T0 s0 (filter model) vs T0 s1-2' --title 'hard12_test_1k (sealed, 1,000 items, 7-12 lines), k 32' --out code/experiments/current/thinking_traces_20261005/stage1_stats/hard12_seed_control.md
```

n = 1000 theorems (hard12_test_1k); 4000 draws, RNG seed 0; resampled: both. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | tokens / attempt |
|---|---|---|---|---|
| T0 s1-2 | 2 | 42.4% [40.7, 44.2] | 75.2% [73.0, 77.3] | 167 |
| T0 s0 (filter model) | 1 | 27.8% [26.7, 28.8] | 71.6% [69.9, 73.2] | 176 |
| T1v1 B0 | 3 | 30.2% [27.8, 32.4] | 61.7% [58.9, 64.5] | 173 |
| T1v1 B256 | 3 | 42.6% [39.5, 45.5] | 80.8% [78.4, 83.2] | 485 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T0 s0 (filter model) − T0 s1-2 | pass@1 | −14.7 [−16.4, −13.0] | <0.0003 | 269/706 (7.6e-46) | [−16.2, −13.2] (999.0) | 1 vs 2 |
| T0 s0 (filter model) − T0 s1-2 | pass@5 | −3.6 [−6.0, −1.2] | 0.004 | 395/604 (4.0e-11) | [−5.9, −1.4] (32.2) | 1 vs 2 |
| T1v1 B0 − T0 s1-2 | pass@1 | −12.3 [−14.8, −9.9] | <0.0003 | 286/700 (1.1e-40) | [−15.8, −8.7] (4.2) | 3 vs 2 |
| T1v1 B0 − T0 s1-2 | pass@5 | −13.5 [−16.4, −10.4] | <0.0003 | 287/708 (1.1e-41) | [−17.1, −9.9] (6.3) | 3 vs 2 |
| T1v1 B256 − T0 s1-2 | pass@1 | +0.2 [−3.0, +3.2] | 0.901 | 496/493 (0.949) | [−5.7, +6.1] (2.8) | 3 vs 2 |
| T1v1 B256 − T0 s1-2 | pass@5 | +5.6 [+2.9, +8.5] | <0.0003 | 520/479 (0.206) | [+2.0, +9.2] (5.5) | 3 vs 2 |
| T1v1 B256 − T1v1 B0 | pass@1 | +12.5 [+8.4, +15.6] | <0.0003 | 808/166 (7.2e-102) | [+3.7, +21.2] (2.2) | 3 vs 3 (paired) |
| T1v1 B256 − T1v1 B0 | pass@5 | +19.1 [+15.1, +22.5] | <0.0003 | 841/154 (4.1e-115) | [+11.1, +27.2] (2.4) | 3 vs 3 (paired) |
| T1v1 B256 − T0 s1-2 | pass@1 | +0.2 [−3.0, +3.2] | 0.901 | 496/493 (0.949) | [−5.7, +6.1] (2.8) | 3 vs 2 |
| T1v1 B256 − T0 s1-2 | pass@5 | +5.6 [+2.9, +8.5] | <0.0003 | 520/479 (0.206) | [+2.0, +9.2] (5.5) | 3 vs 2 |
| T1v1 B0 − T0 s1-2 | pass@1 | −12.3 [−14.8, −9.9] | <0.0003 | 286/700 (1.1e-40) | [−15.8, −8.7] (4.2) | 3 vs 2 |
| T1v1 B0 − T0 s1-2 | pass@5 | −13.5 [−16.4, −10.4] | <0.0003 | 287/708 (1.1e-41) | [−17.1, −9.9] (6.3) | 3 vs 2 |
| T0 s0 (filter model) − T0 s1-2 | pass@1 | −14.7 [−16.4, −13.0] | <0.0003 | 269/706 (7.6e-46) | [−16.2, −13.2] (999.0) | 1 vs 2 |
| T0 s0 (filter model) − T0 s1-2 | pass@5 | −3.6 [−6.0, −1.2] | 0.004 | 395/604 (4.0e-11) | [−5.9, −1.4] (32.2) | 1 vs 2 |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 1k | 3k |
|---|---|---|
| T0 s1-2 | 77.7% [75.4, 80.0] | 88.9% [87.0, 90.6] |
| T0 s0 (filter model) | 74.7% [73.0, 76.4] | 92.9% [91.7, 94.1] |
| T1v1 B0 | 64.0% [60.6, 67.3] | 78.2% [75.8, 80.8] |
| T1v1 B256 | 62.6% [56.9, 68.0] | 84.3% [80.9, 87.5] |

| difference (pp) | 1k | 3k |
|---|---|---|
| T0 s0 (filter model) − T0 s1-2 | −3.1 [−5.4, −0.5] p 0.020 | +4.1 [+2.1, +6.2] p <0.0003 |
| T1v1 B0 − T0 s1-2 | −13.7 [−17.3, −10.2] p <0.0003 | −10.6 [−13.3, −7.7] p <0.0003 |
| T1v1 B256 − T0 s1-2 | −15.2 [−21.0, −9.6] p <0.0003 | −4.5 [−8.1, −1.0] p 0.009 |
| T1v1 B256 − T1v1 B0 | −1.4 [−4.6, +1.7] p 0.402 | +6.1 [+3.9, +8.1] p <0.0003 |
| T1v1 B256 − T0 s1-2 | −15.2 [−21.0, −9.6] p <0.0003 | −4.5 [−8.1, −1.0] p 0.009 |
| T1v1 B0 − T0 s1-2 | −13.7 [−17.3, −10.2] p <0.0003 | −10.6 [−13.3, −7.7] p <0.0003 |
| T0 s0 (filter model) − T0 s1-2 | −3.1 [−5.4, −0.5] p 0.020 | +4.1 [+2.1, +6.2] p <0.0003 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 0 | 0 | 1000 |
| pass@5 | 0 | 0 | 1000 |
| pass@32 | 0 | 596 | 404 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 1000) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T0 s0 (filter model) − T0 s1-2 | pass@1 | – / 0.0 | 18.8 | 0.9 / 0.8 | 2.2 | n/a (1 seed) | 528 |
| T0 s0 (filter model) − T0 s1-2 | pass@5 | – / 0.6 | 25.0 | 1.2 / 1.1 | 3.0 | n/a (1 seed) | 1040 |
| T1v1 B0 − T0 s1-2 | pass@1 | 1.9 / 0.0 | 12.0 | 1.3 / 1.3 | 3.6 | 5 | never |
| T1v1 B0 − T0 s1-2 | pass@5 | 1.9 / 0.6 | 14.1 | 1.5 / 1.5 | 4.2 | 6 | never |
| T1v1 B256 − T0 s1-2 | pass@1 | 2.8 / 0.0 | 10.5 | 1.6 / 1.8 | 5.0 | 9 | never |
| T1v1 B256 − T0 s1-2 | pass@5 | 1.9 / 0.6 | 11.9 | 1.4 / 1.4 | 4.0 | 6 | never |
| T1v1 B256 − T1v1 B0 | pass@1 | 3.7 | 9.9 | 1.9 / 2.2 | 6.2 | 14 | never |
| T1v1 B256 − T1v1 B0 | pass@5 | 3.7 | 15.4 | 1.9 / 2.2 | 6.2 | 16 | never |
| T1v1 B256 − T0 s1-2 | pass@1 | 2.8 / 0.0 | 10.5 | 1.6 / 1.8 | 5.0 | 9 | never |
| T1v1 B256 − T0 s1-2 | pass@5 | 1.9 / 0.6 | 11.9 | 1.4 / 1.4 | 4.0 | 6 | never |
| T1v1 B0 − T0 s1-2 | pass@1 | 1.9 / 0.0 | 12.0 | 1.3 / 1.3 | 3.6 | 5 | never |
| T1v1 B0 − T0 s1-2 | pass@5 | 1.9 / 0.6 | 14.1 | 1.5 / 1.5 | 4.2 | 6 | never |
| T0 s0 (filter model) − T0 s1-2 | pass@1 | – / 0.0 | 18.8 | 0.9 / 0.8 | 2.2 | n/a (1 seed) | 528 |
| T0 s0 (filter model) − T0 s1-2 | pass@5 | – / 0.6 | 25.0 | 1.2 / 1.1 | 3.0 | n/a (1 seed) | 1040 |

