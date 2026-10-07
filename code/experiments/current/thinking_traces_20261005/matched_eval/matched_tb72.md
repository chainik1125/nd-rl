### matched tb72

```
python -m code.thinking.stats table --group T0_stage1_s0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T0_s0/rows_plain_B0.jsonl --group T1v1_stage1_B0=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s0/rows_plain_B0.jsonl --group T1v1_stage1_B256=/scratch/cowifornia/nd-rl/think/runs/eval/tb72_T1v1_s0/rows_plain_B256.jsonl --group EI_T0_r1=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_eiT0_r1/rows_plain_B0.jsonl --group GRPO_T0_it39=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_grpoT0_it39/rows_plain_B0.jsonl --group EI_T1v1e_r1_B0=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_eiT1v1e_r1/rows_plain_B0.jsonl --group GRPO_T1v1e_it39_B0=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_grpoT1v1e_it39/rows_plain_B0.jsonl --group EI_T1v1e_r1_B64=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_eiT1v1e_r1/rows_plain_B64.jsonl --group GRPO_T1v1e_it39_B64=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_grpoT1v1e_it39/rows_plain_B64.jsonl --group EI_T1v1e_r1_B256=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_eiT1v1e_r1/rows_plain_B256.jsonl --group EI_T0_r2=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_eiT0_r2/rows_plain_B0.jsonl --group GRPO_T0_it78=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_grpoT0_it78/rows_plain_B0.jsonl --group EI_T1v1e_r2_B0=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_eiT1v1e_r2/rows_plain_B0.jsonl --group EI_T1v1e_r2_B64=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_eiT1v1e_r2/rows_plain_B64.jsonl --group EI_T1v1e_r2_B256=/scratch/cowifornia/nd-rl/think/runs/eval/matched/tb72_eiT1v1e_r2/rows_plain_B256.jsonl --pair 'GRPO_T0_it39 vs EI_T0_r1' --pair 'EI_T1v1e_r1_B256 vs EI_T0_r1' --pair 'EI_T1v1e_r1_B256 vs EI_T1v1e_r1_B0' --pair 'GRPO_T0_it78 vs EI_T0_r2' --pair 'EI_T1v1e_r2_B256 vs EI_T0_r2' --pair 'EI_T1v1e_r2_B256 vs EI_T1v1e_r2_B0' --k 1,5 --tokens 1000,3000 --title 'matched tb72' --out code/experiments/current/thinking_traces_20261005/matched_eval/matched_tb72.md
```

n = 72 theorems (textbook_dev, textbook_train); 4000 draws, RNG seed 0; resampled: both. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | tokens / attempt |
|---|---|---|---|---|
| T0_stage1_s0 | 1 | 18.6% [11.3, 27.0] | 27.0% [17.8, 37.0] | 116 |
| T1v1_stage1_B0 | 1 | 12.6% [6.7, 19.4] | 19.8% [12.2, 28.4] | 134 |
| T1v1_stage1_B256 | 1 | 19.1% [12.4, 26.4] | 33.3% [24.2, 42.4] | 508 |
| EI_T0_r1 | 1 | 30.8% [21.7, 40.3] | 41.8% [31.3, 51.9] | 162 |
| GRPO_T0_it39 | 1 | 32.5% [22.4, 42.7] | 37.1% [26.5, 47.8] | 141 |
| EI_T1v1e_r1_B0 | 1 | 16.3% [9.8, 23.9] | 26.5% [17.7, 36.0] | 130 |
| GRPO_T1v1e_it39_B0 | 1 | 21.6% [13.5, 30.7] | 28.0% [18.6, 38.0] | 127 |
| EI_T1v1e_r1_B64 | 1 | 23.4% [16.3, 31.0] | 39.8% [29.9, 49.4] | 425 |
| GRPO_T1v1e_it39_B64 | 1 | 32.9% [23.0, 43.0] | 38.9% [28.4, 49.3] | 376 |
| EI_T1v1e_r1_B256 | 1 | 28.7% [20.6, 37.1] | 45.4% [35.2, 55.1] | 777 |
| EI_T0_r2 | 1 | 32.4% [23.4, 41.9] | 44.8% [34.7, 55.0] | 181 |
| GRPO_T0_it78 | 1 | 32.6% [23.1, 42.4] | 40.9% [30.3, 51.7] | 153 |
| EI_T1v1e_r2_B0 | 1 | 21.3% [13.6, 30.0] | 31.0% [21.5, 40.3] | 132 |
| EI_T1v1e_r2_B64 | 1 | 27.8% [20.0, 35.9] | 45.0% [34.8, 54.9] | 452 |
| EI_T1v1e_r2_B256 | 1 | 31.6% [23.2, 40.3] | 48.8% [38.4, 58.4] | 905 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T1v1_stage1_B0 − T0_stage1_s0 | pass@1 | −6.0 [−10.2, −2.6] | <0.0003 | 6/27 (3.2e-04) | [−9.8, −2.2] (71.0) | 1 vs 1 |
| T1v1_stage1_B0 − T0_stage1_s0 | pass@5 | −7.2 [−13.2, −2.0] | 0.004 | 6/27 (3.2e-04) | [−13.0, −1.5] (71.0) | 1 vs 1 |
| T1v1_stage1_B256 − T0_stage1_s0 | pass@1 | +0.5 [−3.3, +4.1] | 0.795 | 27/16 (0.126) | [−3.2, +4.2] (71.0) | 1 vs 1 |
| T1v1_stage1_B256 − T0_stage1_s0 | pass@5 | +6.3 [−0.1, +12.8] | 0.058 | 27/16 (0.126) | [−0.2, +12.8] (71.0) | 1 vs 1 |
| EI_T0_r1 − T0_stage1_s0 | pass@1 | +12.2 [+7.4, +17.5] | <0.0003 | 38/3 (1.0e-08) | [+6.9, +17.5] (71.0) | 1 vs 1 |
| EI_T0_r1 − T0_stage1_s0 | pass@5 | +14.8 [+8.9, +20.9] | <0.0003 | 39/2 (7.8e-10) | [+8.4, +21.2] (71.0) | 1 vs 1 |
| GRPO_T0_it39 − T0_stage1_s0 | pass@1 | +14.0 [+6.5, +21.7] | <0.0003 | 26/12 (0.034) | [+6.1, +21.8] (71.0) | 1 vs 1 |
| GRPO_T0_it39 − T0_stage1_s0 | pass@5 | +10.1 [+1.1, +19.0] | 0.027 | 27/11 (0.014) | [+1.0, +19.3] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B0 − T0_stage1_s0 | pass@1 | −2.2 [−6.7, +1.5] | 0.272 | 16/19 (0.736) | [−6.3, +1.9] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B0 − T0_stage1_s0 | pass@5 | −0.5 [−6.5, +5.4] | 0.874 | 18/17 (1.000) | [−6.4, +5.5] (71.0) | 1 vs 1 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s0 | pass@1 | +3.0 [−2.3, +8.4] | 0.274 | 23/14 (0.188) | [−2.5, +8.5] (71.0) | 1 vs 1 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s0 | pass@5 | +1.0 [−6.6, +8.4] | 0.781 | 24/13 (0.099) | [−6.7, +8.7] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B64 − T0_stage1_s0 | pass@1 | +4.8 [−0.4, +10.1] | 0.068 | 32/10 (9.4e-04) | [−0.6, +10.3] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B64 − T0_stage1_s0 | pass@5 | +12.8 [+5.9, +19.9] | <0.0003 | 32/10 (9.4e-04) | [+5.4, +20.3] (71.0) | 1 vs 1 |
| GRPO_T1v1e_it39_B64 − T0_stage1_s0 | pass@1 | +14.4 [+6.6, +22.5] | <0.0003 | 32/13 (0.007) | [+6.1, +22.7] (71.0) | 1 vs 1 |
| GRPO_T1v1e_it39_B64 − T0_stage1_s0 | pass@5 | +11.9 [+2.7, +21.0] | 0.008 | 34/11 (8.2e-04) | [+2.3, +21.5] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B256 − T0_stage1_s0 | pass@1 | +10.2 [+4.7, +15.7] | <0.0003 | 41/8 (2.0e-06) | [+4.4, +15.9] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B256 − T0_stage1_s0 | pass@5 | +18.3 [+10.5, +26.5] | <0.0003 | 43/6 (5.7e-08) | [+10.1, +26.6] (71.0) | 1 vs 1 |
| EI_T0_r2 − T0_stage1_s0 | pass@1 | +13.8 [+8.7, +19.5] | <0.0003 | 43/4 (2.8e-09) | [+8.2, +19.5] (71.0) | 1 vs 1 |
| EI_T0_r2 − T0_stage1_s0 | pass@5 | +17.8 [+11.4, +24.5] | <0.0003 | 43/4 (2.8e-09) | [+10.9, +24.6] (71.0) | 1 vs 1 |
| GRPO_T0_it78 − T0_stage1_s0 | pass@1 | +14.0 [+7.7, +20.5] | <0.0003 | 31/8 (2.9e-04) | [+7.4, +20.7] (71.0) | 1 vs 1 |
| GRPO_T0_it78 − T0_stage1_s0 | pass@5 | +13.9 [+5.8, +21.8] | <0.0003 | 31/8 (2.9e-04) | [+5.6, +22.3] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B0 − T0_stage1_s0 | pass@1 | +2.7 [−1.9, +7.1] | 0.240 | 24/11 (0.041) | [−1.9, +7.3] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B0 − T0_stage1_s0 | pass@5 | +3.9 [−2.6, +10.1] | 0.227 | 24/11 (0.041) | [−2.5, +10.4] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B64 − T0_stage1_s0 | pass@1 | +9.3 [+3.8, +15.0] | <0.0003 | 37/7 (5.3e-06) | [+3.4, +15.1] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B64 − T0_stage1_s0 | pass@5 | +18.0 [+10.6, +25.6] | <0.0003 | 37/7 (5.3e-06) | [+10.1, +25.9] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B256 − T0_stage1_s0 | pass@1 | +13.0 [+7.1, +19.0] | <0.0003 | 42/7 (3.6e-07) | [+6.9, +19.2] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B256 − T0_stage1_s0 | pass@5 | +21.8 [+13.1, +30.4] | <0.0003 | 43/6 (5.7e-08) | [+12.9, +30.7] (71.0) | 1 vs 1 |
| GRPO_T0_it39 − EI_T0_r1 | pass@1 | +1.8 [−4.9, +8.0] | 0.588 | 22/17 (0.522) | [−4.6, +8.2] (71.0) | 1 vs 1 |
| GRPO_T0_it39 − EI_T0_r1 | pass@5 | −4.7 [−11.9, +1.9] | 0.168 | 15/17 (0.860) | [−11.6, +2.3] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | pass@1 | −2.1 [−7.5, +3.3] | 0.443 | 23/23 (1.000) | [−7.5, +3.4] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | pass@5 | +3.6 [−2.7, +9.8] | 0.282 | 20/21 (1.000) | [−2.9, +10.0] (71.0) | 1 vs 1 |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | pass@1 | +12.4 [+7.2, +17.9] | <0.0003 | 41/8 (2.0e-06) | [+6.7, +18.1] (71.0) | 1 vs 1 (paired) |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | pass@5 | +18.8 [+11.4, +26.4] | <0.0003 | 37/7 (5.3e-06) | [+11.0, +26.7] (71.0) | 1 vs 1 (paired) |
| GRPO_T0_it78 − EI_T0_r2 | pass@1 | +0.2 [−4.3, +4.4] | 0.925 | 21/22 (1.000) | [−4.3, +4.6] (71.0) | 1 vs 1 |
| GRPO_T0_it78 − EI_T0_r2 | pass@5 | −3.8 [−10.9, +2.9] | 0.272 | 16/21 (0.511) | [−10.9, +3.2] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | pass@1 | −0.8 [−7.1, +5.3] | 0.787 | 26/23 (0.775) | [−7.1, +5.4] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | pass@5 | +4.0 [−2.9, +11.1] | 0.277 | 24/20 (0.652) | [−3.1, +11.2] (71.0) | 1 vs 1 |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | pass@1 | +10.3 [+5.0, +15.9] | <0.0003 | 37/8 (1.5e-05) | [+4.6, +16.1] (71.0) | 1 vs 1 (paired) |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | pass@5 | +17.9 [+10.6, +25.7] | <0.0003 | 35/7 (1.5e-05) | [+10.0, +25.7] (71.0) | 1 vs 1 (paired) |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 1k | 3k |
|---|---|---|
| T0_stage1_s0 | 29.5% [19.8, 40.0] | 33.7% [23.4, 44.4] |
| T1v1_stage1_B0 | 21.6% [13.6, 30.8] | 27.0% [17.7, 37.2] |
| T1v1_stage1_B256 | 25.0% [16.7, 33.9] | 34.8% [25.2, 44.5] |
| EI_T0_r1 | 43.1% [32.2, 53.7] | 50.0% [38.9, 60.7] |
| GRPO_T0_it39 | 37.7% [26.9, 48.7] | 39.5% [28.3, 50.5] |
| EI_T1v1e_r1_B0 | 29.0% [19.5, 39.0] | 34.9% [24.7, 45.4] |
| GRPO_T1v1e_it39_B0 | 29.5% [20.0, 39.5] | 33.8% [23.8, 44.1] |
| EI_T1v1e_r1_B64 | 32.6% [23.2, 42.3] | 42.7% [32.3, 52.8] |
| GRPO_T1v1e_it39_B64 | 36.6% [25.9, 47.3] | 41.0% [30.0, 51.7] |
| EI_T1v1e_r1_B256 | 31.5% [22.3, 41.3] | 42.9% [32.3, 53.2] |
| EI_T0_r2 | 45.5% [34.9, 56.0] | 53.3% [42.7, 63.9] |
| GRPO_T0_it78 | 41.7% [30.9, 52.6] | 44.0% [32.8, 54.9] |
| EI_T1v1e_r2_B0 | 33.2% [23.4, 43.3] | 39.4% [28.8, 50.0] |
| EI_T1v1e_r2_B64 | 37.0% [27.2, 47.0] | 47.4% [36.6, 57.8] |
| EI_T1v1e_r2_B256 | 32.8% [23.5, 42.7] | 44.9% [34.1, 55.2] |

| difference (pp) | 1k | 3k |
|---|---|---|
| T1v1_stage1_B0 − T0_stage1_s0 | −7.9 [−14.4, −1.9] p 0.006 | −6.6 [−13.5, −0.1] p 0.046 |
| T1v1_stage1_B256 − T0_stage1_s0 | −4.4 [−10.9, +1.4] p 0.141 | +1.2 [−5.6, +7.9] p 0.779 |
| EI_T0_r1 − T0_stage1_s0 | +13.6 [+7.6, +19.8] p <0.0003 | +16.3 [+9.5, +23.6] p <0.0003 |
| GRPO_T0_it39 − T0_stage1_s0 | +8.3 [−0.9, +17.2] p 0.082 | +5.9 [−3.4, +15.1] p 0.232 |
| EI_T1v1e_r1_B0 − T0_stage1_s0 | −0.5 [−6.8, +5.8] p 0.878 | +1.3 [−4.5, +7.2] p 0.683 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s0 | +0.0 [−7.9, +7.6] p 0.992 | +0.1 [−8.3, +8.3] p 0.977 |
| EI_T1v1e_r1_B64 − T0_stage1_s0 | +3.1 [−3.5, +9.9] p 0.402 | +9.1 [+2.5, +16.4] p 0.007 |
| GRPO_T1v1e_it39_B64 − T0_stage1_s0 | +7.1 [−2.4, +16.4] p 0.138 | +7.3 [−2.6, +17.1] p 0.151 |
| EI_T1v1e_r1_B256 − T0_stage1_s0 | +2.0 [−5.0, +8.6] p 0.589 | +9.3 [+1.6, +17.1] p 0.011 |
| EI_T0_r2 − T0_stage1_s0 | +16.0 [+9.7, +22.7] p <0.0003 | +19.6 [+12.4, +27.1] p <0.0003 |
| GRPO_T0_it78 − T0_stage1_s0 | +12.3 [+4.0, +20.4] p 0.001 | +10.3 [+2.0, +18.8] p 0.015 |
| EI_T1v1e_r2_B0 − T0_stage1_s0 | +3.8 [−3.0, +10.1] p 0.270 | +5.8 [−0.4, +11.9] p 0.066 |
| EI_T1v1e_r2_B64 − T0_stage1_s0 | +7.5 [+0.8, +14.4] p 0.028 | +13.7 [+6.9, +21.2] p <0.0003 |
| EI_T1v1e_r2_B256 − T0_stage1_s0 | +3.3 [−3.8, +10.1] p 0.385 | +11.2 [+3.4, +19.2] p 0.004 |
| GRPO_T0_it39 − EI_T0_r1 | −5.3 [−12.6, +1.1] p 0.115 | −10.5 [−18.2, −3.6] p 0.002 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | −11.6 [−18.5, −5.0] p <0.0003 | −7.0 [−14.0, −0.2] p 0.041 |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | +2.5 [−4.2, +9.5] p 0.486 | +8.0 [+0.8, +15.5] p 0.026 |
| GRPO_T0_it78 − EI_T0_r2 | −3.7 [−11.1, +3.3] p 0.308 | −9.3 [−17.5, −1.0] p 0.029 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | −12.7 [−20.3, −5.4] p 0.002 | −8.4 [−16.1, −0.7] p 0.030 |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | −0.5 [−7.4, +6.8] p 0.902 | +5.5 [−2.2, +13.5] p 0.168 |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 16 | 0 | 56 |
| pass@5 | 16 | 0 | 56 |
| pass@64 | 16 | 15 | 41 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 72) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T1v1_stage1_B0 − T0_stage1_s0 | pass@1 | – / – | 16.1 | 1.9 / 1.9 | 5.3 | n/a (1 seed) | 228 |
| T1v1_stage1_B0 − T0_stage1_s0 | pass@5 | – / – | 24.5 | 2.9 / 2.9 | 8.1 | n/a (1 seed) | 524 |
| T1v1_stage1_B256 − T0_stage1_s0 | pass@1 | – / – | 15.8 | 1.9 / 1.9 | 5.2 | n/a (1 seed) | 219 |
| T1v1_stage1_B256 − T0_stage1_s0 | pass@5 | – / – | 27.7 | 3.3 / 3.3 | 9.2 | n/a (1 seed) | 671 |
| EI_T0_r1 − T0_stage1_s0 | pass@1 | – / – | 22.7 | 2.6 / 2.7 | 7.5 | n/a (1 seed) | 450 |
| EI_T0_r1 − T0_stage1_s0 | pass@5 | – / – | 27.3 | 3.2 / 3.2 | 9.0 | n/a (1 seed) | 651 |
| GRPO_T0_it39 − T0_stage1_s0 | pass@1 | – / – | 33.4 | 3.9 / 3.9 | 11.0 | n/a (1 seed) | 975 |
| GRPO_T0_it39 − T0_stage1_s0 | pass@5 | – / – | 39.0 | 4.6 / 4.6 | 12.9 | n/a (1 seed) | 1327 |
| EI_T1v1e_r1_B0 − T0_stage1_s0 | pass@1 | – / – | 17.5 | 2.1 / 2.1 | 5.8 | n/a (1 seed) | 268 |
| EI_T1v1e_r1_B0 − T0_stage1_s0 | pass@5 | – / – | 25.3 | 3.0 / 3.0 | 8.4 | n/a (1 seed) | 561 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s0 | pass@1 | – / – | 23.5 | 2.7 / 2.8 | 7.8 | n/a (1 seed) | 481 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s0 | pass@5 | – / – | 32.8 | 3.8 / 3.9 | 10.8 | n/a (1 seed) | 940 |
| EI_T1v1e_r1_B64 − T0_stage1_s0 | pass@1 | – / – | 23.2 | 2.7 / 2.7 | 7.7 | n/a (1 seed) | 470 |
| EI_T1v1e_r1_B64 − T0_stage1_s0 | pass@5 | – / – | 31.7 | 3.7 / 3.7 | 10.5 | n/a (1 seed) | 877 |
| GRPO_T1v1e_it39_B64 − T0_stage1_s0 | pass@1 | – / – | 35.3 | 4.0 / 4.2 | 11.7 | n/a (1 seed) | 1090 |
| GRPO_T1v1e_it39_B64 − T0_stage1_s0 | pass@5 | – / – | 40.8 | 4.7 / 4.8 | 13.5 | n/a (1 seed) | 1454 |
| EI_T1v1e_r1_B256 − T0_stage1_s0 | pass@1 | – / – | 24.5 | 2.8 / 2.9 | 8.1 | n/a (1 seed) | 524 |
| EI_T1v1e_r1_B256 − T0_stage1_s0 | pass@5 | – / – | 35.2 | 4.1 / 4.2 | 11.6 | n/a (1 seed) | 1083 |
| EI_T0_r2 − T0_stage1_s0 | pass@1 | – / – | 24.0 | 2.8 / 2.8 | 7.9 | n/a (1 seed) | 501 |
| EI_T0_r2 − T0_stage1_s0 | pass@5 | – / – | 29.3 | 3.4 / 3.5 | 9.7 | n/a (1 seed) | 749 |
| GRPO_T0_it78 − T0_stage1_s0 | pass@1 | – / – | 28.3 | 3.3 / 3.3 | 9.4 | n/a (1 seed) | 700 |
| GRPO_T0_it78 − T0_stage1_s0 | pass@5 | – / – | 35.6 | 4.1 / 4.2 | 11.7 | n/a (1 seed) | 1104 |
| EI_T1v1e_r2_B0 − T0_stage1_s0 | pass@1 | – / – | 19.6 | 2.3 / 2.3 | 6.5 | n/a (1 seed) | 335 |
| EI_T1v1e_r2_B0 − T0_stage1_s0 | pass@5 | – / – | 27.3 | 3.2 / 3.2 | 9.0 | n/a (1 seed) | 649 |
| EI_T1v1e_r2_B64 − T0_stage1_s0 | pass@1 | – / – | 25.0 | 2.9 / 2.9 | 8.2 | n/a (1 seed) | 544 |
| EI_T1v1e_r2_B64 − T0_stage1_s0 | pass@5 | – / – | 33.5 | 3.9 / 3.9 | 11.1 | n/a (1 seed) | 979 |
| EI_T1v1e_r2_B256 − T0_stage1_s0 | pass@1 | – / – | 26.2 | 3.0 / 3.1 | 8.6 | n/a (1 seed) | 598 |
| EI_T1v1e_r2_B256 − T0_stage1_s0 | pass@5 | – / – | 38.0 | 4.5 / 4.5 | 12.6 | n/a (1 seed) | 1263 |
| GRPO_T0_it39 − EI_T0_r1 | pass@1 | – / – | 27.3 | 3.2 / 3.2 | 9.0 | n/a (1 seed) | 649 |
| GRPO_T0_it39 − EI_T0_r1 | pass@5 | – / – | 29.7 | 3.5 / 3.5 | 9.8 | n/a (1 seed) | 769 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | pass@1 | – / – | 23.3 | 2.8 / 2.7 | 7.7 | n/a (1 seed) | 475 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | pass@5 | – / – | 27.5 | 3.2 / 3.2 | 9.1 | n/a (1 seed) | 659 |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | pass@1 | – | 24.2 | 2.7 / 2.9 | 8.0 | n/a (1 seed) | 511 |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | pass@5 | – | 33.5 | 3.8 / 3.9 | 11.1 | n/a (1 seed) | 977 |
| GRPO_T0_it78 − EI_T0_r2 | pass@1 | – / – | 19.0 | 2.2 / 2.2 | 6.3 | n/a (1 seed) | 314 |
| GRPO_T0_it78 − EI_T0_r2 | pass@5 | – / – | 30.1 | 3.5 / 3.5 | 9.9 | n/a (1 seed) | 791 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | pass@1 | – / – | 26.6 | 3.2 / 3.1 | 8.8 | n/a (1 seed) | 618 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | pass@5 | – / – | 30.6 | 3.6 / 3.6 | 10.1 | n/a (1 seed) | 815 |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | pass@1 | – | 24.4 | 2.8 / 2.9 | 8.1 | n/a (1 seed) | 521 |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | pass@5 | – | 33.4 | 3.9 / 3.9 | 11.0 | n/a (1 seed) | 974 |

