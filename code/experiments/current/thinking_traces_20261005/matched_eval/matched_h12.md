### matched h12

```
python -m code.thinking.stats table --group T0_stage1_s12=/scratch/cowifornia/nd-rl/think/runs/eval/h12_T0_s1/rows_plain_B0.jsonl,/scratch/cowifornia/nd-rl/think/runs/eval/h12_T0_s2/rows_plain_B0.jsonl --group T1v1_stage1_B0=/scratch/cowifornia/nd-rl/think/runs/eval/h12_T1v1_s0/rows_plain_B0.jsonl --group T1v1_stage1_B256=/scratch/cowifornia/nd-rl/think/runs/eval/h12_T1v1_s0/rows_plain_B256.jsonl --group EI_T0_r1=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_eiT0_r1/rows_plain_B0.jsonl --group GRPO_T0_it39=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_grpoT0_it39/rows_plain_B0.jsonl --group EI_T1v1e_r1_B0=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_eiT1v1e_r1/rows_plain_B0.jsonl --group GRPO_T1v1e_it39_B0=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_grpoT1v1e_it39/rows_plain_B0.jsonl --group EI_T1v1e_r1_B64=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_eiT1v1e_r1/rows_plain_B64.jsonl --group EI_T1v1e_r1_B256=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_eiT1v1e_r1/rows_plain_B256.jsonl --group EI_T0_r2=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_eiT0_r2/rows_plain_B0.jsonl --group GRPO_T0_it78=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_grpoT0_it78/rows_plain_B0.jsonl --group EI_T1v1e_r2_B0=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_eiT1v1e_r2/rows_plain_B0.jsonl --group EI_T1v1e_r2_B64=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_eiT1v1e_r2/rows_plain_B64.jsonl --group EI_T1v1e_r2_B256=/scratch/cowifornia/nd-rl/think/runs/eval/matched/h12_eiT1v1e_r2/rows_plain_B256.jsonl --pair 'GRPO_T0_it39 vs EI_T0_r1' --pair 'EI_T1v1e_r1_B256 vs EI_T0_r1' --pair 'EI_T1v1e_r1_B256 vs EI_T1v1e_r1_B0' --pair 'GRPO_T0_it78 vs EI_T0_r2' --pair 'EI_T1v1e_r2_B256 vs EI_T0_r2' --pair 'EI_T1v1e_r2_B256 vs EI_T1v1e_r2_B0' --k 1,5 --tokens 1000,3000 --title 'matched h12' --out code/experiments/current/thinking_traces_20261005/matched_eval/matched_h12.md
```

n = 1000 theorems (hard12_test_1k); 4000 draws, RNG seed 0; resampled: both. Brackets: 95% percentile intervals.

| arm | seeds | pass@1 | pass@5 | tokens / attempt |
|---|---|---|---|---|
| T0_stage1_s12 | 2 | 42.4% [40.7, 44.2] | 75.2% [73.0, 77.3] | 167 |
| T1v1_stage1_B0 | 1 | 27.9% [26.3, 29.5] | 59.6% [57.2, 62.0] | 187 |
| T1v1_stage1_B256 | 1 | 43.2% [41.7, 44.8] | 81.1% [79.4, 82.9] | 611 |
| EI_T0_r1 | 1 | 78.0% [76.6, 79.5] | 96.6% [95.7, 97.4] | 182 |
| GRPO_T0_it39 | 1 | 90.3% [88.9, 91.7] | 96.2% [95.2, 97.2] | 142 |
| EI_T1v1e_r1_B0 | 1 | 62.7% [60.5, 64.9] | 85.1% [83.3, 86.9] | 172 |
| GRPO_T1v1e_it39_B0 | 1 | 84.6% [82.8, 86.4] | 92.9% [91.4, 94.3] | 149 |
| EI_T1v1e_r1_B64 | 1 | 44.0% [42.0, 46.0] | 74.5% [72.2, 76.7] | 493 |
| EI_T1v1e_r1_B256 | 1 | 77.7% [76.2, 79.2] | 96.4% [95.5, 97.2] | 637 |
| EI_T0_r2 | 1 | 82.7% [81.4, 84.0] | 97.7% [97.0, 98.4] | 190 |
| GRPO_T0_it78 | 1 | 92.8% [91.4, 94.1] | 96.6% [95.6, 97.6] | 140 |
| EI_T1v1e_r2_B0 | 1 | 77.5% [75.6, 79.4] | 92.5% [91.1, 93.8] | 160 |
| EI_T1v1e_r2_B64 | 1 | 62.2% [60.2, 64.3] | 86.1% [84.2, 87.8] | 453 |
| EI_T1v1e_r2_B256 | 1 | 86.5% [85.3, 87.7] | 98.6% [98.1, 99.1] | 601 |

Paired differences (pp; theorems shared). Sign test: theorems where the seed-mean difference is +/−. t-check: analytic SE, Satterthwaite t 95% interval.

| difference | k | diff [CI] | boot p | sign +/− (p) | t-check [CI] (df) | seeds |
|---|---|---|---|---|---|---|
| T1v1_stage1_B0 − T0_stage1_s12 | pass@1 | −14.6 [−16.3, −12.8] | <0.0003 | 247/723 (9.3e-55) | [−16.2, −12.9] (999.0) | 1 vs 2 |
| T1v1_stage1_B0 − T0_stage1_s12 | pass@5 | −15.6 [−18.1, −13.0] | <0.0003 | 333/651 (2.1e-24) | [−18.0, −13.1] (51.5) | 1 vs 2 |
| T1v1_stage1_B256 − T0_stage1_s12 | pass@1 | +0.8 [−1.0, +2.5] | 0.402 | 496/478 (0.586) | [−0.8, +2.4] (999.0) | 1 vs 2 |
| T1v1_stage1_B256 − T0_stage1_s12 | pass@5 | +5.9 [+3.6, +8.3] | <0.0003 | 595/396 (2.8e-10) | [+3.8, +8.1] (29.7) | 1 vs 2 |
| EI_T0_r1 − T0_stage1_s12 | pass@1 | +35.6 [+33.7, +37.5] | <0.0003 | 897/89 (9.8e-169) | [+33.8, +37.3] (999.0) | 1 vs 2 |
| EI_T0_r1 − T0_stage1_s12 | pass@5 | +21.4 [+19.1, +23.7] | <0.0003 | 933/50 (1.0e-211) | [+19.3, +23.4] (21.5) | 1 vs 2 |
| GRPO_T0_it39 − T0_stage1_s12 | pass@1 | +47.8 [+45.9, +49.8] | <0.0003 | 945/53 (4.1e-212) | [+46.0, +49.7] (999.0) | 1 vs 2 |
| GRPO_T0_it39 − T0_stage1_s12 | pass@5 | +21.0 [+18.8, +23.3] | <0.0003 | 943/40 (7.1e-225) | [+18.9, +23.1] (27.1) | 1 vs 2 |
| EI_T1v1e_r1_B0 − T0_stage1_s12 | pass@1 | +20.3 [+18.1, +22.5] | <0.0003 | 714/265 (3.4e-48) | [+18.2, +22.4] (999.0) | 1 vs 2 |
| EI_T1v1e_r1_B0 − T0_stage1_s12 | pass@5 | +9.9 [+7.6, +12.3] | <0.0003 | 769/214 (5.2e-74) | [+7.7, +12.2] (34.2) | 1 vs 2 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s12 | pass@1 | +42.2 [+40.0, +44.3] | <0.0003 | 895/100 (2.5e-160) | [+40.1, +44.2] (999.0) | 1 vs 2 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s12 | pass@5 | +17.7 [+15.3, +20.1] | <0.0003 | 899/83 (8.5e-174) | [+15.4, +19.9] (33.4) | 1 vs 2 |
| EI_T1v1e_r1_B64 − T0_stage1_s12 | pass@1 | +1.5 [−1.0, +4.0] | 0.226 | 511/470 (0.202) | [−0.8, +3.9] (999.0) | 1 vs 2 |
| EI_T1v1e_r1_B64 − T0_stage1_s12 | pass@5 | −0.7 [−3.7, +2.1] | 0.623 | 609/379 (2.5e-13) | [−3.6, +2.1] (95.2) | 1 vs 2 |
| EI_T1v1e_r1_B256 − T0_stage1_s12 | pass@1 | +35.2 [+33.4, +37.1] | <0.0003 | 903/87 (8.5e-172) | [+33.6, +36.9] (999.0) | 1 vs 2 |
| EI_T1v1e_r1_B256 − T0_stage1_s12 | pass@5 | +21.2 [+19.0, +23.5] | <0.0003 | 930/51 (6.7e-210) | [+19.2, +23.2] (20.4) | 1 vs 2 |
| EI_T0_r2 − T0_stage1_s12 | pass@1 | +40.3 [+38.4, +42.2] | <0.0003 | 926/61 (2.2e-199) | [+38.6, +42.0] (999.0) | 1 vs 2 |
| EI_T0_r2 − T0_stage1_s12 | pass@5 | +22.5 [+20.4, +24.7] | <0.0003 | 949/35 (3.8e-232) | [+20.5, +24.6] (20.6) | 1 vs 2 |
| GRPO_T0_it78 − T0_stage1_s12 | pass@1 | +50.3 [+48.4, +52.3] | <0.0003 | 956/42 (2.1e-226) | [+48.5, +52.1] (999.0) | 1 vs 2 |
| GRPO_T0_it78 − T0_stage1_s12 | pass@5 | +21.4 [+19.2, +23.6] | <0.0003 | 945/37 (9.5e-229) | [+19.3, +23.5] (25.1) | 1 vs 2 |
| EI_T1v1e_r2_B0 − T0_stage1_s12 | pass@1 | +35.0 [+32.9, +37.1] | <0.0003 | 860/132 (1.9e-131) | [+33.1, +36.9] (999.0) | 1 vs 2 |
| EI_T1v1e_r2_B0 − T0_stage1_s12 | pass@5 | +17.2 [+15.1, +19.5] | <0.0003 | 882/99 (1.1e-157) | [+15.1, +19.4] (25.3) | 1 vs 2 |
| EI_T1v1e_r2_B64 − T0_stage1_s12 | pass@1 | +19.8 [+17.3, +22.2] | <0.0003 | 705/281 (1.1e-42) | [+17.4, +22.2] (999.0) | 1 vs 2 |
| EI_T1v1e_r2_B64 − T0_stage1_s12 | pass@5 | +10.9 [+8.1, +13.6] | <0.0003 | 764/224 (1.6e-69) | [+8.3, +13.5] (63.4) | 1 vs 2 |
| EI_T1v1e_r2_B256 − T0_stage1_s12 | pass@1 | +44.0 [+42.3, +45.8] | <0.0003 | 958/36 (1.4e-233) | [+42.4, +45.6] (999.0) | 1 vs 2 |
| EI_T1v1e_r2_B256 − T0_stage1_s12 | pass@5 | +23.4 [+21.3, +25.7] | <0.0003 | 965/19 (6.3e-257) | [+21.4, +25.4] (20.3) | 1 vs 2 |
| GRPO_T0_it39 − EI_T0_r1 | pass@1 | +12.3 [+10.6, +13.8] | <0.0003 | 657/124 (2.1e-88) | [+10.6, +13.9] (999.0) | 1 vs 1 |
| GRPO_T0_it39 − EI_T0_r1 | pass@5 | −0.4 [−1.5, +0.7] | 0.493 | 241/87 (7.2e-18) | [−1.5, +0.7] (999.0) | 1 vs 1 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | pass@1 | −0.3 [−2.2, +1.5] | 0.724 | 421/436 (0.633) | [−2.1, +1.5] (999.0) | 1 vs 1 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | pass@5 | −0.2 [−1.3, +0.9] | 0.725 | 221/220 (1.000) | [−1.3, +0.9] (999.0) | 1 vs 1 |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | pass@1 | +14.9 [+13.1, +17.0] | <0.0003 | 562/267 (5.6e-25) | [+13.0, +16.9] (999.0) | 1 vs 1 (paired) |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | pass@5 | +11.3 [+9.5, +13.0] | <0.0003 | 409/115 (1.2e-39) | [+9.5, +13.0] (999.0) | 1 vs 1 (paired) |
| GRPO_T0_it78 − EI_T0_r2 | pass@1 | +10.0 [+8.6, +11.6] | <0.0003 | 592/102 (8.1e-85) | [+8.5, +11.6] (999.0) | 1 vs 1 |
| GRPO_T0_it78 − EI_T0_r2 | pass@5 | −1.1 [−2.2, −0.1] | 0.040 | 177/71 (1.3e-11) | [−2.1, −0.1] (999.0) | 1 vs 1 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | pass@1 | +3.7 [+2.2, +5.3] | <0.0003 | 460/330 (4.2e-06) | [+2.2, +5.3] (999.0) | 1 vs 1 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | pass@5 | +0.9 [+0.1, +1.7] | 0.035 | 180/118 (3.9e-04) | [+0.1, +1.7] (999.0) | 1 vs 1 |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | pass@1 | +9.0 [+7.4, +10.7] | <0.0003 | 432/282 (2.2e-08) | [+7.4, +10.7] (999.0) | 1 vs 1 (paired) |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | pass@5 | +6.2 [+4.9, +7.5] | <0.0003 | 273/72 (1.1e-28) | [+4.9, +7.4] (999.0) | 1 vs 1 (paired) |

pass@k at matched sampled tokens per theorem (interpolated in log tokens; – out of the read's range; † more than 5% of draws clamped to pass@1 or pass@n).

| arm | 1k | 3k |
|---|---|---|
| T0_stage1_s12 | 77.7% [75.4, 80.0] | 88.9% [87.0, 90.6] |
| T1v1_stage1_B0 | 60.7% [58.3, 63.2] | 76.5% [74.2, 78.8] |
| T1v1_stage1_B256 | 56.4% [54.5, 58.5] | 80.8% [79.0, 82.7] |
| EI_T0_r1 | 96.9% [96.0, 97.7] | – |
| GRPO_T0_it39 | 96.8% [95.8, 97.7] | – |
| EI_T1v1e_r1_B0 | 86.5% [84.7, 88.2] | – |
| GRPO_T1v1e_it39_B0 | 93.6% [92.2, 95.0] | – |
| EI_T1v1e_r1_B64 | 59.4% [57.1, 61.8] | 76.9% [74.5, 79.1] |
| EI_T1v1e_r1_B256 | 85.6% [84.1, 87.1] | 96.1% [95.2, 97.0] |
| EI_T0_r2 | 97.9% [97.2, 98.5] | 99.4% [98.9, 99.8]† |
| GRPO_T0_it78 | 97.2% [96.2, 98.1] | – |
| EI_T1v1e_r2_B0 | 93.4% [92.1, 94.7] | – |
| EI_T1v1e_r2_B64 | 77.2% [75.0, 79.3] | 88.0% [86.2, 89.8] |
| EI_T1v1e_r2_B256 | 92.7% [91.6, 93.8] | 98.6% [98.0, 99.1] |

| difference (pp) | 1k | 3k |
|---|---|---|
| T1v1_stage1_B0 − T0_stage1_s12 | −17.0 [−19.6, −14.3] p <0.0003 | −12.3 [−14.9, −9.8] p <0.0003 |
| T1v1_stage1_B256 − T0_stage1_s12 | −21.3 [−23.7, −18.8] p <0.0003 | −8.0 [−10.2, −5.7] p <0.0003 |
| EI_T0_r1 − T0_stage1_s12 | +19.1 [+16.9, +21.5] p <0.0003 | – |
| GRPO_T0_it39 − T0_stage1_s12 | +19.0 [+16.8, +21.4] p <0.0003 | – |
| EI_T1v1e_r1_B0 − T0_stage1_s12 | +8.7 [+6.3, +11.2] p <0.0003 | – |
| GRPO_T1v1e_it39_B0 − T0_stage1_s12 | +15.9 [+13.5, +18.4] p <0.0003 | – |
| EI_T1v1e_r1_B64 − T0_stage1_s12 | −18.3 [−21.4, −15.2] p <0.0003 | −12.0 [−14.8, −9.3] p <0.0003 |
| EI_T1v1e_r1_B256 − T0_stage1_s12 | +7.9 [+5.6, +10.3] p <0.0003 | +7.3 [+5.5, +9.2] p <0.0003 |
| EI_T0_r2 − T0_stage1_s12 | +20.1 [+17.9, +22.4] p <0.0003 | +10.5 [+8.7, +12.4]† p <0.0003 |
| GRPO_T0_it78 − T0_stage1_s12 | +19.4 [+17.1, +21.8] p <0.0003 | – |
| EI_T1v1e_r2_B0 − T0_stage1_s12 | +15.7 [+13.5, +18.0] p <0.0003 | – |
| EI_T1v1e_r2_B64 − T0_stage1_s12 | −0.6 [−3.6, +2.4] p 0.708 | −0.8 [−3.3, +1.6] p 0.506 |
| EI_T1v1e_r2_B256 − T0_stage1_s12 | +15.0 [+12.8, +17.3] p <0.0003 | +9.8 [+8.0, +11.6] p <0.0003 |
| GRPO_T0_it39 − EI_T0_r1 | −0.1 [−1.2, +0.9] p 0.854 | – |
| EI_T1v1e_r1_B256 − EI_T0_r1 | −11.3 [−12.8, −9.7] p <0.0003 | – |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | −0.8 [−2.5, +0.9] p 0.361 | – |
| GRPO_T0_it78 − EI_T0_r2 | −0.7 [−1.7, +0.3] p 0.178 | – |
| EI_T1v1e_r2_B256 − EI_T0_r2 | −5.1 [−6.3, −4.0] p <0.0003 | −0.8 [−1.5, −0.1]† p 0.034 |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | −0.7 [−2.0, +0.6] p 0.278 | – |

Informative theorems (not 0% or not 100% in every seed of every arm above):

| k | 0% everywhere | 100% everywhere | informative |
|---|---|---|---|
| pass@1 | 0 | 0 | 1000 |
| pass@5 | 0 | 2 | 998 |
| pass@16 | 0 | 606 | 394 |

Power (analytic components; two-sided α 0.05, 80% power, z approximation): minimum detectable difference (MDE) of this design, and the seeds per arm (at n = 1000) or theorems (at these seeds) needed for an MDE of 3.0 pp. Seed SDs come from 2-3 seeds (1-2 df each) and are rough; 0.0 means the seed means spread no more than theorem-level noise predicts.

| difference | k | seed SD a / b (pp) | theorem SD (pp) | SE boot / analytic (pp) | MDE (pp) | seeds for 3.0 pp | theorems for 3.0 pp |
|---|---|---|---|---|---|---|---|
| T1v1_stage1_B0 − T0_stage1_s12 | pass@1 | – / 0.0 | 21.5 | 0.9 / 0.8 | 2.4 | n/a (1 seed) | 626 |
| T1v1_stage1_B0 − T0_stage1_s12 | pass@5 | – / 0.6 | 30.8 | 1.3 / 1.2 | 3.4 | n/a (1 seed) | 1386 |
| T1v1_stage1_B256 − T0_stage1_s12 | pass@1 | – / 0.0 | 20.2 | 0.9 / 0.8 | 2.3 | n/a (1 seed) | 576 |
| T1v1_stage1_B256 − T0_stage1_s12 | pass@5 | – / 0.6 | 24.0 | 1.2 / 1.1 | 3.0 | n/a (1 seed) | 989 |
| EI_T0_r1 − T0_stage1_s12 | pass@1 | – / 0.0 | 22.9 | 1.0 / 0.9 | 2.5 | n/a (1 seed) | 677 |
| EI_T0_r1 − T0_stage1_s12 | pass@5 | – / 0.6 | 20.1 | 1.1 / 1.0 | 2.7 | n/a (1 seed) | 806 |
| GRPO_T0_it39 − T0_stage1_s12 | pass@1 | – / 0.0 | 25.6 | 1.0 / 1.0 | 2.7 | n/a (1 seed) | 792 |
| GRPO_T0_it39 − T0_stage1_s12 | pass@5 | – / 0.6 | 22.9 | 1.1 / 1.0 | 2.9 | n/a (1 seed) | 932 |
| EI_T1v1e_r1_B0 − T0_stage1_s12 | pass@1 | – / 0.0 | 29.6 | 1.1 / 1.1 | 3.0 | n/a (1 seed) | 987 |
| EI_T1v1e_r1_B0 − T0_stage1_s12 | pass@5 | – / 0.6 | 25.7 | 1.2 / 1.1 | 3.1 | n/a (1 seed) | 1080 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s12 | pass@1 | – / 0.0 | 28.7 | 1.1 / 1.0 | 2.9 | n/a (1 seed) | 941 |
| GRPO_T1v1e_it39_B0 − T0_stage1_s12 | pass@5 | – / 0.6 | 25.4 | 1.2 / 1.1 | 3.1 | n/a (1 seed) | 1064 |
| EI_T1v1e_r1_B64 − T0_stage1_s12 | pass@1 | – / 0.0 | 34.6 | 1.3 / 1.2 | 3.4 | n/a (1 seed) | 1263 |
| EI_T1v1e_r1_B64 − T0_stage1_s12 | pass@5 | – / 0.6 | 39.2 | 1.5 / 1.4 | 4.1 | n/a (1 seed) | 2009 |
| EI_T1v1e_r1_B256 − T0_stage1_s12 | pass@1 | – / 0.0 | 22.0 | 0.9 / 0.9 | 2.4 | n/a (1 seed) | 642 |
| EI_T1v1e_r1_B256 − T0_stage1_s12 | pass@5 | – / 0.6 | 19.4 | 1.1 / 1.0 | 2.7 | n/a (1 seed) | 778 |
| EI_T0_r2 − T0_stage1_s12 | pass@1 | – / 0.0 | 22.6 | 1.0 / 0.9 | 2.4 | n/a (1 seed) | 667 |
| EI_T0_r2 − T0_stage1_s12 | pass@5 | – / 0.6 | 19.6 | 1.1 / 1.0 | 2.7 | n/a (1 seed) | 783 |
| GRPO_T0_it78 − T0_stage1_s12 | pass@1 | – / 0.0 | 24.6 | 1.0 / 0.9 | 2.6 | n/a (1 seed) | 748 |
| GRPO_T0_it78 − T0_stage1_s12 | pass@5 | – / 0.6 | 22.0 | 1.1 / 1.0 | 2.9 | n/a (1 seed) | 890 |
| EI_T1v1e_r2_B0 − T0_stage1_s12 | pass@1 | – / 0.0 | 26.3 | 1.1 / 1.0 | 2.7 | n/a (1 seed) | 826 |
| EI_T1v1e_r2_B0 − T0_stage1_s12 | pass@5 | – / 0.6 | 22.1 | 1.1 / 1.0 | 2.9 | n/a (1 seed) | 893 |
| EI_T1v1e_r2_B64 − T0_stage1_s12 | pass@1 | – / 0.0 | 35.1 | 1.3 / 1.2 | 3.4 | n/a (1 seed) | 1293 |
| EI_T1v1e_r2_B64 − T0_stage1_s12 | pass@5 | – / 0.6 | 33.5 | 1.4 / 1.3 | 3.6 | n/a (1 seed) | 1572 |
| EI_T1v1e_r2_B256 − T0_stage1_s12 | pass@1 | – / 0.0 | 20.1 | 0.9 / 0.8 | 2.3 | n/a (1 seed) | 572 |
| EI_T1v1e_r2_B256 − T0_stage1_s12 | pass@5 | – / 0.6 | 19.4 | 1.1 / 1.0 | 2.7 | n/a (1 seed) | 776 |
| GRPO_T0_it39 − EI_T0_r1 | pass@1 | – / – | 26.7 | 0.8 / 0.8 | 2.4 | n/a (1 seed) | 623 |
| GRPO_T0_it39 − EI_T0_r1 | pass@5 | – / – | 17.9 | 0.6 / 0.6 | 1.6 | n/a (1 seed) | 278 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | pass@1 | – / – | 29.1 | 0.9 / 0.9 | 2.6 | n/a (1 seed) | 741 |
| EI_T1v1e_r1_B256 − EI_T0_r1 | pass@5 | – / – | 18.0 | 0.6 / 0.6 | 1.6 | n/a (1 seed) | 282 |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | pass@1 | – | 31.2 | 1.0 / 1.0 | 2.8 | n/a (1 seed) | 851 |
| EI_T1v1e_r1_B256 − EI_T1v1e_r1_B0 | pass@5 | – | 28.1 | 0.9 / 0.9 | 2.5 | n/a (1 seed) | 691 |
| GRPO_T0_it78 − EI_T0_r2 | pass@1 | – / – | 24.4 | 0.8 / 0.8 | 2.2 | n/a (1 seed) | 521 |
| GRPO_T0_it78 − EI_T0_r2 | pass@5 | – / – | 16.8 | 0.5 / 0.5 | 1.5 | n/a (1 seed) | 247 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | pass@1 | – / – | 25.1 | 0.8 / 0.8 | 2.2 | n/a (1 seed) | 548 |
| EI_T1v1e_r2_B256 − EI_T0_r2 | pass@5 | – / – | 13.2 | 0.4 / 0.4 | 1.2 | n/a (1 seed) | 153 |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | pass@1 | – | 26.5 | 0.8 / 0.8 | 2.4 | n/a (1 seed) | 614 |
| EI_T1v1e_r2_B256 − EI_T1v1e_r2_B0 | pass@5 | – | 20.4 | 0.6 / 0.6 | 1.8 | n/a (1 seed) | 362 |

