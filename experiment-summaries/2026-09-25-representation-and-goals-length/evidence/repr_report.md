---
written_on: 2026-09-25
---

# Representation and length generalisation: report

Generated 2026-09-25 06:03 UTC by `analyse.py` from `/scratch/cowifornia/nd-rl/ab3/repr/eval`. Every variant: 039 pretraining recipe (2,911 fixed steps), then arm C RL (L_true 6-11, proofs <= 11 lines, DEFAULT_FLOPS), then k = 32 samples per theorem at T = 0.8 (max_new 1024) at every length. pass@k are unbiased estimates averaged over theorems; brackets are the [min, max] over seeds. Runs evaluated: abs 3, rel 3, nope 3, posline 3.

## Headline

| variant | runs | pass@1 L6-11 pooled | pass@1 L11 | pass@1 L12 | pass@1 L12-18 pooled | slope past 11, L12-18 [95% t] | slope from 11, L11-18 [95% t] | held-out greedy after RL |
|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 3 | 47.7% | 23.2% | 9.6% | 1.57% | -3.22 [-5.15, -1.28] | -1.53 [-1.94, -1.12] | 98.6% |
| rel (relative citations) | 3 | 28.0% | 0.0% | 0.0% | 0.00% | n/a | n/a | 98.7% |
| nope (no positional encoding) | 3 | 47.2% | 25.0% | 9.6% | 1.53% | -3.19 [-5.12, -1.25] | -1.66 [-2.34, -0.97] | 98.8% |
| pos-line (line + offset rotary) | 3 | 46.3% | 22.3% | 4.2% | 0.65% | -5.00 [-20.28, 10.28] | -2.16 [-2.63, -1.68] | 98.5% |

## 1. Reproduction check: abs re-pretrain against the released 039 checkpoints

Both measured with the same code on our cards. val loss = 039's val_loss (train records [0, 1024), no shift); held-out greedy (039) = 039 harness slice heldout_cap6[::2][:2000]; held-out greedy (frozen) = run-3 harness slice [::5][:1000].

| checkpoint | seed | steps | val loss | held-out greedy (039) | held-out greedy (frozen) | train secs |
|---|---|---|---|---|---|---|
| released 039 | 0 | 2911 | 0.0933 | 98.80% | 97.80% | 300 (A40) |
| abs re-pretrain (ours) | 0 | 2911 | 0.0941 | 98.95% | 97.90% | 678 |
| released 039 | 1 | 2931 | 0.0967 | 98.85% | 97.90% | 300 (A40) |
| abs re-pretrain (ours) | 1 | 2911 | 0.0938 | 99.00% | 97.80% | 678 |
| released 039 | 2 | 2913 | 0.0935 | 98.75% | 97.70% | 300 (A40) |
| abs re-pretrain (ours) | 2 | 2911 | 0.0945 | 98.80% | 98.20% | 570 |

All pretrained checkpoints (val loss is per proof token and not comparable across tokenizers: rel has no line-index tokens):

| variant | seed | val loss | held-out greedy (039) | held-out greedy (frozen) | random positions: val loss / greedy (039) | steps/s |
|---|---|---|---|---|---|---|
| abs (baseline) | 0 | 0.0941 | 98.95% | 97.90% | - | 4.29 |
| abs (baseline) | 1 | 0.0938 | 99.00% | 97.80% | - | 4.29 |
| abs (baseline) | 2 | 0.0945 | 98.80% | 98.20% | - | 5.11 |
| rel (relative citations) | 0 | 0.0014 | 98.75% | 98.20% | - | 4.40 |
| rel (relative citations) | 1 | 0.0011 | 99.25% | 98.20% | - | 3.46 |
| rel (relative citations) | 2 | 0.0015 | 98.85% | 98.00% | - | 5.35 |
| pos-rand (random positions) | 0 | 0.5860 | 0.00% | 0.00% | 0.0932 / 77.05% | 4.07 |
| pos-rand (random positions) | 1 | 0.5924 | 0.00% | 0.00% | 0.0932 / 74.05% | 4.27 |
| pos-rand (random positions) | 2 | 0.6726 | 0.00% | 0.00% | 0.0944 / 75.10% | 3.31 |
| nope (no positional encoding) | 0 | 0.0937 | 98.55% | 97.80% | - | 7.07 |
| nope (no positional encoding) | 1 | 0.0936 | 98.35% | 97.80% | - | 5.87 |
| nope (no positional encoding) | 2 | 0.0951 | 98.55% | 97.50% | - | 4.65 |
| pos-line (line + offset rotary) | 0 | 0.0948 | 98.70% | 97.40% | - | 5.18 |
| pos-line (line + offset rotary) | 1 | 0.0944 | 98.45% | 97.90% | - | 3.36 |
| pos-line (line + offset rotary) | 2 | 0.0945 | 99.05% | 98.40% | - | 4.18 |

## 2. pass@k by variant and length (standard sets)

**pass@1**

| variant | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 80.4% [79.4%, 82.0%] | 74.2% [71.2%, 78.2%] | 43.8% [43.3%, 44.3%] | 36.1% [34.4%, 37.0%] | 28.2% [26.3%, 29.7%] | 23.2% [21.9%, 25.1%] | 9.6% [8.6%, 11.2%] | 0.6% [0.2%, 1.3%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| rel (relative citations) | 77.7% [76.4%, 79.9%] | 56.4% [54.6%, 57.6%] | 24.6% [23.7%, 26.1%] | 8.3% [7.6%, 8.8%] | 0.9% [0.7%, 1.2%] | 0.0% [0.0%, 0.1%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| nope (no positional encoding) | 80.2% [79.0%, 82.0%] | 70.2% [59.4%, 77.6%] | 43.5% [42.7%, 44.6%] | 36.5% [36.0%, 37.2%] | 27.8% [27.7%, 27.8%] | 25.0% [24.5%, 25.6%] | 9.6% [6.5%, 12.9%] | 0.4% [0.0%, 0.8%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| pos-line (line + offset rotary) | 79.4% [77.8%, 82.1%] | 72.0% [68.6%, 76.2%] | 42.9% [40.9%, 44.7%] | 35.4% [33.7%, 37.3%] | 25.5% [25.2%, 25.8%] | 22.3% [21.8%, 22.6%] | 4.2% [3.1%, 4.9%] | 0.0% [0.0%, 0.1%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

**pass@4**

| variant | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 89.5% [89.0%, 90.1%] | 83.5% [80.8%, 87.1%] | 51.9% [51.4%, 52.9%] | 42.9% [41.5%, 43.6%] | 33.6% [31.5%, 36.4%] | 28.4% [27.1%, 30.5%] | 15.0% [13.3%, 15.9%] | 1.2% [0.6%, 2.1%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| rel (relative citations) | 85.4% [84.1%, 87.5%] | 61.0% [59.5%, 62.3%] | 28.3% [27.4%, 29.5%] | 10.7% [9.2%, 11.5%] | 1.5% [1.2%, 1.7%] | 0.1% [0.0%, 0.2%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| nope (no positional encoding) | 89.6% [88.1%, 90.6%] | 78.4% [67.8%, 84.4%] | 51.0% [50.6%, 51.8%] | 42.2% [41.7%, 42.6%] | 33.0% [33.0%, 33.0%] | 29.5% [28.6%, 30.6%] | 14.3% [10.5%, 18.2%] | 0.8% [0.0%, 1.7%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| pos-line (line + offset rotary) | 89.2% [87.8%, 91.3%] | 79.7% [76.7%, 84.7%] | 50.6% [47.7%, 52.9%] | 40.7% [39.5%, 42.8%] | 30.3% [30.1%, 30.4%] | 26.8% [26.7%, 27.0%] | 6.7% [5.7%, 7.2%] | 0.1% [0.0%, 0.2%] | 0.0% [0.0%, 0.1%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

**pass@32**

| variant | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 94.7% [94.3%, 95.0%] | 89.1% [86.3%, 93.0%] | 57.7% [56.7%, 58.7%] | 47.3% [45.7%, 48.3%] | 37.4% [34.3%, 41.3%] | 32.3% [30.7%, 35.0%] | 20.9% [18.7%, 23.0%] | 2.2% [1.3%, 3.7%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| rel (relative citations) | 91.1% [89.0%, 94.3%] | 64.9% [64.0%, 66.0%] | 32.2% [31.7%, 33.0%] | 13.3% [11.3%, 14.7%] | 2.2% [1.3%, 3.0%] | 0.1% [0.0%, 0.3%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| nope (no positional encoding) | 94.4% [92.7%, 95.7%] | 83.0% [73.0%, 88.3%] | 55.6% [54.3%, 58.0%] | 46.2% [45.3%, 48.0%] | 37.9% [37.3%, 38.7%] | 33.1% [31.3%, 34.3%] | 18.9% [16.0%, 22.3%] | 1.6% [0.0%, 3.7%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| pos-line (line + offset rotary) | 94.8% [93.0%, 96.0%] | 84.0% [80.3%, 88.3%] | 58.0% [56.0%, 59.7%] | 44.3% [43.3%, 45.7%] | 33.4% [33.3%, 33.7%] | 31.0% [30.7%, 31.3%] | 9.6% [9.0%, 10.0%] | 0.3% [0.0%, 0.7%] | 0.2% [0.0%, 0.6%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

Reference: lengthexp arm C (the older frozen/pretrain.pt, same RL recipe, seeds 101-103), pass@1:

| variant | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arm C (lengthexp, older pretrain) | 56.2% [55.5%, 56.9%] | 36.4% [36.0%, 36.8%] | 19.1% [17.3%, 21.3%] | 17.1% [16.1%, 18.4%] | 10.9% [9.9%, 11.8%] | 12.7% [12.6%, 12.9%] | 3.4% [3.1%, 3.9%] | 0.2% [0.1%, 0.3%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

![pass@1](fig_pass.png)


## 3. Log-odds drop per extra line past the training max (11)

Grouped-binomial logistic fit of pass@1 per run (lengthexp/analyse.py fit_logistic) against d = L - 11. The slope is the change in log-odds per extra line (more negative = steeper fall-off). "n/a" = successes at fewer than two lengths in the window, so the slope is not estimable. Tests against abs: Welch t and the exact permutation test over the 3 + 3 runs (smallest possible p = 0.10).

**Window d>=1 (L12-18)** (d = L - 11)

| variant | slope per run (log-odds per extra line) | mean | 95% t-interval over seeds | mean bootstrap SE | difference from abs |
|---|---|---|---|---|---|
| abs (baseline) | -2.39, -3.32, -3.94 | -3.22 | [-5.15, -1.28] | 0.74 | - |
| rel (relative citations) | n/a, n/a, n/a | - | - | - | - |
| nope (no positional encoding) | -3.34, n/a, -3.03 | -3.19 | [-5.12, -1.25] | 0.70 | +0.03 (Welch p 0.9541, perm p 1.0000) |
| pos-line (line + offset rotary) | -6.20, n/a, -3.80 | -5.00 | [-20.28, 10.28] | 0.55 | -1.79 (Welch p 0.3569, perm p 0.3000) |

**Window d>=0 (L11-18)** (d = L - 11)

| variant | slope per run (log-odds per extra line) | mean | 95% t-interval over seeds | mean bootstrap SE | difference from abs |
|---|---|---|---|---|---|
| abs (baseline) | -1.35, -1.57, -1.68 | -1.53 | [-1.94, -1.12] | 0.12 | - |
| rel (relative citations) | n/a, n/a, n/a | - | - | - | - |
| nope (no positional encoding) | -1.60, -1.96, -1.42 | -1.66 | [-2.34, -0.97] | 0.12 | -0.13 (Welch p 0.5430, perm p 0.6000) |
| pos-line (line + offset rotary) | -2.01, -2.09, -2.37 | -2.16 | [-2.63, -1.68] | 0.22 | -0.62 (Welch p 0.0134, perm p 0.1000) |

**Differences from abs (seed means; Welch t and exact permutation p over 3 + 3 runs, smallest possible 0.10)**

| variant | measure | variant vs abs | difference | Welch p | perm p |
|---|---|---|---|---|---|
| rel (relative citations) | pass@1 L6-11 pooled | 28.00% vs 47.65% | -19.65 points | 0.0001 | 0.1000 |
| rel (relative citations) | pass@1 L11 | 0.02% vs 23.19% | -23.17 points | 0.0018 | 0.1000 |
| rel (relative citations) | pass@1 L12 | 0.00% vs 9.64% | -9.64 points | 0.0071 | 0.1000 |
| rel (relative citations) | pass@1 L13 | 0.00% vs 0.62% | -0.62 points | 0.2151 | 0.1000 |
| rel (relative citations) | pass@1 L12-18 pooled | 0.00% vs 1.57% | -1.57 points | 0.0123 | 0.1000 |
| nope (no positional encoding) | pass@1 L6-11 pooled | 47.21% vs 47.65% | -0.44 points | 0.7385 | 0.7000 |
| nope (no positional encoding) | pass@1 L11 | 24.99% vs 23.19% | +1.80 points | 0.2051 | 0.3000 |
| nope (no positional encoding) | pass@1 L12 | 9.61% vs 9.64% | -0.03 points | 0.9900 | 1.0000 |
| nope (no positional encoding) | pass@1 L13 | 0.38% vs 0.62% | -0.24 points | 0.5936 | 0.7000 |
| nope (no positional encoding) | pass@1 L12-18 pooled | 1.53% vs 1.57% | -0.04 points | 0.9170 | 1.0000 |
| pos-line (line + offset rotary) | pass@1 L6-11 pooled | 46.27% vs 47.65% | -1.38 points | 0.2125 | 0.3000 |
| pos-line (line + offset rotary) | pass@1 L11 | 22.27% vs 23.19% | -0.91 points | 0.4573 | 0.5000 |
| pos-line (line + offset rotary) | pass@1 L12 | 4.21% vs 9.64% | -5.43 points | 0.0075 | 0.1000 |
| pos-line (line + offset rotary) | pass@1 L13 | 0.02% vs 0.62% | -0.60 points | 0.2254 | 0.1000 |
| pos-line (line + offset rotary) | pass@1 L12-18 pooled | 0.65% vs 1.57% | -0.92 points | 0.0199 | 0.1000 |

## 4. Attempt length

Share of attempts (all samples) with at least L_true written lines:

| variant | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 98.0% | 92.3% | 65.0% | 52.8% | 39.4% | 33.4% | 15.0% | 1.8% | 0.0% | 0.2% | 0.3% | 0.1% | 0.1% |
| rel (relative citations) | 96.5% | 77.4% | 36.9% | 17.4% | 3.8% | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| nope (no positional encoding) | 98.1% | 90.3% | 61.6% | 48.4% | 38.3% | 33.1% | 14.0% | 1.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| pos-line (line + offset rotary) | 97.4% | 88.8% | 63.4% | 49.6% | 35.7% | 34.5% | 12.0% | 2.7% | 0.9% | 5.1% | 5.6% | 7.8% | 9.5% |

Median written lines per attempt:

| variant | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 6 | 7 | 8 | 9 | 8 | 9 | 9 | 9 | 7 | 10 | 10 | 10 | 10 |
| rel (relative citations) | 6 | 7 | 7 | 7 | 7 | 7 | 7 | 8 | 6 | 9 | 9 | 9 | 8 |
| nope (no positional encoding) | 6 | 7 | 8 | 8 | 8 | 8 | 8 | 9 | 7 | 10 | 10 | 10 | 10 |
| pos-line (line + offset rotary) | 6 | 7 | 8 | 8 | 8 | 9 | 9 | 9 | 7 | 10 | 10 | 11 | 11 |

## 5. First-error position

Cell: mean / median position of the first failure among failed samples (first line the incremental checker rejects; early QED counts at the QED slot).

| variant | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 5.3 / 5 | 5.4 / 5 | 4.8 / 5 | 4.8 / 5 | 5.0 / 5 | 5.0 / 5 | 5.4 / 5 | 5.5 / 6 | 4.3 / 3 | 7.5 / 8 | 7.4 / 7 | 7.7 / 8 | 8.1 / 8 |
| rel (relative citations) | 4.9 / 5 | 4.6 / 5 | 4.4 / 4 | 4.6 / 5 | 4.6 / 5 | 4.6 / 5 | 4.6 / 5 | 4.6 / 5 | 4.0 / 3 | 6.4 / 7 | 6.3 / 6 | 6.3 / 6 | 6.2 / 6 |
| nope (no positional encoding) | 5.2 / 5 | 5.2 / 5 | 4.5 / 4 | 4.7 / 4 | 4.9 / 5 | 4.8 / 5 | 5.2 / 5 | 5.6 / 6 | 4.2 / 3 | 7.3 / 7 | 7.2 / 7 | 7.3 / 7 | 7.4 / 7 |
| pos-line (line + offset rotary) | 5.1 / 5 | 5.3 / 5 | 4.7 / 5 | 4.8 / 5 | 4.9 / 5 | 5.0 / 5 | 5.6 / 6 | 5.5 / 6 | 4.3 / 3 | 7.0 / 7 | 7.0 / 7 | 7.5 / 7 | 7.8 / 7 |

![attempts and first errors](fig_attempts.png)


**Per-position failure hazard on the long sets** (L12-18 pooled over lengths and seeds): P(first failure at line p | lines 1..p-1 valid); an early QED counts at its QED slot; blank = fewer than 100 samples still at risk; bold = the largest. A spike near p = 12 (training max + 1) would mean attempts die where the training data ends (a length-specific circuit); a flat hazard means errors are spread over lines.

| variant | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 | p16 | p17 | p18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 0.00 | 0.05 | 0.08 | 0.05 | 0.11 | 0.21 | 0.33 | 0.43 | 0.43 | 0.41 | 0.44 | **0.51** | 0.23 | 0.11 |  |  |  |  |
| rel (relative citations) | 0.00 | 0.04 | 0.09 | 0.09 | 0.30 | 0.50 | 0.63 | 0.70 | 0.72 | 0.91 | **1.00** |  |  |  |  |  |  |  |
| nope (no positional encoding) | 0.00 | 0.04 | 0.08 | 0.06 | 0.12 | 0.27 | 0.42 | 0.48 | 0.50 | **0.50** | 0.44 | 0.31 | 0.03 | 0.01 |  |  |  |  |
| pos-line (line + offset rotary) | 0.00 | 0.04 | 0.08 | 0.07 | 0.16 | 0.24 | 0.41 | 0.45 | 0.36 | 0.39 | 0.47 | **0.71** | 0.36 |  |  |  |  |  |

The same on L15-18 only (the constructed long sets):

| variant | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 | p16 | p17 | p18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 0.00 | 0.00 | 0.01 | 0.02 | 0.09 | 0.18 | 0.27 | 0.40 | 0.44 | 0.47 | 0.57 | 0.81 | **0.97** |  |  |  |  |  |
| rel (relative citations) | 0.00 | 0.01 | 0.01 | 0.03 | 0.20 | 0.45 | 0.60 | 0.70 | 0.72 | 0.91 | **1.00** |  |  |  |  |  |  |  |
| nope (no positional encoding) | 0.00 | 0.00 | 0.00 | 0.02 | 0.10 | 0.26 | 0.35 | 0.46 | 0.54 | 0.63 | 0.74 | **0.94** |  |  |  |  |  |  |
| pos-line (line + offset rotary) | 0.00 | 0.00 | 0.01 | 0.04 | 0.15 | 0.21 | 0.33 | 0.43 | 0.38 | 0.44 | 0.59 | 0.83 | **0.99** |  |  |  |  |  |

![hazard](fig_hazard.png)


## 6. In-distribution checks

| variant | held-out greedy after pretraining (039 slice) | held-out greedy after RL (harness) | pass@1 L6-11 pooled (std) | pass@1 L6-11 pooled (matched) |
|---|---|---|---|---|
| abs (baseline) | 98.92% | 98.60% | 47.7% [46.3%, 48.6%] | 29.9% |
| rel (relative citations) | 98.95% | 98.70% | 28.0% [27.5%, 28.7%] | 19.1% |
| nope (no positional encoding) | 98.48% | 98.80% | 47.2% [45.3%, 48.7%] | 26.4% |
| pos-line (line + offset rotary) | 98.73% | 98.53% | 46.3% [45.0%, 47.1%] | 27.8% |

## 7. Training side

| run | status | rounds | samples | rl secs | compute used | stop | solved at the end, by L_true |
|---|---|---|---|---|---|---|---|
| abs_s0 | ok | 59 | 755,168 | 2044 | 0.997 | compute | L6 0.98, L7 0.92, L8 0.93, L9 0.62, L10 0.43, L11 0.61 |
| abs_s1 | ok | 59 | 748,544 | 1802 | 0.996 | compute | L6 0.97, L7 0.86, L8 0.89, L9 0.58, L10 0.39, L11 0.60 |
| abs_s2 | ok | 58 | 735,744 | 1837 | 0.996 | compute | L6 0.97, L7 0.88, L8 0.91, L9 0.68, L10 0.45, L11 0.63 |
| rel_s0 | ok | 71 | 900,608 | 1811 | 0.996 | compute | L6 0.95, L7 0.66, L8 0.49, L9 0.15, L10 0.01, L11 0.00 |
| rel_s1 | ok | 70 | 895,104 | 1468 | 0.997 | compute | L6 0.95, L7 0.66, L8 0.53, L9 0.16, L10 0.03, L11 0.00 |
| rel_s2 | ok | 70 | 888,576 | 1473 | 0.996 | compute | L6 0.96, L7 0.71, L8 0.53, L9 0.15, L10 0.02, L11 0.00 |
| nope_s0 | ok | 60 | 768,000 | 1297 | 0.993 | compute | L6 0.97, L7 0.88, L8 0.90, L9 0.61, L10 0.40, L11 0.59 |
| nope_s1 | ok | 60 | 768,000 | 1912 | 0.992 | compute | L6 0.96, L7 0.85, L8 0.88, L9 0.61, L10 0.42, L11 0.60 |
| nope_s2 | ok | 60 | 768,000 | 1789 | 0.995 | compute | L6 0.96, L7 0.76, L8 0.88, L9 0.60, L10 0.40, L11 0.60 |
| posline_s0 | ok | 61 | 780,800 | 1569 | 0.993 | compute | L6 0.97, L7 0.82, L8 0.88, L9 0.56, L10 0.38, L11 0.56 |
| posline_s1 | ok | 60 | 768,000 | 2180 | 0.993 | compute | L6 0.97, L7 0.81, L8 0.88, L9 0.59, L10 0.39, L11 0.59 |
| posline_s2 | ok | 61 | 774,144 | 1838 | 0.996 | compute | L6 0.98, L7 0.90, L8 0.90, L9 0.58, L10 0.41, L11 0.58 |

## 8. rel: citation-distance coverage and the long-length split

Citation distance k = citing line index - cited line index (rel writes it as the token B<k>). Counts of citations by k in the pretraining data and in the proofs each rel run trained on (abs runs for comparison: the same distances written with N<j> tokens):

| source | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 |
|---|---|---|---|---|---|---|---|---|---|---|
| pretraining (cap-6, all 154,990 records) | 285125 | 165891 | 60364 | 14803 | 8322 | 0 | 0 | 0 | 0 | 0 |
| RL-trained proofs, abs_s0 (distinct found proofs <= 11 lines) | 24537 | 6974 | 6704 | 5460 | 4175 | 2886 | 1198 | 784 | 126 | 2 |
| RL-trained proofs, abs_s1 (distinct found proofs <= 11 lines) | 25124 | 8173 | 6809 | 5982 | 4119 | 3188 | 1282 | 838 | 132 | 0 |
| RL-trained proofs, abs_s2 (distinct found proofs <= 11 lines) | 28762 | 9182 | 7596 | 6893 | 5075 | 3622 | 1805 | 963 | 193 | 4 |
| RL-trained proofs, rel_s0 (distinct found proofs <= 11 lines) | 8280 | 2639 | 3096 | 1558 | 1379 | 0 | 0 | 0 | 0 | 0 |
| RL-trained proofs, rel_s1 (distinct found proofs <= 11 lines) | 8538 | 2639 | 3302 | 1509 | 1434 | 0 | 0 | 0 | 0 | 0 |
| RL-trained proofs, rel_s2 (distinct found proofs <= 11 lines) | 8986 | 3275 | 3384 | 1750 | 1610 | 0 | 0 | 0 | 0 | 0 |

Trained range per rel run (largest k seen at least once in pretraining or its RL-trained proofs): seed 0: B5, seed 1: B5, seed 2: B5. The split uses K = B5 (the smallest).

Theorems at L12-18 split by the largest citation distance their reference proof needs (minlen_proof, else constructed_proof; one reference proof per theorem, so "needs" means this proof needs it). Within = <= 5 (1 theorems), beyond = > 5 (1834 theorems). Every variant is scored on the same split, so abs is the control for the difficulty of the "beyond" theorems.

| variant | pass@1, needs <= B5 | pass@1, needs > B5 |
|---|---|---|
| abs (baseline) | 0.00% | 1.68% |
| rel (relative citations) | 0.00% | 0.00% |
| nope (no positional encoding) | 0.00% | 1.63% |
| pos-line (line + offset rotary) | 0.00% | 0.69% |

The same theorems in bins of the largest citation distance their reference proof needs (pass@1, L12-18 pooled):

| variant | needs B1-B5 (1 thm) | needs B6 (15 thm) | needs B7-B8 (181 thm) | needs B9-B10 (686 thm) | needs B11-B64 (952 thm) |
|---|---|---|---|---|---|
| abs (baseline) | 0.00% | 0.00% | 8.78% | 2.17% | 0.00% |
| rel (relative citations) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| nope (no positional encoding) | 0.00% | 0.00% | 8.78% | 2.05% | 0.00% |
| pos-line (line + offset rotary) | 0.00% | 0.00% | 4.71% | 0.61% | 0.00% |

What the first error is (incremental checker reason codes) for abs and rel at L12-18:

| variant | first-error kinds at L12-18 (share of failed samples) |
|---|---|
| abs (baseline) | bad_line:rule 76.2%, bad_line:cite 12.2%, bad_line:parse 4.4%, bad_line:premise 3.3%, bad_line:range 2.3%, early_qed:not_conclusion 1.0% |
| rel (relative citations) | bad_line:rule 85.0%, bad_line:cite 6.1%, bad_line:parse 3.9%, bad_line:premise 3.8%, early_qed:not_conclusion 0.7%, bad_line:depth 0.3% |

rel failed samples at L12-18: share whose first error is a bad line, and share of those first bad lines that cite beyond the run's trained range:

| run | failed samples | bad line | first bad line cites > trained range |
|---|---|---|---|
| rel_s0 | 62656 | 99.1% | 0.01% |
| rel_s1 | 62656 | 99.5% | 0.00% |
| rel_s2 | 62656 | 99.1% | 0.00% |

Citation distances emitted by rel samples at L12-18 (all lines of all samples), share beyond the run's trained range: rel_s0 0.00%, rel_s1 0.00%, rel_s2 0.00%

![rel split](fig_rel_split.png)

## 9. pos-rand: evaluation on random positions

Not run.

## 10. Matched-construction control (pass@1)

| variant | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|
| abs (baseline) | 38.6% [37.2%, 40.5%] | 40.2% [38.0%, 42.9%] | 37.0% [32.7%, 40.2%] | 27.6% [24.3%, 33.6%] | 20.5% [17.2%, 22.8%] | 15.4% [14.8%, 16.4%] | 6.8% [6.3%, 7.2%] | 0.8% [0.7%, 0.8%] | 0.0% [0.0%, 0.0%] |
| rel (relative citations) | 34.0% [31.5%, 37.8%] | 31.9% [28.7%, 33.8%] | 24.2% [22.0%, 26.9%] | 15.5% [12.7%, 17.3%] | 7.2% [5.4%, 8.4%] | 1.9% [1.0%, 2.7%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |
| nope (no positional encoding) | 36.2% [34.9%, 37.3%] | 34.0% [30.9%, 37.2%] | 29.7% [27.3%, 34.1%] | 26.0% [25.0%, 27.6%] | 18.5% [17.3%, 20.1%] | 13.9% [13.2%, 15.0%] | 6.3% [5.8%, 7.1%] | 0.5% [0.2%, 0.9%] | 0.0% [0.0%, 0.0%] |
| pos-line (line + offset rotary) | 38.5% [36.9%, 40.6%] | 39.0% [35.6%, 40.9%] | 34.2% [30.4%, 38.1%] | 25.4% [22.6%, 30.8%] | 18.2% [17.5%, 19.2%] | 11.3% [10.0%, 12.2%] | 1.8% [1.6%, 2.0%] | 0.0% [0.0%, 0.1%] | 0.0% [0.0%, 0.0%] |

## Checks

- Theorem draw: every evaluated unit has one theorem list (unit_sha) across all labels: yes. Test sets = repr/testsets (frozen copy of lengthexp's final draw, 2026-09-25 02:27 UTC). Reference arm C units on another draw dropped: 0.
- Incremental checker vs nd_verify disagreements over 2,479,872 samples: none.
- Samples that hit max_new: 6,731.
