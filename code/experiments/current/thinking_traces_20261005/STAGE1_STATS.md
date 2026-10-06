---
written_on: 2026-10-06
written_by: agent:claude
---

# Stage-1 results with uncertainty

Scope: the Stage-1 reads in the
[summary](../../../../experiment-summaries/2026-10-05-thinking-traces/README.md#stage-1-results-2026-10-06-3-seeds-per-arm),
re-read with intervals, as [LIT_CRITIQUE.md](LIT_CRITIQUE.md) C4 asks. Everything here is CPU work on the stored evals in
`/scratch/cowifornia/nd-rl/think/runs/eval/` (branch `leon/thinking`, HEAD `b3c78cd`, with the uncommitted
[`code/thinking/stats.py`](../../../thinking/stats.py)). No model was run.

Labels:
- **Observed:** a number computed here from stored outputs.
- **Interpretation:** my reading, which may be wrong.
- A difference is called **resolved** only when both its bootstrap interval and its analytic t-check exclude 0
  (see Method for why both are needed).

Full generated tables, each headed by the exact command that produced it, are in [`stage1_stats/`](stage1_stats/):
- [tb72.md](stage1_stats/tb72.md) and [tb72_seeds_only.md](stage1_stats/tb72_seeds_only.md);
- [tb72_null_T0e.md](stage1_stats/tb72_null_T0e.md);
- [ood_clustered.md](stage1_stats/ood_clustered.md) and [ood_theorems.md](stage1_stats/ood_theorems.md);
- [ood_charles.md](stage1_stats/ood_charles.md) and [ood_textbook_dev.md](stage1_stats/ood_textbook_dev.md).

## Method

**Estimator.** For each arm, seed and theorem, the unbiased pass@k from n attempts with c accepted is
1 − C(n−c, k)/C(n, k). The arm value is its mean over seeds and theorems. textbook72 has n = 256; the OOD reads
have n = 16. The per-seed means match the evaluator's `summary.json` exactly (a regression test checks this).

**Hierarchical bootstrap** (4,000 draws, percentile 95% intervals). Each draw resamples:
- seeds with replacement, independently per arm, because different arms are different models;
- theorems with replacement, one resample shared by every arm and seed, so differences are paired on theorems.

Reads of the *same checkpoints* also share the seed resample: B 0 vs B 256 of T1, for example, is detected from
`ckpt` in `summary.json`. On the OOD sets, the generated (family, knob) cells can instead be resampled as clusters.

**Tests.**
- Bootstrap p: twice the smaller tail mass at 0.
- A sign test over theorems of the seed-mean difference. It conditions on these seeds, so it ignores seed noise
  and is reported only as a descriptive count.

**Analytic t-check.** A method-of-moments decomposition of the crossed seed × theorem model, with a Satterthwaite
t interval:

  Var = Σ σ²_seed / S + Var_t(d_t) / T

The theorem term is cluster-robust when clusters are given. The power calculations below rest on the same
decomposition.

**Why both intervals.** With 3 seeds, the seed level has 10 distinct resamples, and a bootstrap understates
seed-shift variance by a factor 2/3. It also double-counts the seed × theorem residual. I checked calibration in
an ad-hoc simulation (not committed): two arms from the same process, built on T0's per-theorem rates on the logit
scale, 3 seeds, 72 theorems, 300 replicates.

| regime | bootstrap covers 0 | t-check covers 0 |
|---|---|---|
| residual dominates | 97–98% | 96% |
| uniform seed shift dominates | 90% | 95% |
| 6 seeds | 98% | 94% |

On textbook72 at pass@1 the two components are comparable: pooled seed-shift variance 1.85 pp² against
residual/T 1.71 pp². The bootstrap SE is the larger of the two in most comparisons here.

**Matched tokens.**
- Tokens counted: sampled tokens per theorem, i.e. think + `</think>` + action + rejected redraws; prefill is
  excluded, as in `figures.py tokens`.
- For each seed, k* = X / (tokens per attempt), and pass@k* is interpolated linearly in log k between integer k.
- Tokens per attempt are recomputed on every theorem resample.
- When k* falls outside [1, n], the value is clamped and flagged (†: more than 5% of draws clamped).

The README's numbers interpolated the 3-seed mean curve on k = 1, 2, 4, …. This module reproduces them exactly
for T0 and T1 with `grid="pow2"`, and to within 0.3 pp for T1v1.

**Null check on real data.** T0e never thinks, and seeds 1 and 2 were read twice, at B 0 and at B 256
([tb72_null_T0e.md](stage1_stats/tb72_null_T0e.md)).
- **Observed:** B 256 − B 0 at pass@1 / 5 / 256 is:
  - −0.2 [−0.5, +0.1];
  - −0.6 [−1.3, +0.1];
  - +2.1 [−5.6, +8.3] pp.

  All three cover 0.
- At 1k matched tokens the difference is −1.1 [−2.2, −0.1] pp (p 0.04). B 256 samples its own `</think>` tokens:
  124 tokens per attempt against 116, so part of this is real token cost; it is one of 9 tests.

## 1. textbook72: pass@1 and pass@5 per arm (observed)

72 theorems (58 dev + 14 train), 3 seeds per arm, k 256, T 0.8.
- **Main brackets** resample seeds and theorems. They describe the population of theorems like these.
- **(Seeds only)** resamples seeds and keeps these 72 theorems fixed. It describes how reproducible the number is
  on textbook72 itself.

| arm | pass@1 | (seeds only) | pass@5 | (seeds only) | pass@256 | tokens / attempt |
|---|---|---|---|---|---|---|
| T0, no think | 17.2% [10.4, 25.1] | [15.4, 18.6] | 26.0% [17.6, 35.2] | [25.2, 27.0] | 45.4% [34.7, 55.6] | 116 |
| T0e, empty think | 13.6% [7.4, 20.9] | [10.9, 15.0] | 20.7% [13.0, 29.0] | [18.7, 22.5] | 39.4% [28.7, 50.0] | 113 |
| T1 (v2), B 0 | 13.1% [7.0, 20.1] | [11.7, 15.4] | 20.7% [12.5, 29.9] | [17.1, 25.0] | 38.9% [27.8, 50.0] | 129 |
| T1 (v2), B 256 | 15.6% [9.5, 22.6] | [14.5, 17.1] | 27.9% [18.9, 37.4] | [24.7, 31.1] | 51.4% [39.4, 62.5] | 320 |
| T1v1 (v1), B 0 | 13.1% [7.0, 20.2] | [12.6, 13.5] | 20.2% [12.5, 28.9] | [19.3, 21.5] | 36.1% [25.5, 46.8] | 125 |
| T1v1 (v1), B 256 | 16.8% [10.2, 24.1] | [14.9, 19.1] | 29.3% [20.2, 38.7] | [27.3, 33.3] | 56.0% [44.4, 67.1] | 420 |

Theorem difficulty dominates the arm-level intervals; most problems are near 0% or near 100%. **Arms must not be
compared by whether these intervals overlap.** Use the paired differences below.

## 2. Paired differences against T0 (observed, pp)

Same 72 theorems, 3 vs 3 independent seeds. Format: diff [bootstrap 95%], boot p, then the analytic t interval.

| difference | pass@1 | pass@5 | pass@256 |
|---|---|---|---|
| T1 B 256 − T0 | −1.6 [−5.0, +1.5], p 0.33; t [−4.1, +0.9] | +1.9 [−3.2, +7.0], p 0.47; t [−3.6, +7.4] | +6.0 [−0.9, +13.4], p 0.12; t [−0.3, +12.3] |
| T1v1 B 256 − T0 | −0.4 [−4.0, +3.3], p 0.81; t [−3.9, +3.1] | +3.4 [−2.1, +9.2], p 0.23; t [−2.4, +9.2] | **+10.6 [+3.7, +18.1], p 0.002; t [+4.4, +16.9]** |
| T1 B 0 − T0 | **−4.1 [−8.1, −0.4], p 0.03; t [−7.4, −0.7]** | −5.2 [−11.3, +0.4], p 0.07; t [−11.9, +1.5] | −6.5 [−13.9, +0.0], p 0.07; t [−12.1, −0.8] |
| T1v1 B 0 − T0 | **−4.1 [−7.7, −0.9], p 0.015; t [−6.7, −1.5]** | **−5.8 [−10.4, −1.7], p 0.005; t [−9.2, −2.4]** | **−9.3 [−16.2, −2.3], p 0.011; t [−14.6, −4.0]** |
| T0e − T0 | −3.6 [−8.5, +0.7], p 0.10; t [−7.4, +0.2] | **−5.3 [−10.7, −0.4], p 0.03; t [−9.4, −1.2]** | −6.0 [−13.9, +1.4], p 0.12; t [−11.6, −0.4] |

Bold: resolved.

**Within-model and other comparisons:**

| difference | pass@1 | pass@5 | pass@256 |
|---|---|---|---|
| T1: B 256 − B 0 (paired seeds) | **+2.5 [+0.8, +4.3]; t [+1.0, +4.0]** | **+7.1 [+3.5, +10.8]; t [+3.8, +10.5]** | **+12.5 [+5.6, +20.4]; t [+5.7, +19.3]** |
| T1v1: B 256 − B 0 (paired seeds) | +3.7 [+1.0, +7.2]; t [−0.6, +8.0] (df 4.9) | **+9.2 [+4.3, +15.0]; t [+2.7, +15.7]** | **+19.9 [+11.6, +29.2]; t [+12.5, +27.3]** |
| T1v1 B 256 − T1 B 256 | +1.2 [−1.8, +4.2]; t [−2.4, +4.7] | +1.5 [−4.0, +7.1]; t [−5.1, +8.0] | +4.6 [−2.8, +12.5]; t [−2.5, +11.7] |
| T1 B 0 − T0e | −0.5 [−4.4, +3.9]; t [−4.4, +3.4] | +0.1 [−5.6, +6.2]; t [−6.6, +6.8] | −0.5 [−8.3, +8.3]; t [−6.8, +5.9] |
| T1v1 B 0 − T0e | −0.5 [−3.6, +3.4]; t [−3.7, +2.7] | −0.5 [−5.0, +4.5]; t [−4.0, +3.0] | −3.2 [−11.1, +4.2]; t [−8.4, +1.9] |

These two intervals reproduce LIT_CRITIQUE C1's within about 0.4 pp; the critique's bootstrap was a different
ad-hoc run of the same design:
- T1 B 256 − T0 at pass@1: −5.0 to +1.5 here, against −4.8 to +1.6.
- T1v1 B 256 − T0 at pass@256: +3.7 to +18.1 here, against +3.7 to +18.5.

## 3. Matched sampled tokens per theorem (observed)

textbook72, 3 seeds; pass@k at X sampled tokens per theorem.

| arm | 1k | 3k | 10k | 29k |
|---|---|---|---|---|
| T0 | 28.8% [19.9, 38.4] | 34.4% [24.7, 44.8] | 40.2% [29.7, 50.5] | 45.2% [34.2, 55.6]† |
| T1 B 256 | 24.1% [15.7, 33.5] | 32.8% [22.8, 43.1] | 41.7% [30.9, 52.2] | 47.5% [36.1, 58.4] |
| T1v1 B 256 | 23.8% [15.4, 32.9] | 32.6% [22.6, 42.7] | 42.5% [31.8, 52.9] | 49.7% [38.4, 60.2] |

| difference (pp) | 1k | 3k | 10k | 29k |
|---|---|---|---|---|
| T1 B 256 − T0 | −4.7 [−9.9, +0.3], p 0.06 | −1.6 [−7.4, +4.0], p 0.55 | +1.5 [−4.1, +7.1], p 0.61 | +2.3 [−4.2, +9.2]†, p 0.48 |
| T1v1 B 256 − T0 | −5.0 [−10.2, −0.3], p 0.04 | −1.8 [−8.1, +3.6], p 0.53 | +2.3 [−4.0, +8.2], p 0.45 | +4.5 [−1.9, +11.1]†, p 0.16 |
| T1: B 256 − B 0 | +1.3 [−2.4, +4.6], p 0.45 | +4.9 [+0.5, +9.2], p 0.03 | +7.8 [+2.7, +13.3], p 0.006 | +9.1 [+3.0, +16.0], p 0.004 |
| T1v1: B 256 − B 0 | +1.5 [−2.3, +5.2], p 0.41 | +5.4 [+0.4, +10.4], p 0.035 | +10.4 [+3.9, +17.0], p 0.002 | +13.8 [+5.6, +22.4]†, p 0.002 |
| T1v1 − T1, both B 256 | −0.4 [−4.8, +3.5] | −0.2 [−6.2, +5.2] | +0.8 [−5.7, +6.8] | +2.2 [−4.1, +8.8] |

† At 29k, T0 is at the end of its read: 256 × 116 tokens = 29.7k. In 43% of draws its k* exceeds 256 and is
clamped to pass@256. That understates T0, so the 29k column slightly favours the thinking arms. There is no
analytic t-check for matched tokens.

## 4. OOD sets (observed)

**Design.**
- k 16, plain sampling unless noted. textbook_dev is removed, leaving 840 theorems.
- The generated sets' (family, knob) cells are resampled as clusters: 18 reorder/commute cells + 1 case-product
  cell + 80 Charles theorems = 99 clusters. Charles's set has no knob, so each of its theorems is its own unit.
- Seeds:
  - 2 (s0, s1) for T0, T1 at B 0 / 64 / 256, and T0 guided (logical redraws);
  - 1 (s0) for T0e, T1v1 B 256 and T1 guided + B 256.

  With 2 seeds the seed variance has 1 df and the bootstrap has 3 distinct seed resamples. These intervals are
  therefore essentially conditional on these two models.
- Single-seed arms get theorem-level intervals only.

| arm | seeds | pass@1 | pass@16 | tokens / attempt |
|---|---|---|---|---|
| T0 | 2 | 1.8% [0.6, 3.4] | 5.1% [2.0, 9.4] | 140 |
| T1 B 0 | 2 | 1.1% [0.2, 2.3] | 4.3% [1.2, 8.5] | 159 |
| T1 B 256 | 2 | 1.3% [0.3, 2.7] | 5.1% [1.7, 9.8] | 370 |
| T0 guided | 2 | 4.1% [1.7, 7.3] | 11.7% [5.3, 19.9] | 426 |
| T1v1 B 256 | 1 | 1.6% [0.4, 3.1] | 6.2% [2.5, 11.2] | 634 |
| T1 guided + B 256 | 1 | 4.1% [1.4, 7.7] | 15.5% [6.9, 26.8] | 1328 |

Paired differences (pp), clustered; format: diff [bootstrap], then the cluster-robust t interval:

| difference | pass@1 | pass@5 | pass@16 |
|---|---|---|---|
| T1 B 256 − T0 | **−0.5 [−1.1, −0.1]; t [−0.9, −0.1]** | −0.4 [−1.2, +0.2]; t [−1.7, +0.8] | −0.1 [−1.8, +1.8]; t [−8.3, +8.2] |
| T1 B 0 − T0 | **−0.7 [−1.4, −0.2]; t [−1.3, −0.2]** | **−0.9 [−1.8, −0.2]; t [−1.5, −0.3]** | −0.8 [−2.2, +0.4]; t [−1.7, +0.0] |
| T1: B 256 − B 0 (paired) | **+0.2 [+0.0, +0.5]; t [+0.0, +0.4]** | **+0.5 [+0.0, +1.1]; t [+0.1, +0.9]** | +0.8 [−0.4, +2.3]; t [−1.2, +2.7] |
| T0 guided − T0 (paired) | **+2.3 [+1.0, +3.9]; t [+0.8, +3.7]** | **+4.5 [+2.0, +7.8]; t [+1.7, +7.3]** | **+6.5 [+3.1, +11.0]; t [+2.7, +10.4]** |
| T1 B 256 − T0 guided | **−2.8 [−4.8, −1.2]; t [−4.5, −1.0]** | **−4.9 [−8.7, −2.2]; t [−8.0, −1.8]** | **−6.6 [−11.7, −2.9]; t [−11.1, −2.1]** |
| T1v1 B 256 (1 seed) − T0 | −0.2 [−0.8, +0.3] | +0.4 [−0.4, +1.3] | +1.1 [−0.1, +2.5] |

Matched tokens (clustered; k 16 caps T0 at 16 × 140 ≈ 2.2k tokens per theorem):

| difference (pp) | 500 | 1k | 2k | 4k |
|---|---|---|---|---|
| T1 B 256 − T0 | **−1.7 [−3.0, −0.6]** | **−1.7 [−3.1, −0.7]** | **−1.7 [−3.3, −0.5]** | – |
| T0 guided − T0 | +1.2 [+0.4, +2.3] | +2.0 [+0.8, +3.6] | +3.0 [+1.3, +5.4] | – |
| T1 B 256 − T0 guided | −2.8 [−5.2, −1.2] | −3.7 [−6.7, −1.6] | −4.7 [−8.5, −2.0] | −5.7 [−10.5, −2.4] |
| T1: B 256 − B 0 | −0.6 [−1.4, −0.1] | −0.6 [−1.7, +0.0] | −0.6 [−1.9, +0.3] | – |

At these points:
- T0 is at 3.3 / 4.1 / 5.0%.
- T1 B 256 is at 1.6 / 2.4 / 3.3 / 4.4%.
- T0 guided is at 4.4 / 6.0 / 8.0 / 10.1%.

Two notes on this table:
- The bold T1 B 256 − B 0 cells are resolved barely: their lower ends are +0.04 to +0.08 pp.
- Resampling single theorems instead of clusters ([ood_theorems.md](stage1_stats/ood_theorems.md)) narrows the
  per-arm intervals by about half. It flips only borderline calls, each with an interval end within 0.1 pp of 0:
  - T1 B 256 − B 0 at pass@5;
  - two differences at 1k tokens.

The T1 B 64 and T0e rows are in the generated tables.

**Charles's long problems** (Tier A, n = 80, theorems resampled; [ood_charles.md](stage1_stats/ood_charles.md)):

| arm | pass@1 | pass@16 |
|---|---|---|
| T0 | 1.4% [0.1, 3.3] | 5.6% [1.2, 11.2] |
| T1 B 256 | 0.5% [0.0, 1.5] | 3.8% [0.0, 10.0] |
| T0 guided | 4.9% [1.8, 8.6] | 16.2% [7.5, 26.2] |

Differences:
- T1 B 256 − T0 at pass@1: −0.8 [−2.5, +0.2].
- T0 guided − T0 at pass@16: +10.6 [+2.5, +20.0]; its t-check is [−7.7, +28.9] (df 2.3), so this is not resolved.
- T1 B 256 − T0 guided at 1k tokens: −6.5 [−11.2, −2.6].

## 5. How many theorems carry information (observed)

A theorem is uninformative for these comparisons if it sits at 0% (or 100%) in every seed of every arm read.

| set | n | 0% everywhere | 100% everywhere (pass@n) | informative at pass@1 / 5 | informative at pass@n |
|---|---|---|---|---|---|
| textbook72 (6 cells × 3 seeds) | 72 | 24 | 14 | 48 | 34 |
| OOD without textbook_dev (8 cells, incl. guided) | 840 | 658 | 14 | 182 | 168 |
| Charles's long (8 cells) | 80 | 55 | 1 | 25 | 24 |

No theorem is at 100% pass@1 in every cell.

## 6. Power (observed numbers, model-based projections)

Components are pooled over the four independent textbook72 models (T0, T0e, T1, T1v1), at pass@1:
- seed-shift SD 1.36 pp;
- seed × theorem residual SD 11.1 pp per (seed, theorem) cell;
- SD of the true per-theorem difference between arms 4.4 pp (averaged over the three comparisons with T0;
  0.9–7.5 pp individually).

The variance of an arm difference at 3 seeds and 72 theorems is about:
- 47% seed shift;
- 43% seed × theorem interaction;
- 10% theorem sampling.

More seeds shrink the first two. More theorems shrink only the last two.

**Smallest detectable difference** (80% power, two-sided α 0.05):

| design | pass@1 | pass@5 | pass@256 |
|---|---|---|---|
| textbook72, 3 vs 3 independent seeds, pooled, z | 4.5 pp | 7.0 pp | 7.4–8.5 pp (per comparison) |
| same, t with the 4–8 seed df actually available | about 5–6 pp | about 8–9 pp | – |
| same, per-comparison bootstrap SEs | 4.7–6.5 pp | – | – |
| textbook72, same checkpoints B 0 vs B 256 (paired seeds) | 2.2 pp (T1), 4.6 pp (T1v1) | 4.7 / 7.8 pp | 9.5 / 10.3 pp |
| OOD 840, 2 seeds, clustered | 0.6–1.0 pp plain vs plain; 2.1–2.5 pp with a guided arm | 0.8–1.6 pp; 4.0–4.4 with guided | 1.3–3.3 pp; 5.5–6.1 with guided |
| Charles's long, 80 theorems, 2 seeds | 1.8 pp | 4.0 pp | 8.0 pp |

OOD pass@1 base rates are 1–4%, so the OOD detection limits are 30–50% of the base rate. On Charles's set only
effects larger than T0's own pass@1 (1.4%) are detectable.

**To detect 3 pp at pass@1 on textbook72** (z approximation; slightly more with t):
- about **9 seeds per arm** at 72 theorems, or 6 seeds per arm at 144 theorems;
- 7 seeds per arm if the question is only about these 72 theorems (seeds-only resampling);
- with 3 seeds, no number of theorems suffices: the seed-shift term alone leaves a 3.1 pp floor.

LIT_CRITIQUE C4 estimated about 6 seeds. The difference is that this decomposition also counts the seed ×
theorem interaction, which more theorems only partly average out. At pass@5, 3 pp needs both more seeds and more
theorems: about 21 seeds at 144 theorems.

**Seed-level tests have a hard floor.** An exact permutation test that treats seeds as the unit cannot reach
p < 0.05 with 3 vs 3 seeds: 20 splits give a minimum two-sided p of 0.10. 4 vs 4 is the minimum for 0.05
(70 splits, minimum p 0.029).

## 7. What this changes in the Stage-1 reading

**Observed (resolved means both intervals exclude 0):**
1. **Thinking is used, within a model.** For T1, B 256 beats B 0 at pass@1, pass@5 and pass@256. For T1v1 it is
   resolved at pass@5 and pass@256; at pass@1 only the bootstrap excludes 0, the t-check does not (df 4.9).

   At matched tokens, B 256 beats B 0 from 3k up (bootstrap only). LIT_CRITIQUE C3's confound still applies: B 0
   mostly breaks the action syntax.
2. **Against T0, no thinking arm is resolved at pass@1 or pass@5.** The largest point difference is T1v1 at
   pass@5, +3.4 [−2.1, +9.2]. At pass@256, T1v1 B 256 is resolved (+10.6); T1 B 256 is not (+6.0, t [−0.3, +12.3]).
3. **Matched tokens against T0.** Nothing is resolved at 3k, 10k or 29k. At 1k, T1v1 is behind (−5.0, bootstrap
   p 0.04) and T1 is borderline (−4.7, p 0.06). The README's "level around 5k, 2.2–4.5 pp ahead at 29k" is within
   noise.
4. **"Trace training costs the no-think policy 6–9 pp" cannot be separated from the empty-think format.**
   - Both B 0 arms are 4.1 pp below T0 at pass@1 (resolved).
   - T0e sits 3.6 pp below T0 (not resolved at pass@1; resolved at pass@5).
   - T1 B 0 and T1v1 B 0 are 0.5 pp from T0e, with intervals of about ±4 pp.
5. **v1 against v2 is unresolved** at every k and every token budget. The largest gap is +4.6 [−2.8, +12.5] at
   pass@256.
6. **OOD** (2 seeds; conditional on these models):
   - Plain thinking (T1 B 256) is 0.5 pp below T0 at pass@1, and 1.7 pp below at matched 0.5–2k tokens (both
     resolved). That gap is 35–50% of T0's rate at those budgets.
   - Within T1, B 256 beats B 0 by 0.2 pp at pass@1 (resolved barely), but loses 0.6 pp at matched tokens.
   - Guided redraws beat plain T0 (+2.3 pp at pass@1) and beat T1 B 256 at every k and token budget.
   - The README's "thinking is worth about as much as more samples" holds only at equal k (pass@16: −0.1 pp). At
     equal tokens, thinking is worth less than more samples.

**Interpretation:**
- The Stage-1 evidence that the models depend on their think block is solid; it comes from within-model
  comparisons.
- The test-time-scaling claim (thinking beats sampling from T0 at matched tokens) has no support at any budget at
  Stage 1. The large-budget lead and the crossover are well inside the noise.
- Which baseline is right, T0 or T0e, matters as much as the traces. The T0 − T0e gap (3.6 pp at pass@1) is the
  same size as every thinking effect against T0. More seeds of both, as C3 and C4 already ask, would settle it.
- For the RL phase the design consequence is:
  - pre-register within-model comparisons where possible (MDE about 2 pp);
  - otherwise plan for about 9 seeds per arm on textbook72 to resolve 3 pp at pass@1, or move pass@1 comparisons
    to a larger set with pass@1 in the 10–60% range;
  - a single-seed RL arm against a 3-seed baseline resolves only about 6.3 pp or more at pass@1 on textbook72,
    and two single-seed arms about 7.6 pp. This uses the pooled seed variance; one seed cannot estimate its own.

## Reproduce

```bash
cd /scratch/cowifornia/nd-rl/think/nd-rl
code/.venv/bin/python -m code.thinking.stats table \
  --group "T0=DIR/tb72_T0_s0,DIR/tb72_T0_s1,DIR/tb72_T0_s2" \
  --group "T1 B256=DIR/tb72_T1_s0/rows_plain_B256.jsonl,...,DIR/tb72_T1_s2/rows_plain_B256.jsonl" \
  --pair "T1 B256 vs T1 B0" --k 1,5,n --tokens 1000,3000,10000,29000 --out FILE.md --json FILE.json
# OOD: add --exclude textbook_dev --clusters SETS/ood_{or_reorder,and_reorder,commute,case_product}.jsonl
#      --cluster-by family,knob
# figures with shaded 95% bands: python -m code.thinking.figures tokens|lengthbin ... --ci
```

The exact commands are at the top of each file in [`stage1_stats/`](stage1_stats/). The pooled power numbers come
from `stats.pooled_components([T0, T0e, T1 B256, T1v1 B256], [(T0e, T0), (T1 B256, T0), (T1v1 B256, T0)], k)`
together with `mde`, `seeds_needed` and `theorems_needed`.

**Checks run:**
- `pytest code/tests/thinking/`: all pass. 21 tests in `test_stats.py` cover:
  - coverage of a known p;
  - identical arms giving a 0-centred interval;
  - pairing;
  - the matched-token interpolation against `figures.tokens_data`;
  - clusters, the t distribution and the variance components;
  - the CLI and the `--ci` figure bands.
- `ruff check`, `ruff format`, and `pyright` on `thinking/` and `tests/thinking/`: 0 errors.
