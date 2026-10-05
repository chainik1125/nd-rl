---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - brown2024monkeys
---

# Large Language Monkeys: coverage under repeated sampling

Paper: [@brown2024monkeys]
Source: arXiv 2407.21787v3 (HTML rendering of v3 read: abstract, Sec. 1-3, Sec. 5, App. C, D)

## Learnings

- **Coverage as the capability quantity.** Coverage is "the fraction of problems that are solved by any
  generated sample" and it "scales with the number of samples over four orders of magnitude" (Abstract).
  Coverage is computed with the Chen et al. unbiased estimator: "To reduce the variance when calculating
  coverage, we adopt the unbiased estimation formula from Chen et al." (Sec. 2, Eq. 1). With a perfect
  verifier (Lean, unit tests) coverage equals success rate (Sec. 2).
- **They state the random-weights objection themselves and answer it with a budget.** "With unlimited
  samples, any model that assigns a non-zero probability to every sequence will achieve perfect coverage.
  However, repeated sampling is only practical if we can improve coverage with a feasible budget"
  (Sec. 1). There is no principled definition of "feasible"; budgets are 10,000 samples (250 for
  SWE-bench) and FLOP/dollar comparisons (Sec. 2.3).
- **Large gains from k on weak models.** Gemma-2B on CodeContests: "from 0.02% with one sample to 7.1%
  with 10,000 samples" (Sec. 1); Pythia-160M on MATH: "from a pass@1 of 0.27% to a pass@10k of 57%"
  (Sec. 2.2). But "All Pythia models achieve zero coverage on this dataset [CodeContests], even with a
  budget of 10,000 samples" (Sec. 2.2) — a case where k does not rescue a model whose training lacked the
  skill.
- **Exponentiated power law in k.** "log(c) ≈ a k^b" (Sec. 3.1, Eq. 2), i.e. c ≈ exp(a k^b) (Eq. 3), fitted
  with SciPy curve_fit to 40 log-spaced points in [0, 10,000] (App. C.1). Fit quality is uneven: "these laws
  are not as exact as training scaling laws (most strikingly on MiniF2F-MATH)" (Sec. 3.1) — the Lean task.
- **Within a model family, a better model is a horizontal shift in log k.** "the traced S-curves have the
  same slope, but unique horizontal offsets"; "the increase in the log-sample-budget (or equivalently, the
  multiplicative increase in the sample budget) needed to improve coverage from c to c′ is approximately
  constant" (Sec. 3.2, Fig. 6). The shift is log(pass@k⁻¹(c)), with pass@k⁻¹(c) the closest k reaching c.
- **Compute matching across model sizes.** Plotting coverage against inference FLOPs, "On MiniF2F, GSM8K
  and MATH, Llama-3-8B-Instruct always obtains higher coverage than the larger (and more expensive) 70B
  model when the FLOP budget is fixed", while for CodeContests the 70B model wins (Sec. 2.3, Fig. 4).
- **New solves at large k are rare events.** "coverage improves through models generating correct
  solutions to problems they have not previously solved. However, these increasingly rare correct
  generations are only beneficial if verifiers can 'find the needle in the haystack'" (Sec. 1).

## Evidence and limitations

- Evidence: Fig. 2-6, Table 1. Fits are aggregate (dataset-level) curves; no per-problem model and no
  uncertainty on the fitted (a, b) beyond error summaries in Fig. 5.
- The power law is descriptive; the paper offers no reason for it (explained later by
  schaeffer2025powerlaws.md). Extrapolation beyond 10^4 is not tested.
- Coverage on GSM8K/MATH uses an oracle answer check, so at 10^4 samples some "coverage" may be answer
  guessing; the paper does not examine this (our observation; not relevant for Lean, where a proof is
  checked).
- Not checked: Sec. 4 (verification without oracles) in detail.

## Connections and questions

- **Definition offered:** capability at budget k = coverage(k) = pass@k averaged over problems, with an
  automatic verifier; summarised by the fitted curve c(k) = exp(a k^b). Practical capability is
  coverage "with a feasible budget" (Sec. 1), and budgets are compared in samples, FLOPs or dollars.
- **New vs better access:** not distinguished explicitly. But Sec. 3.2 gives a usable decision rule in
  embryo: if model B's coverage curve is model A's curve **shifted horizontally in log k** (same shape),
  then B ≈ A with a constant multiplicative sample advantage, i.e. better access (sample efficiency); if
  B's curve has a different shape or a higher asymptote than A's curve extrapolated, B can do something
  A cannot at any comparable k. The Pythia-on-CodeContests case (zero at 10^4) is the "absent capability"
  pattern.
- **Null / floor:** explicitly raised ("any model that assigns a non-zero probability to every sequence
  will achieve perfect coverage", Sec. 1) and answered only by appeal to a "feasible budget". A principled
  version for us: tie the budget to RL's own sampling budget or to training compute (see
  davidson2023retraining.md, jones2021boardgames.md).
- **Transfer to our setting:** (1) Plot coverage(k) for pend, r8, r16 on the ~320 eval theorems for
  k ≤ 512 (two seeds pooled) and fit c(k) = exp(a k^b); estimate the horizontal shift needed to overlay r16
  on pend (in log k). Report "r16 = pend at M× samples" with a bootstrap interval over theorems. (2) Check
  whether the shift is constant across coverage levels (pure sample-efficiency gain) or grows (r16 solving
  theorems that pend's curve never reaches). Cost: negligible (counts exist). Failure mode: k ≤ 512 is too
  short to see pend's asymptote; the aggregate fit hides per-theorem structure, and Brown et al. found the
  Lean fit the worst — use the per-theorem distribution approach of schaeffer2025powerlaws.md and the
  400,000-sample data for the older model to test the extrapolation.
- Related: chen2021evaluating.md, schaeffer2025powerlaws.md, kazdan2025passk.md, levi2024 (screened).
