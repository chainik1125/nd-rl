---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - wu2024rareoutputs
---

# Estimating the Probabilities of Rare Outputs in Language Models — low probability estimation

Paper: [@wu2024rareoutputs]
Source: arXiv 2410.13211v2 (HTML rendering read: abstract, Sec. 1-3.1, 5, 6)

## Learnings

- **Problem: low probability estimation.** "how can we estimate the probability of a binary property of the
  model's output, even when that probability is too small to estimate by random sampling?" (Abstract).
  Naive sampling with n draws is "almost always 0, making it uninformative at distinguishing between small
  probabilities like 10^-10 and 10^-20" (Sec. 2).
- **Setting.** Argmax (temperature-0) next-token behaviour of small transformers over formally specified
  input distributions with independent tokens; target tokens have "ground truth probabilities between 10^-9
  and 10^-5", with ground truth from brute-force sampling at a larger budget (Sec. 1, Fig. 1).
- **Importance sampling (IS).** Sample from a proposal q concentrated where the behaviour occurs and
  re-weight by p/q: "If we re-weight our observations properly, this gives an unbiased estimator for the true
  probability" (Sec. 3.1, equation in Sec. 3.1). Two variants: ITGIS (independent-token proposal tilted by
  gradients) and MHIS (Metropolis-Hastings) (Sec. 3.1.1-3.1.2).
- **Activation extrapolation.** Fit a distribution to logits from random samples and extrapolate into the
  tail (QLD, GLD) (Sec. 3.2).
- **Result.** "importance sampling outperforms activation extrapolation, but both outperform naive
  sampling" (Abstract; Sec. 5, Fig. 2).
- **Search = estimation when ratios are computable.** "As long as the required importance sampling ratios
  can be computed, any method for red-teaming can be turned into an importance sampling method for low
  probability estimation" (Sec. 6.2).
- **Where IS can fail.** For a hash-like model "finding any input that gives rise to a particular output is
  computationally infeasible", yet the probability is easy to model (Sec. 6.3).
- **Limitations.** "we only use input distributions that factor into independent tokens"; "we only study
  model behaviors that consist of a single token sampled at temperature 0" (Sec. 6.4).

## Evidence and limitations

- Evidence: Fig. 2-4, App. G-J (Itakura-Saito loss against ground truth). Randomness is over inputs; the
  model is deterministic. Long autoregressive behaviours are explicitly out of scope (Sec. 6.4).
- Not checked: Sec. 3.2 details, App. B-F.

## Connections and questions

- **Definition offered:** the quantity of interest is the probability of a formally specified binary
  behaviour under a specified distribution, estimated below the 1/n resolution of naive sampling by
  importance sampling or by tail extrapolation.
- **New vs better access:** not addressed, but the IS identity gives the most direct tool we found for
  measuring the base model's probability of exactly the proofs RL finds. Our transfer (not in the paper):
  with RL policy r16 as proposal q and pend as target p over **outputs**,
  p̂_pend,i = (1/N) Σ_{j: Lean accepts y_j} exp(log π_pend(y_j | x_i) − log π_r16(y_j | x_i)), y_j ~ r16.
  This is unbiased for pend's single-attempt success probability p_pend,i (the "ratios can be computed",
  Sec. 6.2, because both are our own models), it needs only the existing r16 samples plus two teacher-forced
  passes over the accepted proofs, and it reaches probabilities far below 1/256. A theorem where p̂_pend,i
  is, say, 10^-12 while r16 solves it at p ≈ 0.5 is a quantitative "RL made it accessible"; the size of the
  gap is a number, not a choice of k.
- **Null / floor:** the same estimator applied with the random-init model as target gives its (astronomically
  small) p, which answers "at some k even random weights solve it" by stating that k ≈ 1/p and comparing it
  to RL's budget.
- **Transfer to our setting:** cost is one forward pass of pend and r16 per accepted proof (cheap). Failure
  modes: (1) weight degeneracy — the estimate is dominated by the few proofs with the largest pend/r16 ratio,
  and its variance is large when r16 concentrates on proofs pend finds unlikely; report effective sample size
  and compare with the direct estimate where pend's n_ok > 0 (a built-in check). (2) Proofs that pend would
  produce but r16 never samples are missed (positivity holds in principle at temperature 1, but not under
  top-p/top-k truncation), so the IS estimate is a lower-biased estimate in practice. (3) Sampling settings
  must be the ones whose probabilities are computed (temperature, truncation).
- Related: jones2025forecasting.md (elicitation probabilities, IS proposed but not implemented),
  hu2023passuntil.md, kazdan2025passk.md, schaeffer2025powerlaws.md.
