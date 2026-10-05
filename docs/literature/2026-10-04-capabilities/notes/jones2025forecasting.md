---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - jones2025forecasting
---

# Forecasting Rare Language Model Behaviors — elicitation probabilities and Gumbel-tail extrapolation

Paper: [@jones2025forecasting]
Source: arXiv 2502.16797v1 (HTML rendering read: abstract, Sec. 1, 3, 4, 7, App. B.1)

## Learnings

- **Per-query probability as the continuous object behind binary outcomes.** "We define this as the
  elicitation probability of a query: the probability that a sampled output from a query has a specific
  behavior" (Sec. 3.1, Eq. 1). Motivation: "seemingly ineffective jailbreaks in fact elicit harmful outputs
  with non-zero probability under enough repeated sampling" (Sec. 1).
- **Scaling of the extreme quantiles across queries.** "the largest observed elicitation probabilities
  predictably scale with the number of queries" (Abstract); "the logarithm of the largest-quantile
  elicitation probabilities follows a power-law in the number of samples required to estimate them"
  (Sec. 1). Q_p(n) is the top-1/n quantile of the per-query probability distribution (Sec. 3.1, Eq. 2).
- **Gumbel-tail method.** Elicitation score ψ_i = −log(−log p_elicit(x_i)) (Sec. 3.3); for a Gumbel-type tail
  "the tail of the log survival function is an approximately linear function of the elicitation score",
  so Q_ψ(n) = −(1/a)(log n − b) (Sec. 3.3, Eq. 3-5). Fit by OLS "for the ten highest elicitation scores during
  evaluation" (Sec. 3.3). Motivation by analogy with pretraining scaling laws: "increasing n in expectation is
  like adding optimization steps" (App. B.1, Eq. 6).
- **Three risk metrics** built from the forecast quantiles: worst-query risk (max p over n queries),
  behavior frequency (fraction with p > τ), aggregate risk 1 − Π(1 − p_i) (Sec. 3.2).
- **Accuracy.** "when forecasting the risk at 90,000 samples using only 900 samples … our forecasts stay
  within one order of magnitude of the true risk for 86% of misuse forecasts" (Sec. 1); worst-query risk:
  "The average absolute log error is 1.7 for the Gumbel-tail method, compared to 2.4 for the log-normal
  method" (Sec. 4.2); the Gumbel-tail method "tends to underestimate the actual probability only 34% of the
  time, compared to 72% for the log-normal" (Sec. 4.2). Range: "up to three orders of magnitude of query
  volume" (Abstract).
- **Teacher-forced probability of a specific output as a cheap proxy — with a stated caveat.** Three
  proxies are used: probability of a specific harmful output, of a keyword, and of a sampled output
  containing the keyword. "Measuring the probability of a specific output is efficient—it can be done in a
  single forward pass—but may not reflect the actual likelihood of producing 'useful' instructions"
  (Sec. 4.1). Sec. 4.5 repeats the analysis with sampled elicitation probabilities (100-500 samples per query)
  and finds the quantiles "frequently qualitatively linear for large enough n" (Sec. 4.5, Fig. 5).
- **Rare-event cost reduction proposed, not implemented.** "we can adaptively stop sampling from queries
  that are unlikely to have the highest elicitation probabilities"; "We could also more efficiently compute
  probabilities via importance sampling" (Sec. 4.5).
- **On optimisation-based elicitation.** Prompt optimisation or fine-tuning to elicit a behaviour can give
  false positives: "optimizing can find instances of a behavior that are too rare to ever come up in
  practice" (Sec. 7).
- **Limitations stated.** "Since our forecast only uses the largest ten elicitation probabilities, the
  forecasts are sensitive to stochasticity in the specific evaluation set" (Sec. 3.3); no distribution
  shift studied (Sec. 7).

## Evidence and limitations

- Evidence: Fig. 1, 3-5; App. C, E. The extrapolation is across **queries** (inputs), not across samples
  per query; deployment sizes ≤ 3 orders of magnitude beyond evaluation (Sec. 7).
- Most experiments use single-forward-pass log-probabilities of a fixed target string, which the authors
  themselves flag as a proxy (Sec. 4.1).
- Not checked: Sec. 5 (misaligned actions), Sec. 6 (red-teaming), App. C-E details.

## Connections and questions

- **Definition offered:** risk/capability = the distribution over inputs of per-input success
  probabilities (elicitation probabilities), summarised by its extreme quantiles Q_p(n), forecast with a
  Gumbel tail on ψ = −log(−log p).
- **New vs better access:** not addressed. Its "aggregate risk" is the right quantity for a budget-matched
  test, though: the expected number of RL training targets that base-model sampling would solve at RL's own
  budget is Σ_i [1 − (1 − p_i)^{k_RL}] with k_RL ≈ 400 per target. Forecast it from pend's per-theorem p
  distribution and compare with the targets RL actually solved; round-1 RL samples on the 4,495 targets are
  base-model (pend) samples, so the RL logs contain a direct check.
- **Null / floor:** not discussed; the method assumes non-zero elicitation probabilities and extrapolates
  the upper tail, so it cannot by itself separate "never" from "rare".
- **Transfer to our setting:** (1) Use ψ = −log(−log p̂) (the same transform as PassUntil's log(−log PU)) for
  per-theorem p from samples, and compare pend vs r16 tails; (2) test Dan's teacher-forced log-likelihood
  proxy against sampled p_i on the theorems where both exist, as the paper does in Sec. 4.5 (expect the proxy
  to underestimate p_i, since p_i sums over all valid proofs); (3) forecast, from the ~320 eval theorems,
  how many of the 4,495 training targets have pend p_i above 1/400 and compare with RL's solved set. Cost:
  the forward-pass proxy is cheap; sampled p_i exist at n = 256 per seed. Failure modes: the fit uses only
  the top ten values (noisy with ~320 theorems); the method extrapolates the high tail, while the
  creation-vs-access question lives in the low tail (theorems with n_ok = 0).
- Related: wu2024rareoutputs.md, kazdan2025passk.md, schaeffer2025powerlaws.md, hu2023passuntil.md.
