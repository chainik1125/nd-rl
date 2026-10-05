---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - kazdan2025passk
---

# Efficient Prediction of pass@k Scaling (beta-binomial fit + dynamic sampling)

Paper: [@kazdan2025passk]
Source: arXiv 2510.05197v1 (HTML rendering read: abstract, Sec. 1-6, App. B, D.2). Some inline math with
"<" was lost in the HTML-to-text conversion (Sec. 3.1, Sec. 5.1 budget range); those passages are not quoted.

## Learnings

- **Problem statement = extrapolating pass@k beyond the sampled k.** "how can one accurately predict a
  model's behavior when scaled to a massive number of attempts, given a vastly smaller sampling budget?"
  (Abstract). Formal question: with B samples over m problems, predict pass_D@k "for k ≫ B/m" (Sec. 2).
  pass_i@1 is "the model's true probability of success in one attempt" and pass_i@k = 1 − (1 − pass_i@1)^k
  (Sec. 2, Eq. 1).
- **The unbiased estimator stops at k = n.** Chen et al.'s estimator is "only defined when the number of
  samples taken for each problem is greater than or equal to the number of attempts k" (Sec. 3.1, Eq. 3).
- **Critique of log-log regression (Brown 2024, Hughes 2024).** pass_D@k estimates "are not independent
  for different k when they are computed using the same dataset of samples", "are not homoskedastic", may
  not follow a power law, and "Power laws typically apply only for large values of k" (Sec. 3.2; proofs
  App. A). In results, regression is "particularly poor, often diverging to predict impossible pass rates
  greater than 1" (Sec. 5.2).
- **Critique of Schaeffer et al.'s discretized beta.** "Because the bins are wider for smaller values, this
  fitting method consistently produces downward-biased estimates of the distribution U" (Sec. 3.3, Fig. 2);
  in results it "consistently underestimates pass@k for large k" (Sec. 5.2). The scale parameter: "we find
  empirically that using the scale parameter does not improve predictions" (Sec. 3.3; tractable scaled
  likelihood in App. B, Lemma 3).
- **Their estimator.** s_i ~ Binomial(b_i, pass_i@1), pass_i@1 ~ U, and "we model U as a beta distribution"
  (Sec. 4.1); maximise the beta-binomial likelihood (Eq. 15-16) over (α, β) using the raw counts (s_i, b_i),
  which handles unequal b_i; predict pass_D@k = E_{p~Beta(α̂, β̂)}[1 − (1 − p)^k] (Eq. 17).
- **Dynamic sampling.** Since high-k behaviour depends on the difficulty distribution near 0,
  "Distinguishing between an easy problem (pass_i@1=0.25) and a very easy problem (pass_i@1=0.75) provides
  little to no information" (Sec. 4.2); Algorithm 1 repeatedly samples the problem with the fewest successes
  (ties: fewest attempts). Theorem 1: for the plug-in estimator, the variance-minimising allocation is
  b*_i ∝ sqrt(pass_i@1 (1 − pass_i@1)^(2k−1)) (Sec. 4.2; proof App. D). But "it is difficult to empirically
  isolate the benefits of the sampling method alone" on the real data (Sec. 4.2).
- **Evaluation against real held-out large-k data.** Data from Brown 2024 and Hughes 2024 (10,000 samples
  per problem, 100-200 problems); predictions for k from 100 to 10,000; "Ground truth estimates are computed
  for pass@k using all 10,000 available samples" (Sec. 5.1; Fig. 1, Fig. 5).
- **Impossible vs hard is not identifiable without samples on the hard problems.** In a synthetic case
  with half the problems "impossible" (pass_i@1 = 0), uniform sampling "prevents our estimator from
  determining whether these problems are impossible or just hard (i.e., still likely to be solved in k
  attempts). This results in an upwards-biased estimate" (App. D.2, Fig. 6).
- **RLVR motivation.** Predictions matter for RLVR "where training on difficult problems requires correctly
  sizing batches to ensure a non-zero success rate" (Sec. 1.1).

## Evidence and limitations

- Evidence: Fig. 1, 5, 7 (MSE heatmaps over budget × k), Fig. 6 (synthetic impossible problems).
- A plain Beta prior puts no mass at p = 0: every problem is assumed eventually solvable. App. D.2 shows the
  resulting upward bias when some problems really are impossible. This is the crux for our question and the
  paper does not propose a zero-inflated model.
- Extrapolation range tested is ≤ 10^4 (limited by the ground truth). Not tested: 10^5-10^6.
- Our derivation (standard Beta moment, not stated in the paper in this form): under Beta(α, β),
  pass_D@k = 1 − B(α, β + k)/B(α, β), which decays as a power law k^(−α) in the failure mass for large k
  — consistent with schaeffer2025powerlaws.md Thm 3.1.

## Connections and questions

- **Definition offered:** capability at scale = predicted pass_D@k for k far beyond the per-problem sample
  count, obtained from a fitted distribution U of per-problem success probabilities (beta-binomial MLE on raw
  success counts).
- **New vs better access:** not addressed. The paper provides the tool to make a budget-based rule
  quantitative: predict the base model's pass@k at the budget RL actually spent, and test whether RL's
  solved set lies inside it. App. D.2 says precisely what can go wrong: with a Beta prior the base model will
  be predicted to solve "impossible" problems eventually (upward bias), which would make every RL gain look
  like elicitation. A fair test needs a zero-inflated beta-binomial (point mass π₀ at p = 0) and enough
  samples on the zero-success theorems to estimate π₀.
- **Null / floor:** implicit. Under a Beta prior with no point mass the "any k solves it" objection is
  built into the model; the paper's own App. D.2 shows the fix is targeted samples on the hardest problems,
  which is what Algorithm 1 does.
- **Transfer to our setting:** (1) Fit beta-binomial and zero-inflated beta-binomial to pend's n_ok/256
  per theorem (seed 1), predict pass@k up to 1.8·10^6, and check against seed 2 and (for the older 3.2M
  model) against the 400,000-sample data — a real-data backtest at 40× the paper's horizon. (2) Spend new
  base samples with Algorithm 1 on the theorems with n_ok = 0 for pend but solved by r16 (dynamic
  allocation is exactly targeted at them); report the fitted π₀ for pend with and without these samples.
  Cost: fits are seconds; targeted sampling is the only GPU cost (e.g. 10^4-10^5 samples on a few dozen
  theorems). Failure mode: a Beta (or any continuous) prior cannot express "absent", and π₀ is only
  identified by the samples spent on zero-count theorems.
- Related: schaeffer2025powerlaws.md, brown2024monkeys.md, chen2021evaluating.md, levi2024 (screened row).
