---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - schaeffer2025powerlaws
---

# How Do Large Language Monkeys Get Their Power (Laws)? — per-problem p and its left tail

Paper: [@schaeffer2025powerlaws]
Source: arXiv 2502.17578v1 (HTML rendering read: abstract, Sec. 1-5, 7, App. B, F, G headers)

## Learnings

- **Per problem, failure falls exponentially in k; the aggregate power law is a property of the
  distribution of p over problems.** "a simple mathematical calculation predicts that on each problem, the
  failure rate should fall exponentially with the number of attempts. We confirm this prediction
  empirically" (Abstract; Sec. 2, Eq. 7: pass_i@k = 1 − (1 − pass_i@1)^k; Fig. 3). Aggregate power-law
  scaling arises "if the distribution of single-attempt success probabilities is heavy tailed such that a
  small fraction of tasks with extremely low success probabilities collectively warp the aggregate success
  trend into a power law" (Abstract).
- **Exact link (Theorems 3.1-3.2).** The aggregate is pass_D@k = 1 − ∫(1 − p)^k p_D(p) dp (Sec. 3, Eq. 10).
  "the negative log average success rate will exhibit power law scaling in k with exponent b if and only if
  the distribution over problems of single-attempt success probabilities itself behaves like a power law
  near 0 with exponent b−1"; if p_D(p) = C p^(b−1) + … near 0 then −log(pass_D@k) ∼ C Γ(b) k^(−b) (Sec. 3,
  Thm 3.1; necessity Thm 3.2; proofs App. E, E.9). Closed forms: Uniform(0, β) gives k^(−1); Beta(α, β)
  gives k^(−α); Kumaraswamy(α, β) gives k^(−α) (Sec. 3).
- **Fitted distributions need a scale (max) parameter.** Single-attempt rate distributions were "well
  fit by a 3-parameter Kumaraswamy(α, β, a=0, c)"; "the scale parameter was critical to obtain good fits"
  (Sec. 3, Fig. 4); "the largest pass_i@1 values were typically 1-2 orders of magnitude less than 1.0"
  (App. F).
- **Deviations from a power law diagnose a missing tail.** Llama 3 8B IT "could be successfully jailbroken
  on every prompt within the permitted sampling budget and thus had no heavy left tail necessary to create
  the aggregate power law scaling" (Sec. 4).
- **Distributional estimator for extrapolating pass@k beyond the sampled k.** The obstacle: problems whose
  pass_i@1 "lie between (0, 1/Number of Samples) such that, due to finite sampling, we lack the resolution
  to measure"; the fix: "we can fit a distribution's parameters such that the distribution's probability
  mass in the interval (0, 1/Number of Samples) matches the empirical fraction of problems in this tail
  bucket", by "discretizing the distribution according to the sampling resolution" and maximum likelihood
  (Sec. 5). Then simulate pass_D@k at any k via Eq. 11. Concretely a scaled Beta-Binomial likelihood
  P(X = x; α, β, c, n) (App. F, Eq. 48-51).
- **Efficiency claim.** The distributional estimator recovers the exponent "with an order of magnitude
  lower relative error, or equivalently, ∼2−4 orders of magnitude less inference compute" (Abstract;
  Sec. 5, Fig. 7, backtesting on synthetic data), and "performs well even under distributional mismatch"
  (Sec. 5).
- **Terminology.** They prefer "average success rate" to "coverage" "as it avoids the binary implication
  that each problem either is or is not solved after k attempts" (Sec. 1).
- **Caveat they raise.** The tail structure may reflect "benchmark design" and "selection bias, in that
  more interesting patterns such as power law scaling are more likely to garner more interest" (Sec. 7).

## Evidence and limitations

- Evidence: Fig. 3 (per-problem exponential), Fig. 4 (fitted tails), Fig. 6 (agreement of least-squares
  and distributional exponent on real data), Fig. 7 (backtest on synthetic data only).
- The extrapolation is of the **aggregate** pass_D@k under a parametric tail assumption. It says nothing
  about whether an individual unsolved problem has p = 0 or p = 10^−9: the left tail is identified only by
  the count of zero-success problems plus the parametric form.
- The data sets have ≤ 10^4 samples per problem; no test of extrapolation to 10^5-10^6 real samples. The
  real-data comparison (Fig. 6) is agreement between two estimators, not a forecast checked against held-out
  large-k data (that is done in kazdan2025passk.md).
- Not checked: App. C fits, App. E proofs in detail.

## Connections and questions

- **Definition offered:** the capability object is the per-problem single-attempt success probability
  pass_i@1 = p_i and, at the set level, its distribution D over problems (especially its left tail near
  p = 0). pass@k at any k is a functional of D (Eq. 10).
- **New vs better access:** not discussed, but the framework makes a sharp version possible. Fit D for
  pend and for r16 from the same theorems. "Better access" = r16's D is pend's D with mass moved from small
  p to larger p on the same theorems (sharpening); "new" would require theorems whose pend p is in the
  unresolvable bucket (0, 1/n) **and** for which the fitted pend tail predicts essentially no solves even at
  a large principled budget. A decision rule: compute the fitted pend pass_D@k at k = RL's own sampling
  budget (≈ 400 attempts per target, or ≈ 1.8 M in total) and compare with r16's pass@1 or pass@k at small k;
  if pend at the RL budget already covers r16's solves, RL's gains are within the base model's reach.
- **Null / floor:** the paper's framework implies the answer to "at some k even random weights solve it":
  pass_i@k → 1 for any p_i > 0, so the question is the size of p_i, i.e. where D's mass sits. A random-init
  model's D sits at astronomically small p (≈ per-token probabilities multiplied over a proof), far below any
  budget; the distributional view states the floor quantitatively rather than by picking k. Our addition:
  p_i ≥ P_θ(any one valid proof), so the teacher-forced log-likelihood of a reference proof is a sampling-free
  **lower bound** on log p_i, computable for every checkpoint including random init.
- **Transfer to our setting:** fit a scaled Beta-Binomial (App. F) to n_ok out of n = 256 (or 512 with
  both seeds) for pend, r8, r16; report α (the left-tail exponent, which sets the pass@k growth rate), the
  scale c, and the predicted pass_D@k at k = 400 and k = 1.8·10^6. Validate on the older 3.2M model: fit on
  a 256-sample subsample of the 400,000-sample data and check the predicted pass@k at 10^4-4·10^5 against
  the measured values (a backtest the paper did only on synthetic data). Cost: seconds of CPU. Failure
  modes: (1) the left tail is identified only by the zero-count fraction, so the extrapolation depends on
  the parametric form; (2) a mixture with a point mass at p = 0 ("truly unsolvable") fits the same 256-sample
  data, and the choice between them is exactly the creation-vs-access question — so the 400,000-sample data
  are needed to see whether zero-count theorems keep converting at the predicted rate.
- Related: brown2024monkeys.md (aggregate fit), kazdan2025passk.md (better estimators, backtests on real
  data), hu2023passuntil.md (sampling until first success), chen2021evaluating.md (estimator).
