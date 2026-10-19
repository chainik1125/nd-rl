---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - chen2021evaluating
---

# Codex / HumanEval: the unbiased pass@k estimator (estimator section only)

Paper: [@chen2021evaluating]
Source: arXiv 2107.03374v2 (HTML rendering read; Section 2.1, Section 3.3-3.4, Appendix A only)

## Learnings

- **Definition of pass@k and why the naive version is noisy.** Kulal et al. (2019) generate k samples,
  call a problem solved "if any sample passes the unit tests", and report the solved fraction; Chen et al.
  note that "computing pass@ k in this way can have high variance" (Sec. 2.1). Instead they "generate
  n≥k samples per task (in this paper, we use n=200 and k≤100), count the number of correct samples c≤n"
  and compute the unbiased estimator (Sec. 2.1, Eq. 1):

  pass@k := E_Problems[ 1 − C(n−c, k) / C(n, k) ]

- **Numerics.** "Calculating this estimator directly results in very large numbers and numerical
  instability"; Figure 3 gives a product form: `if n - c < k: return 1.0; return 1.0 - np.prod(1.0 - k /
  np.arange(n - c + 1, n + 1))` (Sec. 2.1, Fig. 3).
- **The plug-in estimator is biased.** Estimating with 1 − (1 − p̂)^k "results in a consistent
  underestimate" and "The gap doesn't fully close even when n>5k" (App. A, Fig. 13). Eq. 1 "is unbiased,
  because it estimates the fail probability (1−pass@1)^k as the probability of drawing k failed samples
  without replacement"; the proof uses c ~ Binom(n, p) and sums to 1 − (1 − p)^k (App. A).
- **Variance is shown only empirically.** "The unbiased estimator may have a slightly higher variance
  initially but allows for a fair comparison across different numbers of samples" (App. A, Fig. 13
  caption). No closed-form variance or confidence interval is given in the sections read.
- **pass@k depends on the sampling temperature, and the optimum moves with k.** "higher temperatures are
  optimal for larger k, because the resulting set of samples has higher diversity" (Sec. 3.3, Fig. 5); for
  a 679M model T* = 0.2 for pass@1 and T* = 0.8 for pass@100 (Sec. 3.3).
- **Smooth in scale.** With per-k optimal temperatures, "Performance appears to scale smoothly as a sigmoid
  in log-parameters" (Fig. 6 caption, Sec. 3.3).
- **A parameter-equivalence reading of capability.** They summarise other models as equivalent Codex
  sizes: "GPT-Neo-2.7B roughly equivalent to Codex-85M (30× fewer parameters)" (Sec. 3.4).
- **Log-probability ranking.** Choosing "the sample with the highest mean token log probability
  outperforms evaluating a random sample", while sum log-probability "can perform slightly worse than
  picking randomly" (Sec. 3.3, Fig. 7).

## Evidence and limitations

- Eq. 1 is unbiased per problem for k ≤ n only; the paper uses n = 200, k ≤ 100 (Sec. 2.1). Nothing in the
  sections read addresses k > n.
- Our addition (not in the paper): Eq. 1 is a function of c ~ Binom(n, p), so its exact variance is
  computable for any p as Σ_c Binom(c; n, p)·(est(c) − (1 − (1 − p)^k))². Also a standard result: for
  Binom(n, p) data only polynomials in p of degree ≤ n have unbiased estimators, and 1 − (1 − p)^k has degree
  k, so **no unbiased estimator of pass@k exists for k > n**. Any statement about k beyond the sampled n is a
  model-based extrapolation (see the notes on Schaeffer 2025, Kazdan 2025, Hu 2023).
- The dataset-level estimate also carries problem-sampling variance (finite benchmark), which Eq. 1 does
  not address.
- Only the estimator sections were read; the rest of the paper (Codex training, safety analysis) was not.

## Connections and questions

- **Definition offered:** capability on a problem set = expected fraction of problems with at least one
  functionally correct sample among k, estimated without bias from n ≥ k samples with c correct (Eq. 1).
  The per-problem object underneath is the single-sample success probability p (pass@1), since
  pass@k = 1 − (1 − p)^k (App. A).
- **New vs better access:** not addressed. The paper treats pass@1 and pass@100 as two views of the same
  model (and tunes temperature per k), which already shows that pass@k at fixed k mixes "how likely" with
  "whether at all". Turning it into a rule requires choosing k, which is the user's objection.
- **Null / floor:** none. A useful corollary for us: because pass@k is a deterministic function of p,
  "solvable at some k" is equivalent to p > 0; the substantive question is the size of p, which a random
  model has astronomically small (≈ product of per-token probabilities of any proof). So the principled
  object is p (or −log p), not the choice of k.
- **Transfer to our setting:** compute Eq. 1 per theorem from n_ok out of n (n = 256 per seed, 512 pooled
  over the two seeds) for pend, r8, r16; exact for every k ≤ 512 at negligible cost. Report the exact
  per-theorem variance from the Binomial formula above, and bootstrap over theorems for the set-level
  interval. Failure mode: for theorems with n_ok = 0 the estimator says pass@k = 0 for all k ≤ n, which is
  exactly where "new vs elicited" is decided; it carries no information beyond "p is probably < ~3/n".
  For the RL budget comparison (≈ 400 attempts per target) n = 512 suffices on the eval set; for the
  400,000-sample older model the estimator covers k up to 4·10^5.
- Related notes: brown2024monkeys.md, schaeffer2025powerlaws.md, kazdan2025passk.md, hu2023passuntil.md.
