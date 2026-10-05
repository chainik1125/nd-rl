---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - mousavihosseini2026barrier
---

# The base model barrier: outcome-reward policy gradient is limited by the base's likelihood quantile; process rewards by its token-level (worst-step) quantile

Paper: [@mousavihosseini2026barrier] (Mousavi-Hosseini, Erdogdu, "Post-Training with Policy Gradients:
Optimality and the Base Model Barrier", 2026; ICML 2026 poster per search result, not checked)
Source: arXiv 2603.06957v2 (HTML rendering read: abstract, Sec. 1-7; appendix proofs not read). Found by
search for 2025-2026 theory on RLVR support; it cites the Coverage Principle (prior list, 2510.15020) as
CHG+ [3] and Foster, Mhammedi, Rohatgi (prior list, 2503.07453) as FMR [8].

## Learnings

- **Setting.** Linear autoregressive softmax models over frozen features φ; each context x has a unique
  correct response y*(x) of length N over k tokens, with a token-level γ-margin (Assumption 1); outcome
  reward r = 1[y = y*(x)] (Sec. 2-3). The question is put in our terms: "the extent to which RL can create
  new knowledge that is missing from the base model is unclear" (Sec. 1).
- **Support, defined against the uniform policy.** If the base's likelihood of the correct response is at
  most polynomially small in N, "we consider this sample to be on the base model support, and the
  complexities above remain polynomial"; if it is not significantly better than uniform (k^{−N}), "the
  sample is off-support and improving its likelihood may require exponentially many iterations" (Sec. 1).
- **Per-sample result.** E[p(y*|x) | E_α] ≥ 1 − Õ(1/(π_α α γ² T)) (Thm. 3), where α is the behaviour
  policy's likelihood of y*(x): "the test performance on an individual sample after post-training depends
  on how likely the behavior policy q_t is to generate the correct response for that sample" (Sec. 3.1).
- **Likelihood Quantile (LQ).** Q_q(ε) = sup{α : P_x(q(y*(x)|x) ≤ α) ≤ ε}, the ε-quantile over contexts of
  the base likelihood of the correct response; 1 − π_α "is referred to as the" α-"coverage profile" of the
  Coverage Principle; for the uniform policy Q_q(ε) = k^{−N} (Sec. 3.2). Cor. 4-5: reaching expected error ε
  needs about Q_{q0}(ε)⁻¹/γ² reward queries, which "blows up exponentially" once ε is below the mass of
  contexts where the base is at uniform level. Lower bounds (Thm. 10-12) make this minimax: the barrier "is
  a fundamental property of post-training with outcome rewards" (Sec. 5.1).
- **Abstract claim.** PG "may require a number of reward queries exponential in N to go beyond this
  support, regardless of the pre-training algorithm" (Abstract).
- **Token-level LQ and process rewards.** Q^TL_q(ε) = sup{α : P_x(min_i q(y*_i | x, y*_{1:i−1}) ≤ α) ≤ ε}
  — the quantile of the *worst next-token probability along the correct response*; for the uniform policy it
  is k^{−1}, "notably independent of N"; Q^TL ≥ Q always. With a process reward, Õ(N min(1/Q^TL, k)/γ² +
  1/(γ²ε)) reward queries suffice (Thm. 6), so PG avoids "the curse of dimensionality in" N (Abstract).
- **Evidence.** Synthetic (N = 128, d = k = 32): of 32 mixture centres, 4 have base likelihood < 10⁻¹²;
  "PG with ORM is unable to boost the likelihood over any of these 4 samples, while the average likelihood
  improves under PRM" (Sec. 6, Fig. 1). Qwen3-8B vs Qwen3-8B-Base on MATH-500 with 64 samples per problem:
  below some ε both LQs are zero; "post-training was unable to extend the model's support beyond regions
  where the base model already exhibited non-trivial LQ" (Sec. 3.2, Fig. 2a).

## Evidence and limitations

- Strong assumptions: unique correct response, separability with margin, frozen features (only the last
  layer learns), exact process rewards. The authors list noisy and non-separable responses as open
  (Sec. 7). With frozen features, transfer between contexts is limited to what φ shares; the worst-case
  lower bounds use constructions where contexts do not help each other.
- The Qwen evidence uses 64 samples per problem, so "LQ zero" means below ≈ 1/64 resolution, not below
  k^{−N}.

## Connections and questions

- **Definition offered:** a sample is *on the base support* when the base's likelihood of a correct
  response is non-negligible relative to the uniform policy (polynomial, not exponential, in length);
  the base's capacity for RL is the whole *distribution* of these likelihoods across contexts (the LQ
  function), and, for step-level feedback, the distribution of the worst-step probability (token-level LQ).
- **New vs better access:** yes, as a theorem. With outcome rewards, what RL can reach at Q reward queries
  is the set of contexts with base likelihood ≳ 1/Q (elicitation); contexts at uniform level need
  exponentially many queries, so success there must come from something else (shared features /
  transfer, process feedback, or new data). Our rule (inference): per seed, predict RL's solved fraction from
  pend's likelihood distribution and RL's per-theorem query budget Q_t, i.e. P_t(p_B(t) ≥ c/Q_t); solved
  theorems with p_B(t) ≪ 1/Q_t are *not explained by on-support sharpening* and are the candidates for
  creation-by-transfer. The worst-step version separates two kinds of hard theorem: low sequence
  likelihood but decent worst step (reachable by step-level search or stepping stones) versus a worst step
  near chance (off token-level support: a genuinely missing move).
- **Null / floor:** built in — the uniform policy defines the floor (k^{−N} for sequences, k^{−1} per
  step), which is exactly Dan's random-weights objection made into the reference point. "Off-support"
  means "no better than the null", not "probability zero".
- **Transfer to our setting:** the project's worst-step measure w1 (minimum per-step log p of a reference
  or RL proof under pend) is the paper's token-level likelihood, and p_B(t) bracketing gives the sequence
  level. Compute both LQ functions for pend over the 322 set (J1 scoring already planned), compare with
  random-init (the floor) and with RL's solve set at r8 / r16. For the excluded-middle seed: was A ∨ ¬A
  on pend's token-level support (worst step of the double-negation proof well above chance) while off its
  sequence-level support? If so, EI's rounds can act as an implicit process signal via stepping stones; if
  the worst step was at chance, the theory says outcome-reward EI should not find it without transfer.
  Failure modes: many correct proofs per theorem (the theory assumes one), so likelihood of "the" correct
  response must be replaced by p_B(t) summed over proofs; vocabulary k and length N must be the action-level
  ones of `lean_staten`; the linear frozen-feature model is far from a 10 M-parameter transformer that
  learns features during EI.
- Related: `xie2024xpo.md` (coverage vs coverability), `huang2025bestofn.md` (coverage coefficient),
  `okawa2023multiplicative.md` (worst-step as the bottleneck of a product), `shenfeld2025razor.md`;
  prior-list papers not counted: Coverage Principle 2510.15020, Invisible Leash 2507.14843, sharpening
  2412.01951.
