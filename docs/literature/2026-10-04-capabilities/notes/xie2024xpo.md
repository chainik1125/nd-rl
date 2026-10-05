---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - xie2024xpo
---

# XPO: passive (on-policy) RL cannot find what the base does not cover; exploration is what leaves the support

Paper: [@xie2024xpo] (Xie, Foster, Krishnamurthy, Rosset, Awadallah, Rakhlin, 2024; ICLR 2025)
Source: arXiv 2405.21046v1 (HTML rendering read: abstract, Sec. 1-4, Remarks 2.1, 3.1-3.3, App. D.1 proof of
Prop. 2.1; proofs in App. C not read).

## Learnings

- **Claim.** A one-line change to online DPO, an optimism bonus α Σ_i log π(τ̃_i) on reference samples,
  "empowering the algorithm to explore outside the support of the initial model and human feedback data";
  it converges to a near-optimal policy "irrespective of whether the initial model has good coverage"
  (Abstract).
- **What non-exploratory RL can reach.** Offline methods "are fundamentally limited to behaviors that are
  well-supported by the initial model" (Sec. 2.2). Online, on-policy sampling does not fix this: "Passive
  exploration is intuitively insufficient, as we are unlikely to generate novel and correct proofs by
  chance" (Sec. 1).
- **Prop. 2.1 (the lower bound).** Two actions, r(a) = 1, r(b) = 1/2, π_ref(a) = ε = exp(−c/β) (App. D.1).
  For all T ≤ ½ exp(1/(8β)), with constant probability every iterate of Online DPO stays ≥ 1/8 suboptimal:
  "the sample complexity required by Online DPO is exponential in" 1/β, and the same holds for Iterative
  and offline DPO (Sec. 2.3, Eq. 5). Mechanism: if π_ref "places small probability mass on the optimal action, Online DPO may fail to ever explore this action until the number of iterations is exponentially large"; "more deliberate exploration is required to discover behaviors or capabilities not already covered by" π_ref (Sec. 2.3). The failure is due to "poor coverage from" π_ref, "in spite of on-policy sampling" (Remark 2.1).
- **Two coverage quantities.** Passive methods scale with the concentrability C_conc(Π) = sup_τ sup_π
  π(τ)/π_ref(τ), which equals exp(V_max/β) under bounded density ratios (Sec. 3.2). XPO scales with the
  coverability C_cov(Π) = inf_μ sup_τ sup_{π∈Π} d^π(τ)/μ(τ) (Def. 3.1, Eq. 11), which "measures coverage with
  respect to the best possible distribution \mu" and is bounded by |A|^H. Thm. 3.1: T = Õ(C_cov log|Π| / ε²).
  "KL-regularization does not automatically lead to exploration or grant meaningful control of
  coverability in the small-\beta regime" (Remark 3.1).
- **Implicit reward model.** For the optimal KL-regularised policy, β log π*_β(τ)/π_ref(τ) = r(τ) − V*_β(s1)
  for every trajectory; "\pi^\star_\beta implements an accurate internal reward model" (Sec. 3.1, Eq. 9).

## Evidence and limitations

- The guarantee needs realizability (π*_β ∈ Π), bounded log density ratios, deterministic transitions
  (Remark 3.2) and trajectory-level coverability (Remark 3.3); finite classes. The objective is non-convex
  and harder to optimise as β → 0.
- Experiments are preliminary: Llama-3-8B, iterative DPO with T = 3, chat / academic benchmarks; XPO matches
  heuristic-exploration baselines with "only" 1/4 "the number of generated responses" (Sec. 3.4, Table 1).
  Nothing in the experiments tests leaving the base's support.
- Preference feedback (Bradley-Terry), not a binary verifier; the exp(2 R_max) factor is from that model.

## Connections and questions

- **Definition offered:** coverage of a behaviour by the base = the density ratio π*(τ)/π_ref(τ)
  (concentrability). A behaviour is "not already covered by π_ref" when its base mass ε is exponentially
  small relative to the sampling budget T; Prop. 2.1 makes the budget explicit (T ≤ 1/(2ε)).
- **New vs better access:** implicit and budgeted. With passive (on-policy) training, a target whose base
  mass is ε is found only after ~1/ε samples *on that target*, so anything RL reaches with far fewer
  samples on t must come from somewhere other than t's own samples. Our rule (inference, not in the
  paper): for a theorem t first solved in round r, compare the samples EI drew on t (32 per round) with
  1/p_{r−1}(t) under the previous round's model. If p_B(t) · N_t ≪ 1 (pend could not have been "selected"
  into solving t) but p_{r−1}(t) ≳ 1/32, the probability on t was raised by training on *other* targets:
  transfer through the policy class, which is the only route by which passive RL leaves a theorem's base
  coverage. XPO's C_cov names that route: structure in Π, not mass in π_ref.
- **Null / floor:** handled by the trivial bound C_cov ≤ |A|^H (uniform μ, i.e. random weights): every class
  is coverable at that price; what matters is the gap between that and the base's density ratio, against
  the algorithm's sample budget. This is the frame's K_null vs K_per comparison in the paper's own terms.
- **Transfer to our setting:** EI is passive exploration (samples only from the current policy, no
  optimism, keeps successes). Per theorem, compute the base density ratio of RL's accepted proofs,
  log π_R(y|t) − log π_B(y|t) (J1 scoring, cheap), and its sequence-level supremum as a concentrability
  estimate; compare log p_R(t)/p_B(t) with log N_t. For the excluded-middle seed, the test needs the round
  12-14 checkpoints: did p(A ∨ ¬A) rise across rounds while A ∨ ¬A was never accepted (stepping stones from
  other targets, e.g. ¬¬-elimination or reductio theorems), or did a single lucky draw at round 13 start
  it (selection)? Failure modes: intermediate checkpoints may not exist (pods); p_B(t) is below sampling
  resolution (only the F(t) lower bound and Clopper–Pearson upper bound); the theory's realizability and
  deterministic-transition assumptions hold, but the β → 0 limit (no KL in EI) is where the bounds are
  least informative.
- Related: `huang2025bestofn.md` (coverage C_{π*} = E[π*/π_ref] as the price of inference-time selection),
  `korbak2022rlkl.md` (the KL-RL target), `shenfeld2025razor.md` (on-policy RL stays near the base);
  prior-list papers not counted: Foster, Mhammedi, Rohatgi 2503.07453; Coverage Principle 2510.15020;
  Song, Kempe, Munos outcome-based exploration 2509.06941.
