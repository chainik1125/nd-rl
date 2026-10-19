---
written_on: 2026-10-05
written_by: agent:claude
papers:
  - korbak2022rlkl
---

# RL with KL penalties is Bayesian inference: the KL-RL optimum is the base conditioned on the reward

Paper: [@korbak2022rlkl] (Korbak, Perez, Buckley, Findings of EMNLP 2022)
Source: arXiv 2205.11275v2 (HTML rendering read in full: abstract, Sec. 1-6, Limitations, Appendix).
Also screened at abstract level by L3 (`_screen_L3.md`); this is the in-depth note.

## Learnings

- **Claim.** "KL-regularised RL is equivalent to variational inference: approximating a Bayesian posterior
  which specifies how to update a prior LM to conform with evidence provided by the reward function"
  (Abstract).
- **The target.** π*_KL-RL(x) = (1/Z) π0(x) exp(r(x)/β), where "\exp(r(x)/\beta) is the evidence provided
  by the reward function (scaled by temperature \beta)" (Sec. 4, Eq. 5); it "also happens to coincide with
  the optimal policy for J_KL-RL" (Eq. 6), and J_KL-RL ∝ −D_KL(π_θ, π*) (Eq. 7), i.e. the objective is an
  ELBO on log p(O = 1) (Appendix, Eqs. 11-17).
- **Z is the base's success probability.** In the Appendix the reward is turned into an optimality variable,
  p(O = 1 | x) = exp(r(x)), and the authors "redefined the marginal" p(O = 1) "as the normalising constant" Z
  (Eqs. 8-10); the marginal is "a probability that a random sample from \pi is non-offensive". For a binary
  verifier this Z is exactly the base solve probability p_B(t).
- **Binary rewards.** Following Khalifa et al., π*(x) = (1/Z) π0(x) b(x): excluded strings get probability
  zero "but all other strings keep the original probability \pi_0(x) up to Z (hence no degeneration)"
  (Sec. 5, Modelling). The posterior is the base *conditioned* on success; it never reorders two acceptable
  outputs.
- **Inference vs modelling.** Posteriors "might lie outside the class of probability distributions
  representable by parametric LMs" (Sec. 5, Inference); KL-RL is the variational route, and decoding-time
  methods are the sampling route, "The simplest example of that is filtering (also known as rejection
  sampling)". The split "separates two failure modes: misspecifying the model (i.e. not capturing task
  preferences) and failing to approximate the model well enough" (Sec. 5).
- **Collapse is not an exploration problem.** Plain reward maximisation converges to a Dirac on the best
  sequence; "Even with perfect exploration" (uniform sampling over X) the optimum is the same (Sec. 2).
  "RL avoids distribution collapse only with reward functions that make it equivalent to divergence
  minimisation" (Sec. 1).

## Evidence and limitations

- Position paper; the only formal content is the textbook control-as-inference derivation (Appendix). No
  experiments. The authors concede it does not explain why reward "is approximately linear in
  \sqrt{D_KL(\pi_\theta,\pi_0)} throughout RLHF training" (Limitations) and gives "limited guidance" on design.
- The argument is per-sequence (no prompts / states); generalisation across prompts, which is what a
  parametric policy does when it is trained on many theorems, is outside the analysis.
- Expert iteration has no explicit KL term; it is the amortised (fine-tuned) version of filtering, so the
  posterior is the target of *one* round from a fixed base, not the fixed point of 8-16 rounds.

## Connections and questions

- **Definition offered:** no capability definition. The quantity is the *posterior target* π_B(·|S_t) (binary
  verifier, β → 0) or the exp(r/β)-tilt of the base, with normaliser Z = p_B(t). Under this view, ideal
  KL-regularised RL can only reweight inside the base's support and, for a binary reward, only *rescale*
  the success set as a block.
- **New vs better access:** not discussed. Our derivation (not in the paper): the posterior leaves the ratio
  π(y1)/π(y2) of any two correct proofs unchanged, and tilting with a binary reward moves the success
  log-odds by exactly 1/β nats. That gives two numbers per theorem: (a) **evidence** e(t) =
  logit p_R(t) − logit p_B(t), the reward evidence a Bayesian update of the base would need to reach RL's
  solve rate (pure elicitation reaches any p_R at KL cost exactly the binary kl(p_R ‖ p_B) ≤ log 1/p_B(t),
  because within-class ratios are untouched); (b) **ratio violation**:
  among the accepted proofs F(t), the slope / rank correlation of log π_R(y) against log π_B(y). Rule:
  RL's change on t is *posterior-consistent* (elicitation in the Bayesian sense) if (b) shows slope ≈ 1 and
  no reordering; it is *not explained by conditioning the base on t's own success* if RL concentrates on a
  correct proof the base ranks far below other correct proofs. The second case is the signature of
  amortisation across theorems (transfer from other targets), which is where a parametric learner can
  "create" relative to the per-theorem posterior.
- **Null / floor:** softmax policies give every string positive probability, so the posterior always exists
  and support is never literally the obstacle; the obstacle is Z. The random-weights objection becomes
  "the KL price log 1/Z_null is astronomically large" — the same budget logic as the cards' K_null, with
  budgets in nats (log K) rather than attempts.
- **Transfer to our setting:** needs teacher-forced log p of every accepted proof under pend and r8/r16
  (J1 scoring, already planned) and the bracket on p_B(t) for Z (lower bound Σ_{y∈F(t)} π_B(y|t)). Cheap.
  For the excluded-middle seed: is r16's double-negation proof of A ∨ ¬A the base's *most likely* accepted
  proof of that theorem (posterior mode), or one the base ranks below others in F(t)? Failure modes: EI is
  not KL-regularised and trains on many theorems plus replay, so violations are expected even without
  "creation"; F(t) is only the proofs anyone found, so the ranking is over a truncated set; name-base
  marginalisation must be identical for both models.
- Related: `shenfeld2025razor.md` (same I-projection algebra: KL(q‖p) = KL(q‖p(·|S)) − log p(S)),
  `lin2023urial.md` (token-level shift), `xie2024xpo.md` and `huang2025bestofn.md` (what coverage of the
  base costs), `_screen_L3.md` (Zhao et al. 2024, twisted SMC estimates of Z).
