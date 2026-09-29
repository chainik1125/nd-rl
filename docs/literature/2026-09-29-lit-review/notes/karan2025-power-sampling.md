---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Power sampling from the base model as an RL-matching control
Paper: Reasoning with Sampling: Your Base Model is Smarter Than You Think (Karan & Du, 2025)
Source: https://arxiv.org/abs/2510.14901v1 (v1 reviewed; html text)
## Learnings
- Target distribution p^α (whole-sequence power of the base). §4.1 Prop. 1: low-temperature sampling (per-token p^α renormalised) "does not sample from the power distribution p^α"; the true conditional sums p^α over all futures, so it favours prefixes with few but high-likelihood completions.
- Algorithm 1 (§4.3): blockwise Metropolis–Hastings. Pick a random index, resample the suffix from a proposal (base at temperature 1/α), accept/reject using base likelihoods only. No reward and no verifier is used ("single-shot").
- Settings (§5.1): α = 4.0, T_max = 3072, B = 192, proposal = base at τ = 1/α.
- Table 1 (MATH500 / HumanEval / GPQA): Qwen2.5-Math-7B base 0.496/0.329/0.278, low-temp 0.690/0.512/0.353, power 0.748/0.573/0.389, GRPO(MATH) 0.785/0.537/0.399.
- §5.3, Fig. 4: power samples sit in higher base-likelihood regions, "but still maintains noticeable spread"; GRPO samples are "heavily concentrated at the highest likelihood peak". Fig. 5: power pass@k is above base and GRPO, and converges to base at large k.
- §5.3 cost: token multiplier N_MCMC·T/(4B); with N_MCMC = 10, T = 679, B = 192 that is "8.84×". Accuracy rises until N_MCMC = 10; going from 0 steps to 2 steps gives "3-4%".
## Evidence and limitations
- The comparison uses 7B-scale pretrained LLMs on single-answer tasks. Pass@1 is compared to GRPO at matched model, not at matched sampling compute per problem. At large k, power sampling only reaches the base curve (Fig. 5); it does not go beyond it. So it is an elicitation baseline by construction. It cannot reach sequences the base gives low likelihood.
- Appendix pass@k figures (7–9) were not checked beyond their captions.
## Connections and questions
- B2 (create vs elicit). This is a training-free "pure sharpening" control. If power sampling from our Stage-1 base solves the theorems that EI newly reaches, at matched attempts × tokens, then EI's gain is sharpening (elicitation). If it does not, that is evidence against the plain-sharpening story. It is cheap for us: the model has 3.2 M parameters and proofs are ~100s of tokens, so the 8.84× overhead is minor.
- Whole-proof arm: run Algorithm 1 with B ≈ one proof line.
- State-env: MH over step sequences. Resample from a random step t using the base policy, and accept by the ratio of base step-likelihood products. This is a likelihood-accepted "resume from prefix".
- Direct measurement (Fig. 4 analogue): histogram the base per-token log-likelihood of (a) base-sampled correct proofs, (b) EI-found proofs of newly-reached theorems, (c) power samples. Our small model makes exact likelihoods of any proof cheap.
- Smallest test: on the 29 hardest theorems, compare base pass@k, low-temperature pass@k, and power-sampling pass@k at matched tokens against the EI T1 model.
