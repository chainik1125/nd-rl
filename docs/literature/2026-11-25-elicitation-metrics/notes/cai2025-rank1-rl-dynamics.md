---
written_on: 2026-11-25
written_by: agent:claude
papers:
  - cai2025predictabilityrldynamics  # proposed key; arXiv 2510.00553
  - zheng2024expo  # proposed key; arXiv 2404.16792
---

# RL updates are rank-1-dominated and extrapolable (Cai et al.), and model extrapolation (ExPO)

Papers:
- Cai et al., "On Predictability of Reinforcement Learning Dynamics for Large Language Models", arXiv 2510.00553v3, ICLR 2026.
  Full text in part: a tool extracted passages from the first 100k of 128k characters. Appendices D–E were not seen.
- Zheng et al., "Weak-to-Strong Extrapolation Expedites Alignment" (later "Model Extrapolation Expedites Alignment"), arXiv
  2404.16792. Abstract and a search summary only.

## Learnings

What the papers claim:
- **Cai et al.**
  - **Rank-1 construction** (§2.1, Eqs. 1–3): the rank-1 update is σ1·u1·v1ᵀ of ΔW = W_RL − W_base, rescaled by ‖ΔW‖ / ‖ΔW^(1)‖.
  - **Rank-1 sufficiency:** "the Rank-1 Subspace alone recovers an average of 99.17% of the reasoning capability" (§1). For RLOO, GRPO and DAPO "it can even surpass the fully trained model" (§2.2).
  - **Linear dynamics:** the rank-1 trajectory over checkpoints is fitted linearly to accuracy with single-component PLS. R² is "0.914" on average in §1, elsewhere "exceeding 0.96".
  - **AlphaRL:** extrapolates from early training, effective "even at early stages (only 10% of the total training steps)" (§4.2). The abstract reports "up to 2.5× speedup while retaining >96% of reasoning performance".
  - **SFT and distillation:** "yielded neither property" (§1). But App. C: "Both On-Policy Distillation and DPO exhibit clear low-rank structures".
  - **Scope:** 8–13 models (the count differs between sections), PPO, RLOO, GRPO, Dr.GRPO and DAPO, math benchmarks.
  - **Limitation stated** (§6): "these findings still lack rigorous theoretical foundations".
- **ExPO** (abstract): θ2 = θ1 + α(θ1 − θ0) applied to DPO and RLHF models improves AlpacaEval 2.0 by up to 6.8 % (length-controlled). Models trained
  on 10–20 % of the preference data, then extrapolated, can match the fully trained ones.

Our interpretation:
- **What the metric is.** It describes the shape and size of the update, and it needs no training: "is RL's change from dpo to t one direction per
  matrix, and was that direction set by step 100–500?"
- **What it is not.** It is not a direct latent-capability measure. A rank-1 update can still implement new behaviour.
- **What it pairs with.** It pairs naturally with Dan's `olmo-weights` norms and with EDL's "how many bits".

## Evidence and limitations

- **Inconsistent within itself.** The model counts, the R² figures and the "RL only" claim (contradicted by App. C) all disagree across sections.
- **Underspecified.** It is unclear whether rank-1 is applied to all matrices at once. No seeds or error bars are visible in the parts read.
- **Lineage caveat for Olmo.** `olmo-weights`' review found that steps 150 and 200 are on an abandoned branch, and that later step labels may overstate
  optimizer steps by about 100. Extrapolation must use on-lineage checkpoints.

## Connections and questions

- [REVIEW.md](../REVIEW.md), item 4.
- On `origin/dan`: `experiment-summaries/2026-11-25-olmo-weights-…`.
