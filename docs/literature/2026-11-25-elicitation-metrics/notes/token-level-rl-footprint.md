---
written_on: 2026-11-25
written_by: agent:claude
papers:
  - qwenpilot2026sparsebutcritical  # proposed key; arXiv 2603.22446
  - akgul2026sparsepolicyselection  # proposed key; arXiv 2605.06241
  - wang2025highentropyminority  # proposed key; arXiv 2506.01939
---

# Where RL changes the next-token distribution: sparse, high-entropy, mostly reranking

Papers:
- "Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs", Qwen Pilot Team, Alibaba,
  arXiv 2603.22446v1, ICLR 2026. Full text in part: a tool extracted §1–5 and Table 1 from the first 100k characters. The
  conclusion and the full author list were not seen.
- Akgül, Kannan, Neiswanger, Prasanna, "Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability
  Learning", arXiv 2605.06241. Full text in part: §3, §6 and Tables 1–3, tool-extracted.
- Wang et al., "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM
  Reasoning", arXiv 2506.01939. Abstract only.

## Learnings

What the papers claim:
- **Sparse but Critical.**
  - **Divergence measure** (§2.1): per-position JS divergence between base and RL models on the same contexts, on top-p-truncated distributions.
  - **How many positions change** (§2.2, Qwen2.5-32B, AIME24): "more than 83% of token positions exhibit near-zero divergence" under DAPO, and the share
    "exceeds 98% under SimpleRL".
  - **Cross-sampling** (§3.1): generate from the primary model and, at positions where JS exceeds a fixed ε, take the token from the other model, up to a
    cap on interventions.
  - **Results** (Table 1, Qwen2.5-32B, AIME24, 8k-token budget):
    - Base plus RL tokens at 3.86 % of positions (SimpleRL) or 7.8 % (DAPO) goes from 8.23 % to about the RL accuracy.
    - The reverse swap, at 5 % and 10.1 % of positions, drops the RL models to below 8.3 % and 8.5 %.
  - **Their framing** (§5): "a complementary diagnostic tool".
- **Akgül et al.**
  - **Method** (§3.1): greedy MATH-500 with identical prefixes. "Reranked" means the RL argmax differs from the base argmax but is in the base top-5.
  - **How often and where:** 1.03–3.96 % of positions are reranked (Table 2), with mean base rank 2.14–2.39. These positions have 7.6–12.6× higher base entropy.
  - **ReasonMaxxer:** a contrastive loss at entropy-gated positions, trained on about 1,000 base rollouts, recovers much of the RL gain for under $10 (§6).
  - **Inconsistencies:**
    - §3.1's "zero shifted positions" against Table 2's 0.01–0.12 %;
    - the abstract's "1–3%" against 1.0–4.1 % in §3.1;
    - §6.2's "on par with or better than the RL baseline", which Table 3 contradicts. For example, Qwen3-4B with General-Reasoner scores 0.746 against 0.660 for ReasonMaxxer.
- **Wang et al.** (abstract): RLVR "largely adheres to the base model's entropy patterns". Updating only about 20 % of the tokens, the forks, keeps performance.

Our interpretation:
- **The question for Olmo.** On math RLVR, RL's token-level footprint is a few reranked forks. The question is whether this holds for IF RL on Olmo,
  and whether it holds already at `dpo` or only from some RL step on.
- **What we can compute.** The step-0 per-position storage adopted for stage 1 (NLL, rank, entropy, top-20) lets us compute both of the
  papers' statistics between any two checkpoints with no new GPU time.
- **Behavioural follow-up.** The cross-sampling intervention count is the natural behavioural version. It is costly at 32B because it needs two resident models.

## Evidence and limitations

- **Math only.** All of it is math with base or zero-RL models; none is a thinking model after SFT and DPO, and none is IF.
- **Different measures.** The two papers measure different things (JS on sampled contexts; argmax disagreement under greedy decoding), so their
  "sparsity" numbers are not comparable with each other or with ours.
- **Support is not reachability.** Token-level "in the top-5" is not sequence-level reachability. Reranking at forks compounds over 5k-token responses.

## Connections and questions

- [REVIEW.md](../REVIEW.md), items 3 and 8.
- Our nd forced-first-action factorisation is the one-position case.
- Dan's `olmo-transplant` is the all-positions case.
