---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Self-Evolving Curriculum: a bandit over difficulty levels beats easy-to-hard ordering on held-out harder levels
Paper: Self-Evolving Curriculum for LLM Reasoning (2025)
Source: https://arxiv.org/abs/2505.14970v4

## Learnings
- Curriculum selection as a non-stationary multi-armed bandit, "each arm represents a problem category" (§2.1). Arm reward = "the average absolute advantage across all rollouts associated with the problems drawn from curriculum category c" (§2.2), a proxy for gradient norm. Categories sampled from a Boltzmann distribution over Q_t(c) with temperature τ, then problems uniform within category (§2.3).
- Setup (§3.1): Countdown, Zebra, ARC-1D, math; "training data consists of the three easiest difficulty levels, and the most difficult level is reserved as an out-of-distribution (OOD) evaluation set".
- Results (§3.2, Table 1, Qwen2.5-3B): Countdown OOD "0.48 → 0.54" vs random and "0.32 → 0.54" vs the difficulty-ordered curriculum; Zebra "0.29 → 0.35"; AIME "0.075 → 0.10". For Qwen2.5-7B, SEC is "closer to the random curriculum on tasks like Countdown and ARC"; the gap returns on a harder Countdown split (Table 2).
- "the difficulty-ordered curriculum often yields suboptimal performance, likely due to its fixed difficulty schedule … models may spend excessive training time on easy problems" (§3.2).

## Evidence and limitations
- Comparisons are at equal RL training steps (my reading of the setup; not re-verified per table). Small absolute differences on some tasks; seed counts not checked.

## Connections and questions
- Most direct template for our ladder: rungs (L_true bins) = arms; replace the fixed T1 schedule with a Boltzmann bandit whose reward is the rung's mean |advantage| (for binary reward with group-normalised advantages this is a monotone function of p(1−p) within the rung — our derivation, not the paper's). Evaluation "train on easy levels, test on the held-out hardest level" maps exactly onto L* (B1).
- Their fixed easy-to-hard baseline losing to random is a warning about our T1 schedule; a "uniform over rungs" control is cheap and worth running alongside.
