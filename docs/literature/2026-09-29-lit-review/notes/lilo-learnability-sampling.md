---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# LILO: sample RL prompts by learnability p(1−p)
Paper: LILO: Learning to Reason at the Frontier of Learnability (Foster, Sims, Fellows, Forkel, Foerster; arXiv listing title "Learning to Reason at the Frontier of Learnability"), 2025
Source: https://arxiv.org/abs/2502.12272v6

## Learnings
- Claim (Abstract; §3): policy-gradient methods "provably fail to learn from questions that are too hard … or too easy"; expected improvement is "constrained by the the learnability of the questions", learnability = variance of success p(1−p).
- Method (§4, Alg. 2): sample a candidate pool |D|, roll out N_learnability attempts each, compute p̂(1−p̂), train on the top-|B|. Defaults (§5): "|D| = 4 × |B| and N_learnability = 8"; GSM8K late in training needed |D| = 8×|B|.
- Results (Table 2; §6): speed-ups in training steps 2.5x (PPO/MATH), 1.9x (PPO/GSM8K), 3.2x (VinePPO/MATH), 3.3x (VinePPO/GSM8K), 1.5x (GRPO/ORZ57K); final test accuracy 19.1→21.8, 51.1→53.2, 22.8→24.9, 53.2→55.9, 35.5→37.1. Models Rho-Math-1B and Qwen-2.5-1.5B.
- Compute caveat (§4): with small N_train (=8, their GRPO/PPO), "the sampling overhead is approximately 4×"; speed-ups are in optimizer steps, not samples. They argue the baselines plateau lower (§4 point 2). App. D.1 gives an adaptive-allocation variant (2 attempts per candidate, then refine) with "no additional sampling cost" — "initial results" only (§9).
- §7: curriculum emerges (easier first on MATH); "Learnability doesn't solve RL's generalisation gap"; "Learnability isn't a silver bullet" — on the already-RL-trained Oat-Zero-1.5B "performance only minimally improves".
- §9: LILO "is also stateless"; suggests using learnability to "produce new training data", citing ACCEL-style mutation.

## Evidence and limitations
- Gains are 1.6–2.7 points absolute; single runs per setting as far as the text shows. Not an equal-sample comparison. Math QA with LLMs, not theorem proving.

## Connections and questions
- B1/B5: in our EI, targets with p=0 at the per-round budget contribute nothing and targets with p≈1 contribute redundant data. Smallest adaptation: re-weight ladder targets each round by p̂(1−p̂) estimated from the previous round's attempts (stateful — LILO is stateless, PLR shows staleness matters), with a floor so p̂=0 rungs keep a small share (support expansion at p=0 is what B2 is about, and learnability-only selection would starve it).
- Caution: at pass@400k-type rarity (p ~ 1e-5), p(1−p) ≈ p, so learnability selection reduces to "sample where you already succeed"; it will not by itself push past L* when frontier rungs have p̂ = 0 at the per-round budget.
