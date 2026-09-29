---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Pass@k Training: analytic pass@k group advantage for RLVR
Paper: Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models, Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, Guang Shi (RUC / ByteDance Seed), 2025
Source: https://arxiv.org/abs/2508.10751v1 (reviewed)

## Learnings
- Reward = pass@k of a k-subset. Three implementations: full sampling (split N_rollout into ⌊N/k⌋ groups), bootstrap sampling, and analytic derivation (§2.2–2.4).
- Analytic form (§2.4, App. 8): `R̄_group = 1 − C(N_neg,k)/C(N_rollout,k)`; `Â_pos = (1 − R̄)/σ`; `Â_neg = (1 − R̄ − C(N_neg−1,k−1)/C(N_rollout−1,k−1))/σ`. The advantage "depends only on" N_rollout, N_pos, N_neg and k (§2.4).
- Built on DAPO, "only retaining the clip-higher and token-level policy gradient loss" (§2.1).
- Table 1 (§3.3; Qwen2.5-7B-Instruct, Pass@1/Pass@k): ARC-AGI-1 base 2.4/4.8, Pass@1 T. 3.3/3.8, Pass@k T. 4.0/5.3. Enigmata 4.8/10.1 → 12.9/21.3 → 17.9/29.8. KORBench 36.5/45.9 → 37.7/45.6 → 47.7/63.5. AIME 2025 4.2/15.8 → 5.4/19.1 → 7.1/22.4.
- Evaluation sample count: "32 responses ... for the Maze task and ... 8 responses ... for other tasks" (§7 Experiment Setup; Pass@k computed from these). So "Pass@k" here means k ≤ 8 (≤ 32 on Maze).
- §4.1 / Fig. 9: the total advantage magnitude η peaks at 50% accuracy for Pass@1 but at 25% for Pass@8. "Pass@k Training focuses on optimizing harder problems."
- §3.1: randomly flipping 10–50% of negative rewards ("noise rewards") degrades performance. Entropy regularisation (0.001–0.005) can collapse the model at high coefficients.
- §4: "implicit reward design" variants (combination / adaptive training) outperform plain Pass@k training on Enigmata (Fig. 12–13).

## Evidence and limitations
- The large-k claim is not tested: k ≤ 8 (≤ 32 on Maze). Nothing like 10^3–10^5.
- It continues from a Pass@1-trained model. Most results are LLM puzzle/maths tasks; I did not check seed counts.
- When N_pos = 0, R̄ = 0 and σ = 0 (my reading of the formula), so there is no gradient on all-fail prompts. The paper does not discuss this case in the parts I read.

## Connections and questions
- B2, and B1 through the η shift toward hard prompts. A cheap drop-in: it changes only the advantage from (N_pos, N_neg, k), which is easy with our binary reward and fixed group size. k must be ≤ group size.
- Relative to our proposed DAPO/Dr.GRPO fixes, the new part is the pass@k advantage itself. Clip-higher is already in the DAPO proposal.
- Smallest test: GRPO vs pass@k-advantage GRPO (k = G/4) on the same checkpoint, comparing support curves at 400k. Watch hard-rung (L_true ≥ 12) theorems with low but nonzero base rate.
