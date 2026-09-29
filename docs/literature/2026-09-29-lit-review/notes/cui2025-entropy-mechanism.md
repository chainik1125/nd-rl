---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Entropy mechanism of RL for reasoning LMs: R = −a·exp(H) + b, Clip-Cov / KL-Cov
Paper: The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models, Ganqu Cui et al., 2025
Source: https://arxiv.org/abs/2505.22617v1 (v1, reviewed)

## Learnings
- Empirical law (§2.4): without entropy or KL intervention, validation performance follows `R = −a·exp(H) + b`. So the ceiling at H = 0 is `R = −a + b`. §2 once writes the law as `R = −a exp(H + b)` (a typo inside the paper).
- "73% of the entropy consumption and 76% of the performance gain occurred in just the first 200 gradient steps (1/12 of training)" (§2.3). Fig. 1: "Over 95% entropy drop/performance gains take place at the early stage".
- Coefficients fit on the first 36 steps predict the next 200 steps (§2.4, Fig. 5). Experiments cover 11 base models across 4 families, including Qwen2.5 0.5B–32B (§2.2).
- Mechanism (§3): for softmax policies, the entropy change ≈ −Cov(log π(a|s), change in logit). Under policy gradient the logit change is ∝ advantage, so the covariance stays mostly positive and entropy falls monotonically.
- Method (§4.2, Eq. 10–13): token-wise `Cov(y_i) = (log π(y_i) − mean log π)·(A(y_i) − mean A)`. Clip-Cov detaches a random r·N subset of tokens whose Cov lies in [ω_low, ω_high] (both set ">500×" the mean). KL-Cov applies a KL penalty to the top-k proportion (k ≪ 1) by Cov.
- Table 2 (§4.3; Qwen2.5-7B, avg over 7 maths benchmarks; AIME/AMC as avg@32): GRPO 38.6, Clip-higher 38.8, Clip-Cov 40.4, KL-Cov 40.6. Qwen2.5-32B: GRPO 45.8, Clip-higher 47.2 (the extracted row for Clip-Cov 32B was truncated; not verified).
- The authors "conditionally support" Yue et al.'s ceiling claim. They argue the ceiling comes from "the entropy mechanism", not from RL per se (§2.6).

## Evidence and limitations
- The paper has no pass@k evaluation at all (0 hits for "pass@"). It shows pass@1-type gains and higher entropy, not support expansion.
- The law is fit on LLMs with long CoT and "Zero" RL. Whether it holds for a 107-token-vocabulary model with 6–16-line proofs is untested.

## Connections and questions
- B2 (indirect); relevant to our GRPO collapse. It is a cheap diagnostic: log policy entropy per step in our GRPO/EI runs and fit R vs H. If the fit holds, the ceiling is predictable early, which saves compute (B5).
- Clip-Cov is ~10 lines of code. Our tokens per proof are few, so the "top 0.02% tokens" regime may not transfer. Expected value is low for support expansion because the paper has no large-k evidence.
- Q3 link: the entropy story is per-token. Collapse lowers per-step diversity, which works against "new moves".
