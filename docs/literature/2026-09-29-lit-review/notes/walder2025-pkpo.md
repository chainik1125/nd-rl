---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# PKPO: unbiased pass@k reward transformation for any k ≤ n
Paper: Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems, Christian Walder and Deep Karkhanis (Google DeepMind), 2025
Source: https://arxiv.org/abs/2505.15201v5 (v5, reviewed)

## Learnings
- Binary rewards (§2.2, Eq. 8): the gradient estimator uses per-sample rewards `r_i = k/n` if correct, `(k/n)·ρ(n−1, c, k−1)` if incorrect, with `ρ(n,c,k) = 1 − C(n−c,k)/C(n,k)` (Eq. 3; Theorem 2: unbiased for ∇pass@k). The paper describes this as assigning "some reward to incorrect samples to encourage exploration". Continuous rewards are handled in §3.
- "our transformations are the first to enable robust optimization of the pass@k for any arbitrary k ≤ n" (abstract). Prior work [42] "couples the minibatch size to k".
- Annealing: train with k_opt = 8 up to step 1500, then k_opt = 1. This "improves pass@k_eval without sacrificing pass@1" (Fig. 3).
- Setup: Gemma2-2B/9B, Llama3.1-8B, n = 16 (§5.2). MATH train split (12,000 problems): "higher k_opt ... leads to a consistently higher cumulative solve rate throughout training, as well as a higher entropy" (§5.2.1, Fig. 6a).
- ARC-AGI-1 easy subset, §5.2.4, Tables 5–6 (cumulative train solve rate / test pass@1 / test pass@16, 3 restarts, trained to saturation). Gemma2-9B: k_opt=1 12.00 / 2.00 / 8.18; k_opt=4 82.33 / 22.00 / 38.18; k_opt=8 84.14 / 26.67 / 44.50; EntropyReg (best of 5 coefficients) 24.67 / 4.00 / 8.89. Llama3.1-8B: k_opt=1 22.00 / 3.33 / 8.00; k_opt=8 88.89 / 29.67 / 43.13. "conventional pass@1 optimization stalls" (Fig. 8).

## Evidence and limitations
- The strongest evidence in this cluster that a set-level objective solves training problems pass@1-RL does not (ARC cumulative solve 12 → 84%). But k_eval ≤ 16 and n = 16; there is no very-large-k measurement.
- With c = 0: ρ(n−1, 0, k−1) = 0, so every r_i = 0 and there is no gradient on all-fail tasks. The effect on "stalled" tasks has to come from tasks with c ≥ 1 moving the shared policy.
- Unlike GRPO, rewards are not group-normalised (§4 adds leave-one-out baselines for variance reduction; I did not check them in detail).

## Connections and questions
- B2 / B1. It is the same family as Chen et al. 2508.10751. PKPO has the unbiasedness proof and the k-annealing result. Implementation is one reward-transform function over our group.
- The ARC "cumulative solve rate" is the analogue of our frozen-control-at-equal-attempts comparison. It is the right metric to copy.
- Smallest test: GRPO (k_opt=1) vs PKPO k_opt=G/2 annealed to 1, same checkpoint, ≥2 seeds. Measure the cumulative distinct theorems solved during training vs the frozen control, and the 400k support curve.
