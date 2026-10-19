---
written_on: 2026-10-02
written_by: agent:claude
papers:
  - liuProRLProlongedReinforcement2025
---

# ProRL: prolonged RL "expands reasoning boundaries"

Paper: [@liuProRLProlongedReinforcement2025]
Source: https://arxiv.org/abs/2505.24864v1 (v1 reviewed; html text)

## Learnings
Paper claims:
- RL-trained models outperform the base "including scenarios where base models fail entirely regardless of the number of attempts" (abstract). Some Reasoning Gym tasks go from no base solutions to "100% pass rates (Figure 4)" (Sec. 1).
- Recipe: KL penalty plus periodic "reference policy reset" (Sec. 2.3.1), diverse tasks, and "more than 2k training steps" (Sec. 1). The base is DeepSeek-R1-Distill-Qwen-1.5B (Sec. 2).
- Expansion is largest where the base is weakest: there is "a significant negative correlation between the base model's reasoning boundary and the extent of reasoning improvement" (Sec. 4, Fig. 3). Tasks with high base pass@128 show "minimal or even negative gains".
- Novelty is measured with the Creativity Index (n-gram overlap with a pretraining corpus; Fig. 1 middle).

Our interpretation: the "creation" evidence is base pass@128 = 0 on held-out synthetic tasks. The paper itself notes that the base "struggles with formatting" on Reasoning Gym (Sec. 4) and that "formatting is relatively easy to learn" (App.). Some of the 0 → 100% jumps may therefore be format elicitation, not new reasoning. This is the format transient that EDL (2601.04728) warns about.

## Evidence and limitations
- Read the abstract, Sec. 1, 2.3, 4 (Fig. 3) and App. A. Did not check the per-task Figure 4 values.
- The boundary is measured at k = 128 only. There is no base-likelihood analysis of the new solutions, and no control for answer format.
- A single base model; no seeds are reported (no "seed" mention found in the v1 text). The pretraining corpus of the distilled base is unknown.
- Successor BroRL (2510.01180, abstract only): ProRL "plateaus" and is revived by raising rollouts per prompt to hundreds. Exploration width per target, not only steps, is the lever.

## Connections and questions
- `rl-from-ckpt`: our ladder uses k = 32 per target for 8 rounds, which is short and narrow next to ProRL (>2k steps) and BroRL (hundreds of rollouts). The two-stage view (2510.04028) predicts shrinkage or elicitation in an early "exploitation stage" and expansion only later. "Consistent with elicitation" should therefore be qualified as "at 8 × 32". Cheapest fix: one arm with k = 256 per target (or 24 rounds) on C theorems only, to test whether C ever moves.
- ProRL's negative correlation (largest gains where the base is weak) is the opposite of our finding that C theorems never move. The likely reason is that ProRL's "weak" tasks are failures of format or near-support, while our C theorems sit at a reference worst step of about −12 nats. A format-only failure check is cheap for us: is the C failure mode a Lean-syntax or format failure, or a logical one?
- Related earlier notes: zhu2025-negative-reinforcement.md, cui2025-entropy-mechanism.md (entropy collapse, which motivates ProRL's KL reset).
