---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# Lean first-error position as a tactic-level GRPO reward
Paper: Process-Verified Reinforcement Learning for Theorem Proving via Lean (Kim, Yun et al., KAIST, 2026)
Source: https://arxiv.org/abs/2606.20068v1 (v1 reviewed)
## Learnings
- §3.1 defines per-tactic φ: 1 if the proof passes; d1 if the proof fails but the tactic has no Lean errors; d2 if the tactic contains errors.
- §4.1 adds a First Error Propagation rule: every tactic after the earliest failing one is treated as erroneous.
- Process advantage = φ − mean group outcome, a difficulty baseline. It is added to the GRPO outcome advantage only on the first token of each tactic (§4.2).
- §5.1: 10k STP problems, 15 s Lean timeout, d1 = −0.05, d2 = −0.1, 7B models (STP-Lean, DeepSeek-Prover-V1.5-SFT).
- Results (Table 2, STP): outcome-only GRPO 57.9% MiniF2F pass@64; tactic-only 56.8%; outcome+tactic 59.2%, ±0.5. ProofNet pass@32 is 17.4 (GRPO) vs 18.6 (ours). §5.2 gives +2.5 pp vs the SFT baseline, against +1.2 pp for GRPO.
- Table 4 ablations: "No First Error" 58.2% pass@64 (vs 59.2) and 56.4% pass@32 (vs 57.1). Removing the baseline gives 57.4%. Table 3: first-token credit beats all-token, last-token and high-entropy-token credit.
- §4.2 / App. J: the reward is framed as approximate potential-based shaping. The potential is the probability that the prefix can be completed, and the error state is absorbing with potential 0. Return-based credit was unstable (App. G).
## Evidence and limitations
- Gains are about 1–2.5 pp at 7B and are comparable to the reported ± of 0.3–0.8. The paper does not say how many seeds produced the ±. Tactic-only reward "results in premature convergence" (§5.3), so the outcome term is required.
- There is no pass@k-at-large-k or support analysis, so we cannot tell whether the gain is elicitation or expansion.
## Connections and questions
- This is the only verified source I found that uses first-error position as training credit without reading error text. It fits our setup: the model need not read messages.
- State-env: the env already stops at the first syntactically invalid action. A wrong `have` term is caught only by Lean at the end, and Lean's error line locates the first failing step. So φ is computable with a line→step map.
- For GRPO: give steps before the first Lean error d1 and later steps d2, with first-token credit. For EI it adds little, because hindsight relabelling already harvests the verified prefix.
- Smallest test: one GRPO arm vs outcome-only on T1, same seeds. Expect a small effect at best; measure against our noise floor (a 2-point L* difference is at the edge of resolution at n = 2).
