---
written_on: 2026-09-29
written_by: agent:claude
papers: []
---
# RL learns compositions of known atomic skills; iterative RFT does not
Paper: From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones (Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, et al., 2025)
Source: https://arxiv.org/abs/2509.25123v3 (version reviewed)
## Learnings
- Task (§3.1): 25 string-transformation functions with meaningless names; Level n = n-fold nesting. Stage 1 (RFT) teaches atomic functions; Stage 2 trains compositions (§3.2). Llama-3.1-8B-Instruct.
- RL on Level-1 only: Level 2 "remains below 25%", Levels 3–6 "near zero"; RL including Level 2: Level 3 "from 5% to around 30%", Level 4 "from 1% to 15%" (§4.1, Fig 2).
- Iterative RFT on identical Level-2 data: "on Level 3 it never surpasses 2.6%", Level 2 only 15%, versus RL 64% (L2) and 27% (L3) (§4.2, Fig 3). RFT here = sample, keep correct, SFT, repeat (App. A).
- pass@k (§4.4, Fig 5): on Levels 1–2 the RL–base gap shrinks with k; on Levels 3–6 it widens, e.g. Level 5 "grows from 4% at pass@1 to approximately 25% at pass@1024".
- Atomic skills are a prerequisite for transfer (§4.3, Fig 4).
- RL details: "we use DAPO", 16 rollouts, temperature 1.0, drop all-correct/all-wrong groups, KL and entropy coefficients 0 (App. A; §3.2 says GRPO).
## Evidence and limitations
Single model family; accuracies read from text, not figure data; seeds not reported in what I read; the RFT baseline's number of iterations and sample counts not checked. Pretrained LLM, not from scratch.
## Connections and questions
B2 and B1. Their iterative RFT ≈ our EI (without hindsight); their finding that RFT fails to learn composition while GRPO/DAPO succeeds is consistent with our GRPO "ignition" of a depth-3 pattern from zero base rate, and argues for pushing GRPO (already done/proposed) on deeper-nesting rungs. Key design point: RL must be trained on the *compositional level itself* (Level 2) — training only on atoms gives nothing. For us: include rungs at the target depth/length, not just below. Their pass@k-by-difficulty diagnostic is what our support curves already do. No new method to import beyond that.
