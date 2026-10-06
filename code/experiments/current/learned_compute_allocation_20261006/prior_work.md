---
written_on: 2026-10-06
written_by: agent:codex
papers:
  - pmlr-v267-lee25ab
  - pmlr-v267-shen25o
---

# What the prior methods establish for this experiment

ReVISE explicitly prefers ending after a correct solution and refining after an incorrect one. It trains verification first and then correction with supervised fine-tuning and preference optimization. This supports the feasibility of learning an internal acceptance signal, while providing a different learning objective from a correctness-only stopping experiment. Its confidence-weighted answer voting also differs from choosing a valid proof of one fixed theorem [@pmlr-v267-lee25ab]. Source checked: [methods §§3.2–3.3](https://arxiv.org/html/2502.14565v2).

Satori uses a format-tuning stage and reinforcement learning with restart states. Its reward combines final correctness, an outcome reward model, and reflection bonuses. Those additional signals guide reflection, so the paper does not isolate stopping learned from final correctness alone [@pmlr-v267-shen25o]. Source checked: [methods §4 and reward design](https://arxiv.org/html/2502.02508v3).

Our inference for the sprint is to distinguish three claims: a learned recognition function; learned allocation of independent attempts; and adaptation of further proof generation conditioned on earlier attempts. Success at the first does not establish the other two. Cost-free stopping requires additional causal continuation measurements because prior valid candidates remain available and oracle pool success cannot fall merely by adding candidates.

The two papers are motivation rather than evidence about this 9.56M-parameter proof-state model. The present sprint must measure its own selection headroom, interface effects, and continuation value.
