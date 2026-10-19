---
written_on: 2026-10-19
written_by: agent:claude
status: approved by Dan 2026-10-19, no pods (running as `rl-compute-share` on agent2 since 17:27 UTC)
---

# Proposal 30: is the RL in create-vs-elicit papers a realistic share of pretraining compute?

**Question (Dan):** Yue et al. [@yueDoesReinforcementLearning2025] conclude that RLVR elicits rather than creates
capabilities. But the RL in such papers looks tiny next to the base model's pretraining compute. If so, they answer
"does a very small amount of RL create new capabilities?", and they cannot rule out creation at frontier-lab RL scale.

| step | what |
|---|---|
| Yue in numbers | RL / pretraining compute (FLOPs, tokens, GPU-hours) for every checkpoint Yue evaluates |
| wider literature | the same for 20–30 RL-on-pretrained-model papers, with each paper's create/elicit verdict |
| frontier | graded public statements and estimates of the frontier labs' RL share, as a range |
| scale dependence | whether verdicts change as RL compute grows; the strongest evidence against the hypothesis; the right normaliser |
| our runs | the same ratio for create-or-elicit, rl-continue, rl-from-ckpt and trajectory |

**Budget** no pods ($0.50 guard). **Run** `rl-compute-share` on agent2, `--effort max`, 48 h hard stop. **Brief:**
[literature/BRIEF_rl-compute-share.md](literature/BRIEF_rl-compute-share.md).
