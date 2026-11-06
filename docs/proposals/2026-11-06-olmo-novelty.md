---
written_on: 2026-11-06
written_by: agent:claude
status: approved by Dan 2026-11-06, no pods (queued as `olmo-novelty` on agent2 after `olmo-iffu`)
papers:
  - liuProRLProlongedReinforcement2025
  - yueDoesReinforcementLearning2025
---

# Proposal 38: how novel are our Olmo create-vs-elicit results?

**Question (Dan):** position each finding of the Olmo runs against the literature. Which results replicate, extend or
contradict prior work, and what is genuinely new? Compare in particular with ProRL [@liuProRLProlongedReinforcement2025].

| step | what |
|---|---|
| claims | C1 7B RL-Zero (no reach, no narrowing); C2 32B AIME (RL barely moves it); C3 IFBench reach (43–49 of 218, ≈ 20 % RL-only against Yue's ≤ 1 %); C4 about a third of RL's checker successes gamed; C5 IFEval brevity and OMEGA token cap; C6 DPO failure modes; C7 methods; C8 `olmo-iffu` |
| ProRL | side by side with its critics (Invisible Leash at k up to 16,384 with format-tolerant extraction; RL Squeezes; BroRL); ProRL's "largest where the start is weakest" correlation computed on our stored results |
| searches | large-k boundaries for instruction following (none found so far), domain splits within one RL run, measured checker gaming, trajectory studies using public checkpoints |
| outputs | a novelty matrix, a ranked novel / extends / replicates / contradicted list, the five strongest reviewer objections, a ≈ 600-word related-work draft |

**Budget** no pods ($0.50 guard), Claude time only. **Run** `olmo-novelty` on agent2, queued after `olmo-iffu` is
recorded, `--effort max`, 48 h hard stop. **Brief:** [literature/BRIEF_olmo-novelty.md](literature/BRIEF_olmo-novelty.md).
