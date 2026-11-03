---
written_on: 2026-11-02
written_by: agent:claude
status: approved by Dan 2026-11-02, up to $50 (running as `olmo-rlzero-pilot` on agent1)
---

# Proposal 34: does RL from the base create reach along Olmo 3.1 7B RL-Zero-Math's trajectory? (pilot for the Olmo 3.1 Think 32B test)

**Question (Dan):** our controlled results say RL mostly amplifies what pretraining contains. Does that hold for a real
open model with public RL checkpoints?

Olmo 3.1 Think 32B is the near-frontier test:
- ≈ 14 % of its pretraining GPU-hours went to RL;
- 46 RL checkpoints are public;
- no create-vs-elicit test exists above ≈ 2 × 10⁻² RL share.

It is expensive, though: about $150–250, by an untested estimate. This pilot runs first, on the 7B RL-Zero-Math model:
- RL starts straight from the base;
- there are 28 checkpoints;
- its RL data are public;
- its RL share is ≈ 10⁻³.

| step | what |
|---|---|
| models | `Olmo-3-1025-7B` (base) and `Olmo-3.1-7B-RL-Zero-Math` at about 6–7 log-spaced RL steps up to 2,800 |
| problems | about 90–120 from AIME 2024–2026 (after the Dec 2023 cutoff), checked for overlap with the RL data |
| sampling | k 512 for the base and the final checkpoint, k 64 for intermediates, the base extended adaptively to k 2,048 on problems RL solves; AI2's own template and verifier |
| measures | pass@k; per-problem entries and exits against cumulative RL ÷ pretraining compute; the base's likelihood of RL's new solutions; nearest RL training prompt for each new solve |
| decision memo | measured throughput and a costed design for the 32B test |

**Novelty** (2026-11-02 search): no published work runs pass@k across Olmo 3's RL checkpoints. The closest is "What
Outcome RL Buys" (arXiv 2610.34894), which uses 20 released RL models' final checkpoints at reach@8 / 32 and finds
reach ≤ 2.1 points.

**Budget** $50 / 90 pod-hours, with a stage gate after the throughput measurement. **Run** `olmo-rlzero-pilot` on
agent1, `--effort max`, 72 h hard stop. **Brief:** [followups/BRIEF_olmo-rlzero-pilot.md](followups/BRIEF_olmo-rlzero-pilot.md).
