---
written_on: 2026-11-04
written_by: agent:claude
status: approved by Dan 2026-11-04, up to $150 (running as `olmo-ifomega` on agent1)
---

# Proposal 36: does Olmo 3.1 Think 32B's RL create reach where RL actually moved it, on IFBench and OMEGA?

**Question (Dan):** the AIME test (`olmo-think32b`,
[summary](../../experiment-summaries/2026-11-03-olmo-think32b-one-reach-candidate-dpo-credited-7-in-256-by-rl-verifier-states-answer-once-verdict-inconclusive/README.md))
found one reach candidate. But RL barely moved AIME pass@1 (0.716 → 0.746), so AIME is a weak probe of what RL
installed. Test where RL moved the model most, which is also what it trained on most:
- precise instruction following: IFBench +34 points DPO → step 2,300 (AI2's evals), from 29.8k IF prompts in RL;
- OMEGA maths: +8 points, from 15k OMEGA prompts in RL.

| step | what |
|---|---|
| tasks | IFBench (300, constraints new to training; strict checkers), IFEval (in-distribution control), OMEGA-explorative `test_in` (held-out problems at the training difficulty) and `test_out` (harder), about 8–12 families |
| models | `Olmo-3-32B-Think-DPO` (the RL start); `Olmo-3.1-32B-Think` at steps 100, 750 and 2,300 |
| sampling | as `olmo-think32b`: DPO k 16, RL k 8; symmetric extension to a second success, up to 256 |
| grades | the official checkers (IF strict prompt-level; OMEGA's answer checker); RL's maths verifier for OMEGA; a hand-check of every deciding response (IF: gamed vs genuine) |
| contamination | every eval prompt against all of `Dolci-Think-RL-32B` |

**Budget** $150 / 120 pod-hours, with a stage gate after measuring response lengths. Cut in this order if needed:
IFEval, then OMEGA `test_in`, then step 100. **Run** `olmo-ifomega`, `--effort max`, 72 h hard stop. **Brief:**
[followups/BRIEF_olmo-ifomega.md](followups/BRIEF_olmo-ifomega.md).
