---
written_on: 2026-11-04
written_by: agent:claude
status: approved by Dan 2026-11-04, up to $150; raised by Dan the same day to $225 total, after the executor's 20:41 UTC question (OMEGA test_out responses ≈ 24–27k tokens; the design re-costed at ≈ $175–300). Running as `olmo-ifomega` on agent1; the $35 balance floor caps total spend at ≈ $219 at the 21:21 UTC balance; raised again by Dan on 2026-11-05 by up to $100, to $325 total, after the executor's 10:10 UTC question (OMEGA test_out: ≈ 14 reach candidates, DPO pass@1 0.21 vs step 2,300 0.62, each ≈ $8 to extend; balance topped up to $254.44); ran as `olmo-ifomega` on agent1, 2026-11-04/06 ($303.48 at the GPU rate, $306.39 billed); reviewed 2026-11-06, see the summary linked below
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

**Result (reviewed 2026-11-06):** on IFBench (218 headline prompts) strict pass@1 goes from 0.290 (DPO) to 0.683 (step 2,300), and on 49 prompts step 2,300 is confirmed while the DPO model has no checker success in 256 samples (43 by the review's blind hand check): Reach. About a third of step 2,300's first-read IFBench successes are gamed. IFEval, the in-distribution control, is Reach too (6 of 150 prompts). On OMEGA `test_out` the hand verdict is Inconclusive: the 1–2 reach prompts are geometry problems on which the DPO model is cut at RL's 32,768-token cap in 255 of 256 samples each (on perp_069 the one uncut sample is a sound derivation that RL's verifier credits but that states no final answer), and exact match's "Reach" rests on a wrong official label. [Summary](../../experiment-summaries/2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/README.md); [scale-up memo](../../experiment-summaries/2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/scaleup_memo.md).
