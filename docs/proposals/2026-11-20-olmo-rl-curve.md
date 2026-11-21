---
written_on: 2026-11-20
written_by: agent:claude
status: proposal 46, written at Dan's request 2026-11-20 from Dmitry's suggestion. Not yet approved or launched.
papers:
  - yueDoesReinforcementLearning2025
---

# Proposal 46: how instruction following changes over the course of RL (`olmo-ifcurve`)

## The short version

Dmitry suggested looking at how IFBench performance changes over the course of RL training. So far every Olmo 3.1
Think 32B comparison has used two or three points of one RL run:
- the DPO start;
- step 750;
- step 2,300;
- plus step 100 on AIME, and on IFBench at 8 draws.

AI2 released 46 checkpoints of that run (`step_0050` … `step_2300`, every 50 steps; checked on Hugging Face
2026-11-20), so the whole trajectory is available.

Endpoints cannot show three things that a curve can:

1. **When each reach prompt becomes solvable, and how.** There are 48 instruction-following reach prompts: the DPO
   start has no genuine success in 256 draws on any of them, and none in 1,024 on 10. Does each switch on abruptly,
   or does its success rate climb gradually from a small value? A gradual climb from rates near 1 in 1,000 is what
   rare-event amplification (claim 6) predicts.
2. **Whether RL learns the habit before the skill.** `olmo-ifladder` sorted the 48 prompts into three classes:
   - 25–26 that prompting the DPO start recovers;
   - 15–16 that only a LoRA on 16 RL trajectories recovers;
   - 7 that neither recovers.

   `olmo-answeronly` found that, given RL's thinking, the DPO start fails largely through how it writes the answer.
   If RL learns that habit first and the harder skill later, the prompt-recoverable prompts should be reached early
   and the 7 "neither" prompts late.
3. **Whether gaming comes first.** About 36 % of step 2,300's first-read checker wins are gamed. Does RL first find
   checker-passing shortcuts and only later genuine answers, or the reverse?

It also gives Dmitry's aggregate curve: IFBench pass@1 and pass@8 by RL step, next to the nearly flat AIME curve we
already have.

## What we already have

- **IFBench and IFEval** at `dpo`, `s0750` and `s2300`, and IFBench at `s0100`
  ([olmo-ifomega](../../experiment-summaries/2026-11-04-olmo-ifomega-reach-on-ifbench-49-strict-43-hand-and-ifeval-6-omega-test-out-inconclusive/README.md)).
  The fixed phase drew 16 per prompt for `dpo` and 8 per RL checkpoint, with extensions on reach candidates.
- **`s0750` on the 43 IFBench reach prompts** at 32 draws, confirmed on 26 of the 43
  ([olmo-iffu](../../experiment-summaries/2026-11-06-olmo-iffu-brevity-hint-elicits-dpo-on-15-16-of-48-if-reach-prompts-10-of-10-survive-1024-draws-token-budget-on-1-of-3-geometry/README.md)).
- **AIME 2025–26** at `dpo`, `s0100`, `s0750` and `s2300`: pass@1 0.716 → 0.746
  ([olmo-think32b](../../experiment-summaries/2026-11-03-olmo-think32b-one-reach-candidate-dpo-credited-7-in-256-by-rl-verifier-states-answer-once-verdict-inconclusive/README.md)).
- **The per-prompt classes**
  ([olmo-ifladder](../../experiment-summaries/2026-11-09-olmo-ifladder-lora-on-16-rl-trajectories-elicits-40-41-of-48-if-reach-prompts-control-0/README.md)).
  The 7 "neither" prompts are `ifeval_3478` and IFBench 3, 67, 166, 188, 268 and 284.
- **The answer-habit result**
  ([olmo-answeronly](../../experiment-summaries/2026-11-10-olmo-answeronly-one-instruction-lifts-dpo-on-rl-thinking-0p14-to-0p59-0p63-genuine-r-0p66-between/README.md)),
  and the hand-gate rules and readers from those runs.

## Design

**Model and checkpoints.** `allenai/Olmo-3.1-32B-Think` at nine RL steps, plus its DPO start
(`allenai/Olmo-3-32B-Think-DPO`): 0, 50, 100, 200, 350, 500, 750, 1,100, 1,600 and 2,300.
- The spacing is roughly logarithmic, because early RL moves fastest.
- Reuse stored samples wherever the protocol matches; that covers most of `dpo`, 750 and 2,300. About seven
  checkpoints are new.
- Decoding is `olmo-ifomega`'s: the Think RL settings, with a 32,768-token response cap.
- `allenai/Olmo-3-32B-Think` (Olmo 3.0) is the first 750 steps of the same run, so it adds nothing here.

| arm | prompts | draws per checkpoint | what it measures |
|---|---|---|---|
| A, aggregate curve | the 218 IFBench prompts left after `olmo-ifomega`'s SFT exclusion | 8 | strict pass@1 and pass@8 by step, overall and per constraint family, with prompt-bootstrap intervals |
| B, reach emergence | the 48 reach prompts (42 IFBench, 6 IFEval) | 32 | each prompt's genuine rate by step, and its first-genuine step |
| C, gaming over training | no new draws | | the blinded hand gate on every deciding success in B, plus a seeded sample of up to 40 strict successes per checkpoint from A; the gamed share by step |
| D, lengths | no new draws | | median thinking and answer lengths by step, overall and on the reach prompts |

**Arm B's extension.** At each prompt's first checkpoint with a genuine success, extend the checkpoint before it to
128 draws. That bounds the rate just before the prompt switches on.

**Optional arms**, in the order to cut them:
- **F, math contrast:** AIME at the six new checkpoints, 8 draws each. It fills in the flat curve; cut it first.
- **E, a second model:** `allenai/Olmo-3-7B-Think`, which has the same recipe, its own DPO start, and 55 checkpoints
  every 25 steps. Arm A only, at 14 checkpoints every 100 steps with 4 draws each. It gives only the aggregate curve,
  because the 7B model's reach set is unknown.
- **B+, densify:** for a seeded 10 reach prompts, sample every 50-step checkpoint between the last one with no
  success and the first with a rate ≥ 0.25.

**Grading:** the strict IFBench and IFEval checkers, plus the blinded hand gate under `olmo-ifomega`'s rules with
`olmo-iffu`'s addendum.

## Readings, with thresholds to pre-register

- **Timing by class (point 2).** Compare the median first-genuine step of `olmo-ifladder`'s three classes.
  - **"Habit first":** the prompt-recoverable median is below the teachable median, the teachable median is at or
    below the "neither" median, and the prompt-recoverable vs rest difference has a prompt-bootstrap 90 % interval
    above 0.
  - **"No order":** anything else.
- **Gradual or abrupt (point 1),** per reach prompt that RL reaches:
  - **gradual:** at least two sampled checkpoints lie between the first with any genuine success and the first with a
    rate ≥ 0.25;
  - **abrupt:** the prompt goes from 0 of 128 to a rate ≥ 0.25 at the next sampled checkpoint.

  Report the counts. If most prompts are gradual, report the earliest detectable rates; that is claim 6's test.
- **Gaming (point 3).** Compare the gamed share among strict successes at early checkpoints (step 200 or earlier) with
  late ones (step 1,100 or later).
  - **"Gaming first":** early exceeds late by ≥ 0.2.
  - **"Gaming grows":** late exceeds early by ≥ 0.2.
  - **"Flat":** otherwise.
- **Brevity.** Find the step by which median answer length is within 20 % of its step-2,300 value, and the step by
  which half the reach prompts are reached. **"Brevity first"** if the first is earlier.
- **Aggregate shape.** Compare pass@8 gains with pass@1 gains by step.
  - **Widening:** both rise together.
  - **Sharpening:** pass@1 rises toward a flat pass@8.

## What it does not test

- **Why a prompt switches on.** The curve times it; the mechanism needs `olmo-ifladder`- or `olmo-answeronly`-style
  probes at the switch point, which are a natural follow-up.
- **Other RL runs.** This is one run with one seed. The checkpoint survey
  ([open RL checkpoints](../literature/2026-11-08-open-rl-checkpoints/README.md)) names Tülu 3.1 8B (61 checkpoints)
  and Nemotron-Cascade-8B (stage checkpoints) as the cross-family repeats.
- **Rates near 1 in 1,000.** With 32 draws, a rate below about 3 % often shows as zero. Arm B's extension to 128
  resolves rates of 1–2 % just before a prompt switches on. Rates near 1 in 1,000 would need several thousand draws,
  beyond this budget.

## Cost

Draw costs come from earlier runs on community H200s:
- about $0.0023 per DPO draw on IFBench (`olmo-iffu` test 3);
- RL checkpoints write about 1.5× longer (`olmo-ifomega`), so about $0.003–0.0035 per draw.

| item | draws | cost |
|---|---|---|
| arm A, 218 × 8 × 7 checkpoints | 12,208 | ≈ $40 |
| arm B, 48 × 32 × 7 checkpoints | 10,752 | ≈ $35 |
| arm B's extensions | ≈ 4,600 | ≈ $15 |
| arms C and D | none | GPU-free |
| model loads for 7 checkpoints, and stage 0 | | ≈ $10–15 |
| **total** | | **≈ $100–105; budget $120** |

Optional arms: E ≈ $10–15; F ≈ $20–25, since AIME draws are long. All are projections, to be re-costed at stage 0.
