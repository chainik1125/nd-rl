---
written_on: 2026-11-09
written_by: agent:claude
status: proposal 45, written at Dan's request 2026-11-09. Not approved, not run. It reuses `olmo-transplant`'s stored traces and pipeline.
---

# Proposal 45: when the pre-RL model fails on RL's thinking, is it only how it writes the answer? (`olmo-answeronly`)

## The short version

`olmo-transplant`
([summary](../../experiment-summaries/2026-11-08-olmo-transplant-dpo-given-rl-thinking-0p14-vs-rl-0p92-genuine-on-41-ifbench-reach-prompts-readings-partly-and-graded/README.md))
handed the pre-RL model (`dpo`) the RL model's (`s2300`) complete, successful thinking and let it write only the final
answer. It wrote a genuine answer 0.14 of the time; the RL model, on the same thinking, 0.92.

That gap has two very different explanations, and the run did not separate them:

- **A habit.** `dpo` understands the thinking and could give the drafted answer, but it writes the way it always
  writes: an opening line, the answer, then an explanation. The extra words break the constraint. One instruction
  would fix it.
- **A skill.** `dpo` cannot use the thinking. It does not find or trust the draft in a long trace written in another
  model's style, and solves the task again on its own, badly. No instruction fixes that.

This proposal repeats the transplant with the habit removed, in four cheap ways, and sees how much of the 0.14 → 0.92
gap closes. Nothing is trained and no new thinking is generated.

## Why the transplant result leaves this open

The summary's own mechanism counts point both ways.

- **For the habit:**
  - `dpo`'s transplant answers have a median of 126 words against 26 for RL's own answers.
  - 256 of its 920 answers (28 %) contain RL's drafted answer word for word, and 183 of those fail, every one by
    adding words.
  - `olmo-iffu` had already shown that "Answer in at most 60 words." unlocks `dpo` on 16 of 48 reach prompts with no
    transplanted thinking at all.
- **For the skill:**
  - Only 43 of 920 answers (4.7 %) copy the draft exactly, and 53 % of `dpo`'s successes share less than half their
    word 3-grams with the draft. The review's words: "given RL's thinking, `dpo` mostly writes its own answer rather
    than the one drafted."
  - The RL model copies its draft exactly in 66 % of answers.

So about a quarter of the failures are visibly "right answer, wrapped in extra words". For the other three quarters
we do not know whether `dpo` ignored the draft out of habit or could not use it.

## Design

**Prompts and traces:** `olmo-transplant`'s, unchanged: the 41 IFBench reach prompts, the 115 stored `s2300` genuine
traces, and its 10 calibration prompts. Traces are inserted by stored token ids, as before.

**Conditions.** Each is the transplant (RL's full thinking, then the writer produces only the answer) plus one change.
Every condition runs with both writers, 8 answers per prompt × trace, at the same decoding.

| | change | what it removes |
|---|---|---|
| T1 / T2 | none (reuse `olmo-transplant`'s rows; do not re-sample) | baseline: 0.14 / 0.92 |
| A1, instruction | append to the user turn: "After thinking, reply with only your final answer, exactly as drafted, with no introduction or explanation." | the habit, by asking |
| A2, brevity | append `olmo-iffu`'s hint, "Answer in at most 60 words." | the habit, by the hint already known to work without thinking |
| A3, closing thought | append one sentence to the end of the thinking, before `</think>`: "I will now output exactly the final draft above and nothing else." | the habit, from inside the thinking |
| A4, prefill | after `</think>`, supply the first 5 tokens of the trace's own final answer; the writer continues | the opening line only; trailing explanations can still occur |

- **A1 and A2 change the prompt, so the thinking was written for a slightly different prompt.** The `s2300` writer
  under the same change is the control for that: if an instruction breaks the RL model too, the condition says
  nothing (`olmo-iffu` saw the brevity hint break `s2300` on 6–7 prompts).
- **A3 is the one I expect to be most informative.** If one sentence of thinking closes the gap, the "missing skill"
  is a single step that `dpo` reads and follows.
- **A4 is graded on the prefill plus the continuation.** Report separately how often the writer stops where the draft
  stops.

**Optional arm, if `olmo-ifladder`'s review stands.** That run's unreviewed status line says a LoRA of `dpo` on 16
genuine `s2300` trajectories reaches 40 of the 48 reach prompts, and a control LoRA on `dpo`'s own successes reaches
none. The adapters are in the bucket. Two cells would show which half the 16 examples taught:
- the LoRA as *writer* on RL's thinking (did it learn to write the draft out?);
- plain `dpo` as writer on the LoRA's own successful thinking (did it learn to think the way RL does?).

**Grading:** the strict IFBench checker plus the blinded hand gate, with `olmo-transplant`'s rules and read caps.
Report strict and genuine, and the exact-copy and near-copy shares the review computed.

## Readings, with thresholds to pre-register

Let R = (`dpo`'s genuine rate) ÷ (`s2300`'s genuine rate) within a condition, on prompts where the condition does not
break `s2300` (its rate stays ≥ 0.75). Take the best of A1–A3 per prompt set, chosen on the calibration prompts, not
on the reach prompts.

- **R ≥ 0.75: a habit.** With the thinking in hand, `dpo` lacks only output discipline. Then RL's whole gain on these
  prompts is *producing* that thinking, and the next test is whether `dpo` can produce it (proposal 42's
  prefix-seeded sampling).
- **R ≤ 0.40: a skill.** Even told to, `dpo` cannot turn RL's thinking into the answer. Following a long trace to its
  conclusion is itself something RL (or 16 trajectories, per `olmo-ifladder`) taught.
- **Between:** report it per constraint family; the families probably differ.
- **A4 high while A1–A3 stay low:** `dpo` can continue the draft once started but does not start it, which is the
  habit located at the first few tokens.

The blog post's "is the RL model better at thinking?" is answered by this run together with what is already measured:
the RL writer on `dpo`'s failed thinking scores 0.06, so writing discipline alone is not enough; this run says whether
thinking alone is.

## What it does not test

- Whether `dpo` can generate such thinking itself. That is proposal 42.
- Anything outside these 41 IFBench prompts and these traces, which were picked from RL's successes.
- "Genuine" is still judged by AI readers. The human calibration set in
  [`docs/writeups/2026-11-09-ifbench-hand-label-calibration/`](../writeups/2026-11-09-ifbench-hand-label-calibration/README.md)
  is the check on that.

## Cost

`olmo-transplant` sampled 17,472 answers for $17.82. This run is 4 new conditions × 2 writers × 920 answers on reach
(7,360) plus the calibration prompts, all prefill-heavy with short outputs: **≈ $10–20, budget $30**. The optional
LoRA arm adds about $5–10 through vLLM's LoRA path. These are projections from that run's prices, to be re-costed at
stage 0.
