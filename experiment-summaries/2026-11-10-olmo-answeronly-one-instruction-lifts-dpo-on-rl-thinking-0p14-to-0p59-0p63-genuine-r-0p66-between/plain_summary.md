# olmo-answeronly in plain language

**The models.** Two versions of one 32-billion-parameter open model (Olmo 3) that "think" in writing before
answering. `dpo` is the version before reinforcement learning (RL); `s2300` is the same model after 2,300 RL steps. On
41 instruction-following tasks with tricky rules, the RL model succeeds and the pre-RL model essentially never does.

**What was already known.** An earlier run pasted the RL model's own successful thinking into the pre-RL model, which
then only had to write the final answer. It still mostly failed: 14 % genuine successes, against 92 % for the RL model
reading the same thinking. Its answers were long, often wrapping the right answer in extra text that broke the rules.

**The question.** Is that a padding habit, or a missing skill?

**What we did.** We repeated the transplant with one change at a time, each aimed at the padding habit:

1. an instruction after the request: "reply with only your final answer, exactly as drafted, with no introduction or
   explanation";
2. the hint "Answer in at most 60 words";
3. one sentence added to the end of the pasted thinking: "I will now output exactly the final draft above and nothing
   else";
4. the first few words (up to five tokens) of the RL model's own answer written in for it.

An automatic checker graded 12,440 new answers; blinded AI readers checked a sample of passing answers (478) for whether they really did the task.

**What we found.**

- **The instruction works best by far.** The pre-RL model's genuine rate rose from 14 % to 63 %; the RL model stayed at
  93 %. One sentence closed about 60 % of the gap.
- **But not all of it.** About 40 % of the gap remains even when the model is told exactly what to do. About
  two-thirds of its answers are still not exact copies of the drafted answer, and it does worst (as does the RL model)
  on sentence-length rules.
- **The other three changes help much less** (they reach 22–32 %, from 14 %). Starting the answer for it barely helps.
- **A side test.** The pre-RL model fine-tuned on just 16 of the RL model's worked examples (an earlier run) uses the RL
  model's thinking three-quarters as well as the RL model does (68 %), with no instruction. The plain pre-RL model, given that
  fine-tuned model's thinking, again mostly fails (18 %).

**What it means.** Given good thinking, the RL model gives its drafted answer far more often than the pre-RL model, and one
instruction recovers most of the difference. Part of the gap is something more: with both the thinking and the instruction in
hand, the pre-RL model still succeeds only about two-thirds as often as the RL model.

**Limits.** 41 prompts chosen because RL beats the pre-RL model there; the pasted thinking was picked from the RL
model's successes; "genuine" is judged by AI readers, not humans. Cost $9.27.
