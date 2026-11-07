---
written_on: 2026-11-07
written_by: agent:claude (executor, run olmo-iffu; reviewed 2026-11-07, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-07).** This is the plain-language summary of run `olmo-iffu`, written by the
> executor. The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-iffu`, file `olmo_iffu/plain_summary.md` at
> [`e57060d0`](https://github.com/dan-pandori/nd-takehome/blob/e57060d0/olmo_iffu/plain_summary.md). The corrections
> from its [review](https://github.com/dan-pandori/nd-takehome/blob/c0c6e053/review_olmo-iffu.md) are applied, and each
> change is marked *[review]*:
> - perp_069 was a reach prompt only under the doubly-credited reading: the DPO model already had two
>   RL-verifier-credited responses in 256 that read as sound (§Compare; phase-2 corrections);
> - the geometry result is one of three puzzles (perp_068), not "two … at least twice … mostly a length limit"
>   (§Verdict, "Not supported" and reword item 1);
> - the RL model's "about 70 %" is `omega` checker credit, not hand-sound (reword item 1);
> - the hint breaks the RL model on 6 more IFBench prompts, or 7 if `ifbench_170` counts, not 8 (reword item 2);
> - step 750's 26 of 43 is within 32 fresh draws, so the pre-registered ≤ 64 count is unmeasured (reword item 6);
> - 17 of the 52 are explained, not 18 (§Verdict, "Not supported");
> - "every deciding one included" is not quite true: `ifbench_226`'s RL check stopped at 5 of 16, and one `dpo` walk
>   (`ifbench_78`) skipped three items (§Compare; reword item 5).
>
> Each mark follows the passage it covers.
>
> One note is the librarian's, marked *[librarian]*: RunPod's billed total. With the marks the text grows from 498 to
> 716 words (`wc -w` from the title down, marks included), against the brief's 500.

# olmo-iffu in plain words

**The question.** An earlier run compared a 32-billion-parameter language model before and after reinforcement
learning (RL). It found 52 prompts the RL model solves and its starting point never solved in 256 tries ("reach"). One of them, the
geometry puzzle perp_069, is a reach prompt only if an answer credited by just one of the two automatic maths checkers
does not count: the starting model had already written two such answers to it, and both read as sound *[review]*.
Reach could mean RL taught something new, or three duller things: the starting model needs more room to finish, it
fails because it writes too much (the RL model's answers are much shorter), or 256 tries are too few. This run
tested each.

**Test 1: more room.** Three reach prompts are geometry puzzles (how many drawn diagonals of a polygon meet at right
angles). The starting model ran out of its 32,768-token budget on nearly every try. With the model's full context,
about 65,000 tokens, it finished almost every time, after about 46,000 tokens. Checking every credited answer by
hand, and counting each puzzle the way it was a reach prompt *[review]*, it solved one puzzle (perp_068) correctly at
least twice in 64 tries and the other two once each. So the length limit explains one of the three geometry reach
prompts *[review]*. But the starting model is right only 2–5 % of the time, while the automatic checker credits the RL
model on two of these puzzles about 70 % of the time within the old budget *[review]* (28 of 40 tries each, in the
earlier run). That is checker credit, not a hand check: the earlier run's review read 6 of 9 such credits on one of
the puzzles as sound, so the RL model's rate of sound answers is unmeasured and probably lower *[review]*.

**Test 2: a brevity hint.** We added "Answer in at most 60 words." to instruction-following prompts. The starting
model's answers shrank from about 230 words to about 60. It then solved 4 of the 6 IFEval reach prompts (tasks like
"use the letter w at most once" or "write in capitals") and 12 of the 42 IFBench ones. On 6 more IFBench prompts (7 if
one judgement call goes the other way) *[review]* the hint also broke the RL model, because squeezing, say, seven
uses of one keyword into 60 words forces gibberish. Those 6–7 *[review]* tell us nothing. About a third of the instruction-following reach is a verbosity habit that a one-line prompt
change unlocks: elicited, not created.

**Test 3: more tries.** On 10 IFBench reach prompts we sampled the starting model 1,024 times, four times as often
as before. It never produced a genuine success, so its success rate there is below 0.3 % per try (95 % bound).
Yet 4 of those 10 prompts became easy with the brevity hint. Unfindable in 1,024 tries need not mean absent.

**Test 4: an intermediate RL checkpoint.** The model a third of the way through RL (step 750) already solves 26 of
the 43 IFBench reach prompts, at least twice within 32 tries. The plan allowed up to 64 tries, so with more it could
be more *[review]*.

**What is left.** Of the 52 reach prompts, 17 *[review]* are now explained: 1 *[review]* by the token budget, 16 by brevity. Two prompts
survived both 1,024 tries and the hint. They are the best candidates for something RL really added. Most of the rest
were tested only with the hint.

**Caveats.** 368 credited answers were read by hand, nearly every deciding one included. Two reading walks were
incomplete: one stopped early and one skipped three answers. Read to the end by the review, the hint also breaks the
RL model on one prompt where the starting model succeeds anyway (by the majority of three readers, though not under the
first readers alone), and no result for the starting model changes *[review]*. Many checker passes were gaming,
such as keyword stuffing or text without sentence breaks that slips past a per-sentence rule. A few labels are
judgement calls and are flagged in the files. One prompt counts only if ALL CAPS counts as title case. Cost: $53.87
of $60 (*[librarian]* $54.34 as billed by RunPod, disk included).
