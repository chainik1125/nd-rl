# olmo-iffu in plain words

**The question.** An earlier run compared a 32-billion-parameter language model before and after reinforcement
learning (RL). It found 52 prompts the RL model solves and its starting point never solved in 256 tries ("reach").
Reach could mean RL taught something new, or three duller things: the starting model needs more room to finish, it
fails because it writes too much (the RL model's answers are much shorter), or 256 tries are too few. This run
tested each.

**Test 1: more room.** Three reach prompts are geometry puzzles (how many drawn diagonals of a polygon meet at right
angles). The starting model ran out of its 32,768-token budget on nearly every try. With the model's full context,
about 65,000 tokens, it finished almost every time, after about 46,000 tokens. Checking every credited answer by
hand, it solved two puzzles correctly at least twice in 64 tries and the third once. So the geometry "reach" was
mostly a length limit. But the starting model is right only 2–5 % of the time, while the RL model solves two of these
puzzles about 70 % of the time within the old budget.

**Test 2: a brevity hint.** We added "Answer in at most 60 words." to instruction-following prompts. The starting
model's answers shrank from about 230 words to about 60. It then solved 4 of the 6 IFEval reach prompts (tasks like
"use the letter w at most once" or "write in capitals") and 12 of the 42 IFBench ones. On 8 more IFBench prompts the
hint also broke the RL model, because squeezing, say, seven uses of one keyword into 60 words forces gibberish. Those
8 tell us nothing. About a third of the instruction-following reach is a verbosity habit that a one-line prompt
change unlocks: elicited, not created.

**Test 3: more tries.** On 10 IFBench reach prompts we sampled the starting model 1,024 times, four times as often
as before. It never produced a genuine success, so its success rate there is below 0.3 % per try (95 % bound).
Yet 4 of those 10 prompts became easy with the brevity hint. Unfindable in 1,024 tries need not mean absent.

**Test 4: an intermediate RL checkpoint.** The model a third of the way through RL (step 750) already solves 26 of
the 43 IFBench reach prompts.

**What is left.** Of the 52 reach prompts, 18 are now explained: 2 by the token budget, 16 by brevity. Two prompts
survived both 1,024 tries and the hint. They are the best candidates for something RL really added. Most of the rest
were tested only with the hint.

**Caveats.** 368 credited answers were read by hand, every deciding one included. Many checker passes were gaming,
such as keyword stuffing or text without sentence breaks that slips past a per-sentence rule. A few labels are
judgement calls and are flagged in the files. One prompt counts only if ALL CAPS counts as title case. Cost: $53.87
of $60.
