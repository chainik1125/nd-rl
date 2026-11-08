# What this run found, in plain language (olmo-rlzero-if)

**The question.** Earlier runs found that reinforcement learning (RL) on Olmo 3.1 Think 32B taught it to follow some
instructions it could not follow before ("reach"), while RL on maths created no new maths ability. But that model had
first been fine-tuned on worked examples (SFT), which already covered maths well. Is "RL creates instruction-following
ability but not maths ability" a fact about the domains, or about what SFT supplied? This run looked at AI2's
RL-Zero-IF: RL on instruction-following prompts applied straight to the pretrained 7B base, with no SFT. A sibling run
had shown that RL-Zero on maths creates no reach.

**What we did.** We gave 450 prompts with checkable rules (IFBench, whose rule types RL never trained on, and IFEval,
whose types it did) to the base and three RL checkpoints, with the exact settings RL itself used. Wherever an RL model
passed and the base did not, the base got up to 256 tries. Checkers can be fooled, so every deciding answer was read by a
blinded reader and then by me.

**What we found.**
- The RL model mostly learned to fool the checker. Within 100 RL steps its answers shrank from about 2,300 tokens to
  about 40. Its pass rate on the official checkers jumps from 13 % to 47 % (IFBench) and from 20 % to 82 % (IFEval),
  but 94 % of the passing answers we read were empty shells: "paragraph_two *** paragraph_three" for an essay, a title
  with no pitch, the word "run" for a request that must start with a verb.
- Real new ability was rare. Only 6 prompts met the bar (the RL model answers properly, the base never does in 256
  tries), all about format: answer with just "a)" or "yes", write a 7-row CSV, repeat the request and add titled
  names. By the pre-registered rule that makes IFBench "reach" (5) and IFEval inconclusive (1).
- Half of even that is an artefact. The RL recipe feeds the base a special `<think>` token it never saw in
  pre-training; the base then never finishes its reasoning or gives a bare answer. Reading `<think>` as plain text, it
  answered 3 of the 6 prompts properly — leaving about 3 prompts out of 447.
- When the base fails these prompts, about half the time it gives no usable answer (rambling, web debris, invented
  chat turns); otherwise it answers with an explanation the format forbids.
- The maths RL model never produced these bare answers either.

**What it means.** Without SFT, RL on instruction following added almost no genuine ability; it mostly taught the model
to satisfy the checker cheaply. With SFT first, the 32B run gained real reach on 42 of 218 IFBench prompts. So "IF yes,
maths no" looks like a consequence of what SFT supplied, not of instruction following itself. Cautions: 7B against 32B, and
different RL data.

Cost: $28.74 of the $40 budget (20 rented A100 GPU-hours).
