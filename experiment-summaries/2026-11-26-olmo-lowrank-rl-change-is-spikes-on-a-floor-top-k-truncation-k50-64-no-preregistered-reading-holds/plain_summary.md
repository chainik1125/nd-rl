---
written_on: 2026-11-27
written_by: agent:claude (executor, run olmo-lowrank; reviewed 2026-11-27, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-27).** This is the plain-language summary of run `olmo-lowrank`, written by the
> executor. The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-lowrank`, file
> `olmo_lowrank/plain_summary.md` at
> [`2fbecc8c`](https://github.com/dan-pandori/nd-takehome/blob/2fbecc8c/olmo_lowrank/plain_summary.md), the executor's
> claim-check commit, unchanged at the review's final commit `0aa3eff0`. The corrections from its
> [review](https://github.com/dan-pandori/nd-takehome/blob/0aa3eff0/review_olmo-lowrank.md) are applied, and each
> change is marked *[review]*. The numbers are the review's §Verdict "Must be reworded" items:
> - (1) the "near-rank-1 jump" at steps 1,100–1,150;
> - (5) "never solved" (tries and bound), "the other weights do not matter", and the adapter's "five times more than
>   chance" (the control adapter);
> - (6) the gamed shares depend on the readers' standard.
>
> Not applied: two optional wordings in the review's claims table, which it calls acceptable here. Claim 29, "more than
> half the change is spread over thousands of directions", "holds (energy-weighted); optional: 'about half'". Claim
> 30, the top 1 or 4 pieces "do nothing": "'nearly nothing' is more exact; fine as a plain-language summary". The
> executor's text is 498 words, within the brief's 500; the marks take this copy above it. See the
> [summary](README.md) for the full list.

# olmo-lowrank in plain words

**The question.** Reinforcement learning (RL) turned AI2's Olmo 3 32B "DPO" model into a better instruction follower
(Olmo 3.1 32B Think, step 2,300) by changing each of its 448 big weight matrices a little. What shape does that change
have, and which part of it does the work?

**Shape.** A change to a matrix can be written as a sum of rank-1 pieces, each with a size; sorted by size, these form
its spectrum. If a few pieces hold most of the change, it is "low-rank"; if the sizes fall off slowly, it looks like
noise. We measured the share held by each change's top k pieces against random noise of the same size and against the
noise from rounding weights to 16 bits.

- RL's change is neither low-rank nor noise. Its top 16 pieces per matrix hold 22 % of it (noise: 0.75 %). A typical
  matrix has about 230 pieces above same-size noise; more than half the change is spread over thousands of directions.
- The concentration builds up during RL (5 % at step 500, 22 % at 2,300), partly through one jump between
  steps 1,100 and 1,150. *[review]* The jump is not "near-rank-1". It is concentrated in one direction per matrix:
  that direction holds about 15 % of the interval's change, about 20 times its share in the other intervals, and it is
  strongest in the value and output projections of the late layers.
- Supervised fine-tuning moved the weights far more (31× for 32B, measured in olmo-weights) but more diffusely (3 % in
  the top 16); DPO's change is the most concentrated (35 %). The 7B model shows the same order.
- An adapter trained earlier on 16 RL answers points into RL's top directions five times more than chance, but
  only slightly. *[review]* A control adapter, trained the same way on 16 of DPO's own answers, already sits at 2.3
  times chance, so the RL-trained adapter is 2.2 times the control.

**Which part does the work.** We rebuilt the model as DPO plus only the top k pieces of RL's change in every matrix
(k = 1 to 256), added through an adapter rather than rounded into the weights, and gave it 48 hard instruction-following prompts that DPO never
solved (*[review]* no genuine success in the 256 or more earlier tries per prompt that picked them, and no
checker pass in 768 new tries here, 16 per prompt: a rate below 0.39 % at 95 % confidence), and 218 IFBench ones.
G is the share of RL's gain recovered (0 = DPO, 1 = RL). On the hard prompts,
blinded readers checked the first eight checker-passed answers per prompt and model for answers that only fool the
checker.

- The top 1 or 4 pieces do nothing. The top 16 recover a sixth of the gain on the hard prompts (a third on IFBench),
  the top 64 two thirds, and the top 256, only 43 % of the change, nearly all.
- Copying the 448 matrices from the RL model recovers everything; *[review]* the other weights' RL changes are not
  needed for the gain. ("Do not matter" is too strong: on the 218 IFBench prompts this model passes the checker more
  often than RL itself, G 1.07 [1.01, 1.13].)
- The change minus its top 16 pieces still recovers half; neither part suffices.
- So none of the three pre-registered explanations fits (a low-rank core, something spread evenly, something outside
  these matrices). Half the gain needs between 16 and 64 pieces per matrix.
- A third of RL's checker-passing answers on the hard prompts were gamed; the truncated models game less (11–23 %).
  *[review]* These shares depend on the readers' standard. The review's own blinded readers called 37 % of RL's
  gamed, and 12–29 % of the top-16, top-64 and top-256 models'.

**Limits.** One RL run, one benchmark family, the same k for every matrix; a truncation that works is one sufficient
change, not necessarily what RL "meant".
