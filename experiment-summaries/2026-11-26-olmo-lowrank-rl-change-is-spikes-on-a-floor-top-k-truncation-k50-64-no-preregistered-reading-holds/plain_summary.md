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
- The concentration builds up during RL (5 % at step 500, 22 % at 2,300), partly through one near-rank-1 jump between
  steps 1,100 and 1,150.
- Supervised fine-tuning moved the weights far more (31× for 32B, measured in olmo-weights) but more diffusely (3 % in
  the top 16); DPO's change is the most concentrated (35 %). The 7B model shows the same order.
- An adapter trained earlier on 16 RL answers points into RL's top directions five times more than chance, but
  only slightly.

**Which part does the work.** We rebuilt the model as DPO plus only the top k pieces of RL's change in every matrix
(k = 1 to 256), added through an adapter rather than rounded into the weights, and gave it 48 hard instruction-following prompts that DPO never
solved, and 218 IFBench ones. G is the share of RL's gain recovered (0 = DPO, 1 = RL). On the hard prompts,
blinded readers checked the first eight checker-passed answers per prompt and model for answers that only fool the
checker.

- The top 1 or 4 pieces do nothing. The top 16 recover a sixth of the gain on the hard prompts (a third on IFBench),
  the top 64 two thirds, and the top 256, only 43 % of the change, nearly all.
- Copying the 448 matrices from the RL model recovers everything; the other weights do not matter.
- The change minus its top 16 pieces still recovers half; neither part suffices.
- So none of the three pre-registered explanations fits (a low-rank core, something spread evenly, something outside
  these matrices). Half the gain needs between 16 and 64 pieces per matrix.
- A third of RL's checker-passing answers on the hard prompts were gamed; the truncated models game less (11–23 %).

**Limits.** One RL run, one benchmark family, the same k for every matrix; a truncation that works is one sufficient
change, not necessarily what RL "meant".
