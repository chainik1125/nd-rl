---
written_on: 2026-11-25
written_by: agent:claude (executor, run olmo-weights; reviewed 2026-11-25, corrections marked by the librarian)
---

> **Reading copy (librarian, 2026-11-25).** This is the plain-language summary of run `olmo-weights`, written by the
> executor. The source is fork `dan-pandori/nd-takehome`, branch `dan_olmo-weights`, file
> `olmo_weights/plain_summary.md` at
> [`c13a307a`](https://github.com/dan-pandori/nd-takehome/blob/c13a307a/olmo_weights/plain_summary.md) (unchanged at
> the review's final commit `e8d33db7`). The corrections from its
> [review](https://github.com/dan-pandori/nd-takehome/blob/e8d33db7/review_olmo-weights.md) are applied, and each
> change is marked *[review]*. The numbers are the review's §Verdict "Must be reworded" items; the last mark comes
> from its §Comparison section "New finding":
> - (1) RL's large change per step relative to its learning rate: it holds only at the released 25–50-step spacing,
>   and as a property of RL it is "Not supported" (two places);
> - (6) the claim inherited from run `olmo-ifladder` is labelled (with one *[librarian]* gloss, the definition of its
>   "reach" prompts);
> - (new finding) the 32B RL checkpoints at steps 150 and 200 lie on an abandoned branch.
>
> The executor's text is 494 words, within the brief's 500; the marks take this copy above it. See the
> [summary](README.md) for the full list.

# olmo-weights in plain language

**The question.** A language model is a long list of numbers, its weights: 32 billion for Olmo 3 32B, 7 billion for
Olmo 3 7B. Training nudges these numbers. AI2 released snapshots ("checkpoints") from every stage: pretraining on web
text, mid-training on curated data, long-context training, SFT (imitating worked examples), DPO (preferring better
answers) and RL (being rewarded when an answer checks out). How much do the weights move at each stage?

**How we measured.**
- **Relative change.** Subtract two consecutive checkpoints and divide the size of the difference by the size of the
  weights: 1 % means the difference is a hundredth of the weights' size.
- **Per step.** Checkpoints were saved every 1,000 training steps in pretraining but every 25–50 in RL, so we also
  divide by the number of steps. We processed 453 checkpoints (15 terabytes).

**What we found.**
- **The stage switches are obvious, and mostly set by the learning rate** (the size of step training takes). From
  pretraining through SFT, the relative change per step is about 0.6–0.9 times the learning rate.
- **RL is different.** Its learning rate is tiny and constant, yet its changes per step come out about ten times
  larger than the learning rate predicts. *[review]* That holds only at the released spacing of 25–50 steps between
  RL checkpoints. Measured over about 1,000 steps (from the start of RL), RL moves 0.7–1.1 times its learning rate per
  step, like every other stage, and about 11–25 times less per step than SFT. Two things inflate the short-interval
  figure: consecutive RL changes partly undo each other, so the distance grows more slowly than the number of steps,
  and, plausibly, rounding (next). The review finds no support for RL's large change per step, relative to its
  learning rate, as a property of RL. A likely part of the reason is rounding. Weights are stored in a 16-bit
  format with only two to three significant digits, so a tiny update either does nothing or jumps a whole rounding
  step. Between RL checkpoints, 80–92 % of the weights do not change at all, and most of the rest move by one rounding
  step.
- **RL's changes mostly cancel out**: each interval tends to undo part of the previous one. So although an RL step is
  nearly as large as an SFT step, RL's total movement is small. *[review]* "Nearly as large" compares RL's 25–50-step
  intervals with SFT's 1,000-step ones. At the same spacing an RL step is about 11–25 times smaller, roughly the ratio
  of their learning rates. After 2,300 RL steps the 32B model is 0.36 %
  away from where RL began, against 11 % for SFT. Two thirds of the 32B model's weights (88 % for 7B) end within one
  rounding step of their starting values.
- **Where the change happens.** In every stage, most of it is in the big matrices of the attention and feed-forward
  layers; small "normalisation" vectors barely move.
- **The LoRA comparison.** An earlier run trained a small add-on (a "LoRA") on 16 answers written by the RL model; it
  taught the starting model most of RL's new tricks. *[review]* That is run `olmo-ifladder`'s reviewed result: the
  add-on `lora_main`, on RL's starting model `allenai/Olmo-3-32B-Think-DPO`, elicits 40–41 of 48 instruction-following
  "reach" prompts (*[librarian]* prompts that RL's final model solves but on which that model had no genuine success
  in 256 tries, per olmo-ifladder's summary), against 0 for the control add-on. Its change points slightly but
  unmistakably along RL's change: about a thousand times more than random directions, and 28 times more than a control
  add-on. The match grows as RL goes on.
- **Bookkeeping surprises.** The released 32B SFT model is an equal mix of two training runs, stored under a
  misleading step name; DPO started from that mix. A few RL-Zero checkpoint names hold the wrong weights; we left
  them out. *[review]* One more, found by the review: the 32B RL checkpoints at steps 150 and 200 lie on a side
  branch that the run abandoned. The released line restarts from step 100, and steps 250 to 2,300 descend from it,
  so their step labels probably overstate the training steps by about 100.

**What it does not show.** Moving weights is not changing behaviour: a small change can matter a lot. Proposal 48 measures behaviour directly.
