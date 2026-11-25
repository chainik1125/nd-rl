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
  larger than the learning rate predicts. A likely part of the reason is rounding. Weights are stored in a 16-bit
  format with only two to three significant digits, so a tiny update either does nothing or jumps a whole rounding
  step. Between RL checkpoints, 80–92 % of the weights do not change at all, and most of the rest move by one rounding
  step.
- **RL's changes mostly cancel out**: each interval tends to undo part of the previous one. So although an RL step is
  nearly as large as an SFT step, RL's total movement is small. After 2,300 RL steps the 32B model is 0.36 %
  away from where RL began, against 11 % for SFT. Two thirds of the 32B model's weights (88 % for 7B) end within one
  rounding step of their starting values.
- **Where the change happens.** In every stage, most of it is in the big matrices of the attention and feed-forward
  layers; small "normalisation" vectors barely move.
- **The LoRA comparison.** An earlier run trained a small add-on (a "LoRA") on 16 answers written by the RL model; it
  taught the starting model most of RL's new tricks. Its change points slightly but unmistakably along RL's change:
  about a thousand times more than random directions, and 28 times more than a control add-on. The match grows as RL
  goes on.
- **Bookkeeping surprises.** The released 32B SFT model is an equal mix of two training runs, stored under a
  misleading step name; DPO started from that mix. A few RL-Zero checkpoint names hold the wrong weights; we left
  them out.

**What it does not show.** Moving weights is not changing behaviour: a small change can matter a lot. Proposal 48 measures behaviour directly.
