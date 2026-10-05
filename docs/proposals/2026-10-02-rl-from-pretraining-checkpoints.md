---
written_on: 2026-10-02
written_by: agent:claude
status: approved by Dan 2026-10-02 with the step-1,600 arm; run `rl-from-ckpt`, then `grpo-best`
---

# Proposal 19: RL from earlier pretraining checkpoints — does RL only finish what pretraining started?

## Why

`trajectory` (cap 12, Robbie's recipe, proof-state interface; reviewed) found the following. Theorems that only
RL solves were already within about one improbable step at the end of pretraining: worst step about −6 nats,
pass@256 about 0.15. RL then concentrated on the model's own proof. The theorems RL never solved had several
hard steps.

That is consistent with elicitation, but it is one starting point. Starting RL from **less-pretrained**
checkpoints turns it into a test.
- **Elicitation:** RL solves a theorem only if its hardest step is already likely enough when RL starts. One
  threshold curve then fits every starting point.
- **Creation:** RL from an early checkpoint solves theorems whose hardest step is far below that threshold,
  learning moves pretraining had not yet made likely.

`trajectory` already measured every theorem's per-step log-likelihood under every one of these checkpoints. So
the starting position of each theorem is known before RL starts, which is what makes this cheap and sharp.

## Design

- **Starting checkpoints:** `trajectory`'s three cap-12 seeds (9.56M, `lean_staten`, K12). Dan's quarters map to
  saved steps as follows:

  | start | step | share of pretraining | why |
  |---|---|---|---|
  | none | 0 (initialisation) | 0 % | RL with no pretraining |
  | early (suggested addition) | 1,600 | ≈ 7 % | see below |
  | ¼ | 5,000 | ≈ 21 % | nearest saved step to 6k |
  | ½ | 12,000 | ≈ 50 % | |
  | ¾ | 16,000 | ≈ 66 % | 20k (≈ 83 %) sits next to the end on every curve |
  | end | ≈ 24,100 | 100 % | already run: `trajectory`'s own ladders, reused |

- **Why add 1,600.** From initialisation the model solves nothing (pass@256 = 0 on every group), and expert
  iteration has no signal without a single success. That run will most likely stay at zero, which is worth
  showing but says little about the threshold. Most of the movement in `trajectory`'s curves is between steps
  400 and 12k (log scale): at 1.6k, pass@256 on the theorems pretraining eventually solves is about 0.25. So 1,600
  is the earliest start where RL has something to work with. It adds 3 ladders. Drop it if you prefer exactly
  quarters.
- **RL:** the full T1 ladder, identical to `trajectory` (8 rounds, k 32, T 0.8, the same targets and replay), from
  each start × 3 seeds: 15 new ladders.
  - **Stop rule for the step-0 arm:** if rounds 1–2 accept no proof on any target, stop and report. Expert
    iteration cannot move without a success.
- **Read-outs:** the same 322 theorems (textbook72 + holdout250). Score r0, r2, r4 and r8 of every ladder: pass@k
  at k up to 256, both sampling seeds, and teacher-forced log-likelihood of the eventual and reference proofs.

## Analysis

- **The threshold test.** For each (theorem, seed, start), take x = the worst-step log p of the theorem's
  reference proof under the starting checkpoint (from `trajectory`), and y = whether RL from that start solves it
  at r8. Fit P(solve | x) per start. Elicitation predicts one curve for all starts, with a threshold near −6 to −8
  nats. Creation predicts that early starts solve theorems from far below it. Report the count solved from below
  −12 nats at the start.
- **RL as a substitute for pretraining:** final performance by start, against the pretraining compute saved and
  the RL compute spent (compute rows).
- **Trajectories:** the `trajectory` figures for each start.

## Cost and time

- **15 ladders** at about $4 each (A40 / A6000), fewer if the step-0 arm stops early: ≈ $55–60.
- **Read-outs:** ≈ $15–20.
- **Total ≈ $70–80** (budget $90). Running in parallel, about 10–12 h end to end.
- **Without the 1,600 arm,** about $15 less.
