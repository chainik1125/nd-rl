---
written_on: 2026-11-25
written_by: agent:claude (orchestrator)
---

# Olmo elicitation curves: readings fixed in advance

Written 2026-11-25 2:20 am PT, before any pod exists and before any EDL number. Design and
terms: `SPEC.md`. Amendments are appended below with a time stamp, each before the data it changes.

## What the measurements can and cannot say

The EDL targets are the final model's own successful responses. So every stage-1 number measures how
far a checkpoint is from the final model's way of answering these prompts, in nats, and how much of
that distance 64 examples remove. It is not a direct measure of capability: a checkpoint could solve
the prompts in another way and still score far. The sampling arms (stage 2) are the behavioural
side. All of it is one RL run of one model.

## Stage 0c: replication gate

Fresh draws at `dpo` and `s2300`, 218 arm-A prompts, 2 draws each, strict checker.
- **Pass:** at both checkpoints the 95 % prompt-bootstrap interval of (our pass@1 − Dan's stored
  pass@1) includes 0, or the difference is below 5 points.
- **Fail:** stage 2 does not start on that pod until the cause is found. Stage 1 does not depend on
  the sampler and continues.

## Stage 1: EDL

Units: nats per response token unless said otherwise (length-normalised form), total and by segment
(thinking, answer). Intervals: 95 %, bootstrap over test prompts.

Checks of the method (if one fails, the stage-1 curves are reported as unvalidated):
- **C1, null control.** `EDL*` at `s2300` is below 10 % of `EDL*` at `dpo`.
- **C2, the learner learns.** At `dpo`, at the best learning rate, `L_test(64) < L_test(0)` with the
  interval of the difference excluding 0.
- **C3, order noise.** At the three checkpoints with two orders, the two orders' `EDL(64)` differ by
  less than a third of the `dpo`-to-`s2300` difference in `EDL*`.

Predictions:
- **P1, likelihood.** `L_0(t)` falls along the run: Spearman correlation with RL step at most −0.8
  over the nine checkpoints.
- **P2, EDL.** `EDL*(t)` falls along the run: Spearman at most −0.8.
- **P3, most of the distance is elicitable early.** At `dpo`, 64 examples close at least half of the
  gap in `L_test` to `s2300`: `(L_0 − L_64) / (L_0 − L_0(s2300)) >= 0.5`, interval above 0.25. This
  is the EDL-side counterpart of Dan's LoRA result (16 examples reach 40–41 of 48 prompts).
- **P4, timing.** The RL step at which `EDL*` has lost half of its `dpo` value is not later than the
  step at which arm-A pass@1 has made half of its gain (Dan's curve). Linear interpolation between
  checkpoints; "not established" if the two steps fall between the same pair of checkpoints.

Descriptive, no prediction:
- The shape of EDL per token over n' at each checkpoint (falling throughout, or rising first: the
  paper's elicit and teach signatures). With one pass and five evaluation points this is coarse.
- Thinking against answer: which segment carries the distance, and which closes faster.
- `L_reach` against `L_test`: whether learning from ordinary prompts transfers to the reach prompts.

Learning-rate grid: {3e-5, 1e-4, 3e-4}. A pilot at `dpo` may shift the grid by one step if the best
rate is at an edge; the grid is then frozen and the pilot rows at dropped rates are kept on file.

## Stage 2: sampling arms (checker passes only)

- **T, transplant.** Predicted: without the instruction, the strict pass rate rises along the run
  from Dan's 14 % at `dpo` toward the final model's own rate (Spearman at least 0.8). Descriptive: at
  which step half of that gain is made, with and without the instruction.
- **H, hint.** Predicted: the hint's gain on the reach prompts (hinted minus plain strict rate)
  shrinks along the run and is within 5 points of 0 at `s2300`.
- No genuine-versus-gamed reading in the first pass. Any arm shown says so.

## Stopping rules

- Money: stop launching new stages when the spend recorded in the pod ledger passes $95; finish and
  fetch what is running; all pods deleted by $110.
- Time: 2026-11-25 2:00 pm PT. What is unfinished then is reported as unfinished.
- Priority if either binds: stage 1 at all nine checkpoints, then T, then H, then L.

## Amendment 1 (2026-11-25 2:36 am PT, before any pod exists)

Leon, 2:35 am PT: "I want you to do a mini run first I suspect 5 checkpoints is sufficent for what we
need so you dont need to run all 10".

- **Checkpoints of the mini run:** `dpo` (0), `s0100`, `s0500`, `s1100`, `s2300`. They span Dan's
  curve (reach prompts with a strict pass: 0 / 18.8 / 79.2 / 97.9 / 97.9 % of 48) and leave out
  step 200, which is off the released line.
- **Stage 1** runs at all five; two data orders at `dpo`, `s0500`, `s2300`, one at the other two.
- **Stage 2** (T and H) runs at `s0100`, `s0500`, `s1100`; Dan's stored cells cover `dpo` and
  `s2300` and are reused. T keeps his design: 115 traces over the 41 prompts, 8 answers per trace.
- **Stage 0c** unchanged (`dpo`, `s2300`).
- P1, P2 and the T prediction keep their Spearman thresholds, now over five points (a single
  adjacent swap gives ±0.9, so the threshold of 0.8 allows one). P4 interpolates between the five.
- The other four checkpoints are a possible second pass, decided after the mini run.

## Amendment 2 (2026-11-25 2:40 am PT, before any pod exists)

From the pool builder's length table (stored `s2300` strict passes; thinking is 96–98 % of the
tokens, median eligible draw 5,190 tokens):

- **Response-token cap: 6,144.** 125 train-or-test prompts survive (49 with a hand-labelled genuine
  pick) and 42 reach prompts (38 genuine). At 4,096 only 85 survive, short of the 96 needed, and the
  bias toward short responses is larger. n = 64 and m = 32 are unchanged; the reach set keeps the
  genuine picks only.
- **Learning rate inside `EDL*`.** The leading convention is ndbench's: the learning rate with the
  shortest online code length (the sum of online nats), then `EDL* = EDL at that rate +
  ln(number of rates)`. The minimum of EDL over rates, as first written in `SPEC.md`, is reported
  beside it as secondary; it went spuriously negative on ndbench's null control.
