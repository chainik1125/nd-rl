# Autoresearch loop for test-time loops

Started 2026-10-07 (agent:claude, at Robbie's request) after the
[looped-transformer spike](../looped_spike_20261006/README.md) found that extra test-time loops
stop helping at about 4. Rules in [program.md](program.md); how to run it in [LOOP.md](LOOP.md).

Forked from [../autoresearch](../autoresearch/README.md), whose harness stays frozen for the
pretraining loop. What differs here:

| | pretraining loop | this loop |
|---|---|---|
| agent edits | `pretrain.py`: `pretrain()` | `pretrain.py`: `pretrain()` and `rl()` |
| RL | harness-owned, 4 rounds, samples at the model's default loops | agent-owned, 1,800 s, through a sampling oracle |
| metric | dev solved (length 7+), 64 samples | the same at `model.n_loops = 8`; also recorded at 2 |
| total budget | 2,400 s | 5,400 s |
| bar | incumbent's 3-seed mean | incumbent's 3-seed mean; target: best spike model (program.md) |

`harness.py` hands `rl()` the target statements and an oracle that returns the model's own
attempts that Lean and nd_verify accept. Lean-gate time is not charged to the RL budget. The other
scripts are the pretraining loop's with paths, pod names and the schema adapted; `pools/` is a
copy. Decisions behind this design (Robbie, 2026-10-07): RL may diverge from the old recipe, the
bar is the best of the spike's looped models and not run 198, three seeds per idea in parallel.

## Findings, first 12 hours (2026-10-07, 09:20 to 21:00 PT)

Twelve looped experiments (500 to 512, Fable 5.1 and Opus 5.5) and one operator control (590),
three seeds each. Every number is in [RESULTS.md](RESULTS.md) and `results/*/results.json`.
The loop stopped at 21:02 PT when the RunPod balance fell below its $5 floor.

Observed:

| | dev solved at 8 loops | at 2 loops | status |
|---|---|---|---|
| 500 baseline: looped 1-8, RL samples at 1/2/4/8 loops | 675 | 657 | keep |
| 502 RL retries only 1-loop failures, at a random 2/4/8 loops | 690 | 672 | keep |
| 507 RL fine-tune in token-budget batches | 732 | 714 | keep |
| 509 Muon for the RL fine-tune | 748 | 727 | keep |
| 510 more, shorter RL rounds | 764 | 752 | keep |
| 512 two alternating shared core pairs (9.56M params) | 766 | 739 | keep |
| 590 control: run 198's untied six blocks, 507's RL, knob ignored | 792 | 792 | control |

Discards: 501 RL only at 8 loops (629 / 493), 503 train each proof at or above its discovery
depth (663 / 630), 504 truncated backprop through loops (662 / 670), 505 countdown embedding
(681 / 662), 506 untied first pass (665 / 654), 508 train on 1-4 loops (683 / 678), 511 wide
looped trunk at the control's parameter count (746 / 727).

- The spike's bar (700) was passed at 507. Every gain since the baseline came from RL efficiency
  (more rounds in the 1,800 s budget, a better fine-tune), not from loops.
- The gap between 8 loops and 2 loops was +18, +18, +18, +21, +12, +27 across the kept models:
  2 to 4% for three times the compute per sample. No experiment widened it without lowering the
  2-loop score; truncated backprop erased it.
- The same RL recipe lifts regular pretraining more: the untied control scores 792 with 507's
  `rl()`, against 680 under the old harness. At equal RL, and at equal parameters (511, 512), the
  looped trunk is 26 to 46 dev theorems behind it and three times as expensive to sample. The
  control was not given 509's or 510's RL changes.
- 505 measured why loops flatten: the core state changes by 28%, 10%, 5% over the first three
  passes and about 2% per pass after that, so passes 4 to 8 sit near a fixed point.
- 502's randomized retries give an unbiased within-run depth effect on targets that failed 8
  tries at 1 loop: 11.4% solved at 2 loops, 13.8% at 4, 15.9% at 8.

Interpretation, not established: on this task and at this size, weight sharing costs more than
re-applying the core buys, and the RL schedule matters more than either. Open: the untied model
with 510's RL, and anything that keeps the looped state from converging by the fourth pass.

