---
written_on: 2026-10-06
written_by: agent:claude
status: complete
---

# Looped-transformer spike: close to run 198, not past it; test-time loops help only up to ~4

Question (Robbie, 10/2 group todo): does a weight-shared, looped trunk let us buy proofs on hard problems by
running more loops at test time? This is a one-night spike on run 198's `pretrain.py`, one seed per arm, through
the frozen autoresearch harness. It is not part of the autoresearch loop.

Report with charts: <https://claude.ai/artifact/BPrfvMZP45s6dYLCzp8vfn> (private to Robbie). Every number:
[results_table.md](results_table.md).

## Setup

- `pretrain_looped.py` = run 198's file (sha `6bdbe0047ac4`) with the trunk changed to one untied entry block,
  two shared core blocks applied n times, one untied exit block (6.40M params against 9.56M). Each core pass
  starts from `LayerNorm(state) + entry output`. Training draws n per batch from `ND_TRAIN_LOOPS`, in pretraining
  and in the harness's RL fine-tune; sampling uses `model.n_loops`, 2 by default (6 blocks per token, as 198).
- Arms: fixed 2 loops, random 1-4, random 1-8, log-spaced 1-16 (`1,2,3,4,6,8,12,16`). Reference: run 198 seed 0
  from Hugging Face.
- `loop_sweep.py`: eval only, one checkpoint at several loop counts, on textbook72 (k=256), long dev (the 329 dev
  theorems of true length 10+, k=64) and each run's own never-solved RL targets (k=128).
- `more_rl.py`: 8 more expert-iteration rounds from `final.pt` with the harness's own RL stage (12 rounds total).
  Outside the frozen protocol; its dev scores are not comparable with autoresearch scores.

## Results

Observed, one seed per arm:

| | harness score | long dev, per-sample success by test-time loops |
|---|---|---|
| run 198 | 680 (seeds 670, 680, 694) | 19.0% |
| looped, fixed 2 | 666 | 2.1, 16.8, 14.4, 9.5, 4.4 at 1, 2, 3, 4, 6 |
| looped, 1-4 | 644 | 13.4, 15.5, 16.3, 16.8, 16.6, 16.5, 16.5 at 1, 2, 3, 4, 6, 8, 12 |
| looped, 1-8 | 602 | 9.5, 13.9, 16.6, 16.9, 16.8, 16.9, 16.7 at 1, 2, 4, 6, 8, 12, 16 |
| looped, 1-16 | 593 | 9.0, 12.3, 13.8, 13.7, 13.8, 13.8 at 1, 2, 4, 8, 16, 24 |

After 12 RL rounds: run 198 dev 716, long dev 121 solved (24.8% per sample). Looped 1-4: dev 690, long dev 118,
116, 115, 114, 115, 115, 115 solved at 1, 2, 3, 4, 6, 8, 12 loops (21.0, 22.9, 23.1, 23.3, 23.1, 23.0, 23.1%).

- Random-loop models climb to about 4 loops and are then flat out to 24, whatever range they trained on. The
  ceiling (16.8%) equals the fixed-2 model at 2 loops and is below run 198.
- A wider training range costs harness score at 2 loops and does not raise or extend the ceiling.
- After 12 RL rounds the loop benefit is nearly gone; the looped model at 1 loop (4 blocks per token) is within
  3 long-dev problems of run 198 at 6 blocks.
- Loops do not crack the RL targets that 12 rounds could not: 8 to 12 of 280 at any loop count.
- Before RL, loop count has no effect on any pool.

Interpretation, not established: extra loops compensate for under-training, and RL removes the need for them.
Open: sampling at more than 2 loops during RL, loss at every loop, and more seeds.

## Limits

One seed per arm. Run 198 was tuned over about 200 autoresearch runs on the same dev pool and the looped trunk
not at all. RL round-1 counts vary from about 260 to 760 between near-identical runs and say nothing; an early
claim built on them was withdrawn. textbook72 (16 to 21 solved everywhere) is too small to separate the models.

## Reproduce

`launch.py up NAME`, `launch.py run NAME ARM` (ARM = `baseline`, `2`, `1,2,3,4`, ...), optionally
`launch.py more NAME ARM` and `launch.py sweep NAME ARM`, then `launch.py down NAME`. `analyze.py` writes the
table and `report.py` the page. `check_cache.py` tests cached decoding against the full forward. Five A40 pods,
about 13 pod-hours, roughly $6.

Checkpoints: `robbiethompson2018/nd-rl-checkpoints`, `runs/looped-spike-20261006_<arm>_s0` (4 rounds) and
`..._loops1234-rl12_s0`, `..._run198-rl12_s0` (12 rounds; there `rl_round_01`-`08` are rounds 5-12).
