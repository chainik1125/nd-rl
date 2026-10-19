# Run brief: ckpt-avg — does checkpoint averaging remove the seed "modes"? (proposal 14, item 8)

Run id `ckpt-avg`. Repository `~/work/ckpt-avg` (branch `dan_ckpt-avg` from `origin/dan_stage1-dynamics`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` (nd-rl canonical) governs. Budget **$2**, ceiling
**4 pod-hours**. Authorised by Dan 2026-09-28 as nearly free: **no training** — evaluations of saved
checkpoints only.

Read first: the `stage1-dynamics` summary in `~/nd-rl/experiment-summaries/` and its review;
`NOISE_FLOOR.md`; `train.py`'s per-length validation code.

## Question

`stage1-dynamics` found the depth-3 "mode" is not a property of a seed but an oscillation within each
run: all 12 of its 24,000-step runs crossed the high/low cut 1–10 times, so the project's largest noise
source is where a run happens to stop. If averaging a run's late checkpoints (stochastic weight averaging
along one trajectory) or choosing its checkpoint by per-length validation loss removes that
oscillation, the seed-to-seed spread should collapse and every later comparison needs fewer seeds. Does
it, and at what cost to the mean?

## Material

`hf://buckets/dan-pandori/nd-rl/stage1-dynamics/ckpts/`: the WSD runs (arm W, 8 seeds, constant-lr
stable phase to 24,000 steps with decay branches at 6k/12k/24k) and arm F (4 seeds), with trajectory
checkpoints every 2,000 steps (check what is actually there); arm C (cosine, 6k) has final checkpoints
only, so it is a reference, not a subject. Model for every number: 3,214,336 parameters, from scratch,
`lean_seq`, cap 6 — name the checkpoint files.

## Design (a suggestion)

**Split the held-out set first.** `stage1-dynamics` measured its per-length validation loss on
`data/p2/heldout.jsonl`, the same file as held-out accuracy, so selecting a checkpoint by that loss and
then scoring it on the same file is circular. Split the 5,000 theorems in two, stratified by length and
by the depth-3 slice, with a fixed seed: half A for anything that selects, half B for every reported
accuracy. Commit the split before any evaluation.

Per run, compare on half B (greedy, per length and the depth-3 slice):
1. the endpoint checkpoint(s) as `stage1-dynamics` reported them;
2. **uniform weight averages** of the last K stable-phase checkpoints before a decay point, for a few K
   (e.g. 2, 4, 8), and of the decayed endpoint with its neighbours if that is well-defined;
3. **the checkpoint with the lowest per-length validation loss on half A** (6-line and depth-3 slices).

Report per variant: mean and across-seed sd on each length bin and the depth-3 slice, the fraction of
seeds in the high mode with a Wilson interval, and — for the averages — whether the averaged model is
above, between or below its constituents. Weight averaging along one constant-lr trajectory is standard;
averaging across the decay is less so — if an average is broken (accuracy collapses), say so rather than
tune it.

## Pre-registration (`preregistration/ckpt-avg.md`, before any evaluation)

Your expectations, and the adoption rule: averaging (or loss-based selection) becomes the project's
Stage-1 standard if it cuts the across-seed sd on the depth-3 slice by ≥ 2× (from ≈ 0.3) and shrinks the
6-line sd, without lowering the mean by more than 2 pp on any bin. State which variant you expect to win.

## Deliverables

`preregistration/ckpt-avg.md`; the split and the averaging/evaluation scripts; `run_ckpt_avg.md`
(≤ 300 words + one figure: depth-3 slice per seed, endpoint vs best variant); `numbers.md` § ckpt-avg;
`log.md`; `artifacts/ca/summary.json`; `STATUS.md` line `CKPT-AVG DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/ckpt-avg/artifacts`; pods deleted; `touch ~/runs/ckpt-avg/executor.done`.
If the rule is met, add a one-paragraph recommendation for the Stage-1 standard to `run_ckpt_avg.md`.
