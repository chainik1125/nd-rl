# Run brief: compute-match — our 3.2M recipe given the GPU time Robbie's recipe used

Run id `compute-match`. Repository `~/work/compute-match` (branch `dan_compute-match` from the fork's `origin/dan`, made by
`newrun`). Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$30**, ceiling **60
pod-hours**. Authorised by Dan 2026-10-02 as a nice-to-have.

Read first:
- The `best-state` summary (Observation 6: compute) and the `state-cap12` and `noise-floor` summaries in
  `~/nd-rl/experiment-summaries/`.
- `state_train.py`, `state_ladder_ei.py`.
- `pod/sc12/sn_seed.sh` on `origin/dan_state-cap12` (the SN-cap12 recipe).

## Why

`best-state` found that Robbie's 9.56M recipe beats our 3.2M recipe after the same T1 ladder, at both caps. But
each of its runs used about 1.6–1.9× our GPU-seconds on the same GPU type: Stage-1 about 1,270 A40-s against
565–782 s, and the ladder 25.5–30.5k A40-s against 16.0–16.6k s at cap 12. So "better recipe" and "more compute"
are not separated. This run gives our recipe the same compute.

## Design

- **Recipe:** SN-cap12, our 3.2M `lean_staten` model on K12 (`hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/train_k12.jsonl.gz`),
  3 fresh seeds, on A40s only, so GPU-seconds compare directly with `best-state`'s A40 numbers.
- **Stage-1, compute-matched:** train until the run's A40 GPU-seconds equal best-cap12's mean Stage-1 time (take
  it from `best-state`'s compute rows). Save the step-6,000 checkpoint on the way. It is the unmatched control
  (the old recipe) and is read too.
- **T1, compute-matched:** the `state-cap12` ladder protocol with **k raised from 32 to match best-cap12's mean
  ladder GPU-seconds.** Estimate k from a one-round pilot. Keep 8 rounds and the same targets, so only the
  number of attempts per round changes. State the chosen k before running the full ladders.
- **Read-outs, exactly as `best-state`:** textbook72 pass@256, Robbie's dev metric, holdout250, the long pools,
  held-out greedy. Frozen and T1.

## Pre-registration (before the first pod)

- **Quantity:** best-cap12 T1 − compute-matched-ours T1 on textbook72 and the dev metric, with `best-state`'s MDDs.
- **Expectation:** state it.
- **Falsifier:** for "the recipe, not the compute, explains `best-state`": the gap closes to within the MDD.

## Deliverables

- `preregistration/compute-match.md`, `run_compute_match.md` (≤ 300 words + table), `numbers.md` § compute-match,
  `log.md`, and `STATUS.md` line `COMPUTE-MATCH DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/compute-match/{ckpts,artifacts}`.
- Pods deleted; `touch ~/runs/compute-match/executor.done`.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
