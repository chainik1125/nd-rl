# Run brief: state-readouts — naming vs state on the survivors, and the textbook schemata on SN-cap12

Run id `state-readouts`. Repository `~/work/state-readouts` (branch `dan_state-readouts` from the fork's `origin/dan`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs; record per-arm compute as it asks. Budget **$12**,
ceiling **24 pod-hours**. Authorised by Dan 2026-09-30. No training: sampling read-outs of existing checkpoints only.

Read first:
- The `support-curves`, `support-followups`, `support-state`, `state-env`, `state-cap12` and `ds-composition`
  summaries in `~/nd-rl/experiment-summaries/`.
- `STATE_ENV.md`, `state_sample.py`, and `state_eval.py`.

**Code not on `dan`:** `lpool_reread.py` is only on `origin/dan_state-cap12` (and `origin/dan_long-pool`). Take it with
`git checkout origin/dan_state-cap12 -- lpool_reread.py` if you use it.

Two independent parts. Report each separately.

## A. Is the survivor gain from seeing the state, or from environment-assigned names?

`support-state` found that the SN base reaches 28 / 29 of `support-curves`' survivors per seed at 200,000 attempts
per temperature. Those are theorems the whole-proof base never reached in 400,000 attempts. But SN differs from the
whole-proof model in two ways: it **sees the proof state**, and the **environment assigns hypothesis names**. The
reviewer's point was that the run cannot separate the two. The `state-env` arms can:
- **S** (`lean_state`) sees the state and chooses its own names.
- **SH** (`lean_stateh`) also sees the actions so far.

Checkpoints are in `hf://buckets/dan-pandori/nd-rl/state-env/ckpts/se/`, plus any further S Stage-1 seeds that
`state-frontier` trained.

- **Design:** the S and SH **Stage-1 bases**, at least 2 seeds each, on the 29 survivors. Use `support-state`'s
  protocol exactly: 200,000 attempts per theorem at each of its temperatures, the same stop rule, and the same
  sampler settings.
- **Report:** survivors reached per seed and arm, and p̂ per theorem. Compare against SN's inherited 28 / 29 and the
  whole-proof base's 0 / 29.
- **Pre-register a reading:**
  - S and SH near SN (≥ 20 / 29): seeing the state is what matters.
  - S and SH near 0: the environment's naming is what matters.
  - In between: both.

## B. Does SN-cap12 get past the textbook schemata that no arm has moved?

Several textbook schemata (most De Morgan and distribution forms, excluded middle, Peirce) have stayed at ≤ 2
solves in every arm. See `ds-composition`'s 19-schema table and proposal 14 item 4.

- **Checkpoints:** SN-cap12 T1 and frozen, s0–s3 (`hf://buckets/dan-pandori/nd-rl/state-cap12/ckpts/`).
- **Pools:** the textbook theorems of `data/ladder/transfer.jsonl` (the ones behind the schema table) and the
  textbook theorems of `data/ladder/transfer_long.jsonl`.
- **Settings:** k 256, T 0.8, `max_steps` 96.
- **Report:** solves per schema, per seed and arm, next to the schema table's existing arms.
- **Pre-register:** how many of the dead schemata you expect SN-cap12 T1 to move to ≥ 5 solves.

## Deliverables

- `preregistration/state-readouts.md`, `run_state_readouts.md` (≤ 350 words), `numbers.md` § state-readouts, `log.md`,
  and `STATUS.md` line `STATE-READOUTS DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/state-readouts/artifacts`.
- Pods deleted; `touch ~/runs/state-readouts/executor.done`.

Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
