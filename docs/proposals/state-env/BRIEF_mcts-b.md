# Run brief: mcts-b — MCTS as the RL expert (proposal 22 Phase B), only if mcts-a's gate passed

Run id `mcts-b`. Repository `~/work/mcts-b` (branch `dan_mcts-b` from the fork's `origin/dan`, made by `newrun`; starts after
`mcts-a` is recorded). Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$45**, ceiling **90
pod-hours**. Authorised by Dan 2026-10-02, conditional on the gate. Proposal: nd-rl `docs/proposals/2026-10-02-mcts.md`, Phase B.

## First: check the gate

Read `mcts-a`'s reviewed summary in `~/nd-rl/experiment-summaries/` and the `MCTS-A GATE:` line in its `STATUS.md`.
- **If the gate failed, or the reviewer did not let it stand:** do not create any pod. Write `run_mcts_b.md` (≤ 150
  words: why Phase B did not run, and what `mcts-a` suggests instead), then add the `STATUS.md` line
  `MCTS-B DONE <UTC> (not run: gate)` and `touch ~/runs/mcts-b/executor.done`.
- **If it passed,** continue.

## Design

- **Starting checkpoints:** `trajectory`'s end-of-pretraining checkpoints, `stage1_best12_s{0,1,2}_b1200.pt`. The
  search and value code is `mcts-a`'s, which is on `dan` if it has merged, or on `origin/dan_mcts-a`.
- **Arms,** each with 3 seeds and 8 rounds, using the same targets (`rl_targets.jsonl`), replay and fine-tune steps
  as `trajectory`'s T1 ladders:
  - **B1, found proofs:** train on the shortest found proof per theorem (AlphaProof-style).
  - **B2, visit distributions:** train toward each searched state's visit distribution (AlphaZero-style),
    including searches that did not finish.
  - In both, the value head is retrained each round (on the trunk), on `rl_targets` and generator-pool states only.
- **Compute:** match each ladder's GPU-seconds to `trajectory`'s EI ladders by using fewer, deeper searches per
  round. State the matched budget before the first pod, and record compute.
- **The EI arm is not re-run:** it is `trajectory`'s ladders.

## Read-outs

From r2, r4 and r8 of each ladder, scored two ways:
- **Without search** (plain sampling, k 256, both sampling seeds): textbook72, holdout250, `trajectory`'s groups A,
  B and C per seed, and the long pools. Compared with `trajectory`'s EI, this asks whether search taught the policy
  something.
- **With search** (PUCT with the value, at `mcts-a`'s matched budget): the same pools. This is the best-system
  number, against Robbie's combined model and `best-state`.

## Pre-registration (before the first pod)

- **The headline:** B1 and B2 against EI on group C, solved without search at r8, and the MDD from `trajectory`'s
  seed spread.
- **Expected:** an expectation for each arm.
- **Falsifier:** for "a search expert teaches the policy more than sampling".

## Deliverables

- `preregistration/mcts-b.md`, `run_mcts_b.md` (≤ 400 words + figures), `numbers.md` § mcts-b, `log.md`, and
  `STATUS.md` line `MCTS-B DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/mcts-b/{ckpts,artifacts}`.
- Pods deleted; `touch ~/runs/mcts-b/executor.done`.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
