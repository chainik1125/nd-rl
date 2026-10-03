# Run brief: rl-continue — is the cap-12 RL saturated? Eight more rounds on the same targets

Run id `rl-continue`. Repository `~/work/rl-continue` (branch `dan_rl-continue` from the fork's `origin/dan`, made by `newrun`).
Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$15**, ceiling **30 pod-hours**. Authorised
by Dan 2026-10-03: "Let's first do the cheap direct check."

## Why

`trajectory`'s cap-12 ladders (Robbie's recipe, `lean_staten`, K12) add few new targets in their last rounds:

| | r6 | r7 | r8 |
|---|---|---|---|
| s0 | +18 | +14 | +5 |
| s1 | +20 | +9 | +9 |
| s2 | +27 | +13 | +12 |

That leaves 97.2–98.0 % of the 4,495 `rl_targets` solved, and 95.0–95.8 % of the 2,285-theorem transfer pool. The
orchestrator extrapolated that 8 more rounds add about 20–40 targets and 10–20 transfer theorems, with group C flat.
This run checks that directly.

## Design

- **Continue `trajectory`'s three T1 ladders for rounds 9–16,** from their r8 checkpoints
  (`hf://buckets/dan-pandori/nd-rl/trajectory/ckpts/tj/ladder/la_T1_best12_s{0,1,2}_r8.pt`), with the **same
  accumulated found set**. Use `state_ladder_ei.py --start_round 9 --resume`, which loads
  `artifacts/<name>/found_8.jsonl` and its checkpoint. Place `trajectory`'s r8 found files and checkpoints at those
  paths first.
- Keep everything else identical to `trajectory`: k 32, T 0.8, the same targets, replay, fine-tune steps and batch.
- **Show the continuation is seamless:** the r9 checkpoint's starting state equals r8's, and the found set is
  carried over.
- **Code not on `dan`:** the `pod/tj/` ladder scripts are on `origin/dan_trajectory`; take them with
  `git checkout`.

## Read-outs

- **Per round** (from the round JSONs): new targets, `targets_cum`, `transfer_cum`, and sample accuracy. Plot rounds
  1–16 for all three seeds next to the extrapolation.
- **At r12 and r16:** textbook72, holdout250, `trajectory`'s group C (per seed), `rr600` 13–16 and `transfer_long2`,
  at k 256 with both sampling seeds. Use `trajectory`'s read-out code so the numbers compare directly with its r8.

## Pre-registration (before the first pod)

The extrapolation above as the expectation: per seed, 20–40 new targets over rounds 9–16, 10–20 transfer, and group C
unchanged within the sampling spread.

Falsifier for "saturated", either of:
- ≥ 60 new targets on ≥ 2 of 3 seeds;
- group C pass@256 rises beyond the seed spread.

## Deliverables

- `preregistration/rl-continue.md`, `run_rl_continue.md` (≤ 250 words + the rounds 1–16 figure), `numbers.md`
  § rl-continue, `log.md`, and `STATUS.md` line `RL-CONTINUE DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/rl-continue/{ckpts,artifacts}`.
- Pods deleted; `touch ~/runs/rl-continue/executor.done`.

The r16 checkpoints are the **control arm** for proposal 23 (RL on Charles's textbook release), so keep them.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
