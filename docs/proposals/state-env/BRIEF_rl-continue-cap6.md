# Run brief: rl-continue-cap6 — the cap-6 ladders, eight more rounds on the same targets

Run id `rl-continue-cap6`. Repository `~/work/rl-continue-cap6` (branch `dan_rl-continue-cap6` from the fork's `origin/dan`,
made by `newrun`). Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$12**, ceiling **24
pod-hours**. Authorised by Dan 2026-10-03, as the cap-6 companion to `rl-continue`, which runs the same protocol on cap 12.
**Read `docs/proposals/state-env/BRIEF_rl-continue.md` first.** Everything there applies, except as follows.

## What differs

- **Ladders:** `trajectory-cap6`'s three best-cap6 T1 ladders (Robbie's recipe, `lean_staten`, the cap-6 control
  set). They continue for rounds 9–16 from:
  - `hf://buckets/dan-pandori/nd-rl/trajectory-cap6/ckpts/tj6/ladder/la_T1_best6_s{0,1,2}_r8.pt`;
  - the found sets at `trajectory-cap6/artifacts/tj6/la_T1_best6_s{0,1,2}/found_8.jsonl`.

  Use `state_ladder_ei.py --start_round 9 --resume`. Replay comes from the cap-6 set, exactly as in
  `trajectory-cap6`.
- **Where cap 6 stands at r8:**
  - training targets: 4,257 / 4,235 / 4,204 of 4,495 (93.5–94.7 %);
  - new per round at r6 / r7 / r8: +53 / +20 / +16, +46 / +32 / +25, +52 / +43 / +31;
  - transfer pool: 88.9–90.6 %.
- **Group C:** `trajectory-cap6`'s own per-seed groups (52 / 52 / 60).
- **The question it adds:** does cap 6 converge towards cap 12's ceiling, or stop below it? Compare against
  cap 12's r8 (97.1–98.0 % of targets, 95.0–95.8 % transfer) and against `rl-continue`'s r16 once it exists.

## Pre-registration (before the first pod)

- **Expectation, per seed over rounds 9–16:** 40–70 new targets and 30–60 transfer theorems; group C unchanged within
  the sampling spread; r16 targets still below cap 12's r8.
- **Falsifier for "saturating":** ≥ 100 new targets on ≥ 2 of 3 seeds, or group C pass@256 rising beyond the seed
  spread.

## Deliverables

As in `rl-continue`, but named `rl-continue-cap6`:
- `preregistration/rl-continue-cap6.md`, `run_rl_continue_cap6.md` (≤ 250 words + a figure of rounds 1–16 with cap
  12 overlaid when available), `numbers.md` § rl-continue-cap6, `log.md`, and `STATUS.md` line
  `RL-CONTINUE-CAP6 DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/rl-continue-cap6/{ckpts,artifacts}`.
- Pods deleted; `touch ~/runs/rl-continue-cap6/executor.done`.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
