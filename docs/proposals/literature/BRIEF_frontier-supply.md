# Run brief: frontier-supply — new EI targets just past the frontier

Run id `frontier-supply`. Repository `~/work/frontier-supply` (branch `dan_frontier-supply` from the fork's
`origin/dan`, made by `newrun`; starts after `long-pool-2` is recorded). Role: executor. `AGENT_POLICY.md`
governs. Budget **$20**, ceiling **40 pod-hours**. Authorised by Dan 2026-09-29 (proposal 16 item 1).

Read first:
- nd-rl `docs/proposals/2026-09-29-literature-shortlist.md` §1.
- `docs/literature/2026-09-29-lit-review/REVIEW.md` (shortlist #2 and its notes on STP, Lee et al. and
  Goedel-Prover-V2).
- The `long-pool`, `long-pool-2`, `state-cap12` and ladder-A summaries (ladder-A's T2 / T4 reallocation
  did not move the frontier).
- `state_ladder_ei.py`, `state_sample.py`, `state_env.py`, `gen.py`.

**Code not on `dan`:** `lpool_reread.py` and `state-cap12`'s pod scripts (`pod/sc12/`, including `sn_seed.sh`, the SN-cap12 Stage-1 recipe) are only on `origin/dan_state-cap12` (`lpool_reread.py` also on `origin/dan_long-pool`). Take them with `git checkout origin/dan_state-cap12 -- lpool_reread.py pod/sc12/` and merge them with this run.

## Why

Expert iteration learns only on theorems where the model sometimes succeeds (p̂ > 0). The fixed target pool
has almost none past the frontier, and reallocating attempts within it did not help (ladder-A). The
literature's fix is **supply**: build new targets just past what the model can do, and keep the ones it
solves rarely (STP 2502.00212v4 §3.2 keeps p̂ ∈ (0, 1/4]; Lee et al. 2502.01612v2 §7.1: targets that are
"too difficult" hurt).

## Design

- **Model:** SN-cap12 (state policy on K12's cap-12 set). Stage-1 s0–s3 are in
  `hf://buckets/dan-pandori/nd-rl/state-cap12/ckpts/sc12/`. Train s4 and s5 with the identical recipe:
  6 seeds.
- **Arms:** from each Stage-1 checkpoint, run two ladders under `state-cap12`'s protocol
  (`state_ladder_ei.py`, 8 rounds, k 32, T 0.8, replay K12):
  - **control:** targets from `rl_targets.jsonl`;
  - **supply:** the same, plus supply targets, at the **same total attempts** per round (state the split).
- **Supply candidates**, rebuilt each round from the current model:
  - (a) extensions and mutations of theorems it solved this round, aimed at upper-bound lengths `L*`+1 …
    `L*`+4 (for example, add a hypothesis layer, chain two solved theorems, strengthen the goal);
  - (b) the open goal at the point of failure in failed state-env attempts, as a new theorem under its
    hypotheses.
- **Filter:** 32 attempts per candidate; keep p̂ ∈ (0, 1/4]. Report candidates made, passed, and passed
  per source (a) / (b).
- **No evaluation leakage:** candidates come only from `rl_targets` attempts and their own descendants.
  Check each against `transfer_long`, `transfer_long2` and every held-out pool by renaming class, and drop
  matches. Report the count.
- **Same-checkpoint variance** (lit-review reviewer, F3): on 2 seeds, run the control arm twice with
  different EI sampling seeds. That gives the arm-vs-arm spread with no treatment, which the comparison
  needs.
- **Read-out:** final checkpoints on `transfer_long2` (per upper-bound bin) and `rr600` 13–16, k 256,
  `max_steps` 96, `lpool_reread.py`.

## Pre-registration (before the first pod)

- **Quantity:** the per-seed solve count in the two bins nearest the control's frontier (chosen from
  `long-pool-2`'s re-read) and `L*`.
- **Statistics:** the paired MDD, using the same-checkpoint spread once measured (state the provisional
  value and update it). Per-seed values, IQM, stratified-bootstrap 95 % CI.
- **Expectation:** ≥ 5 % of candidates pass the filter, and supply beats control by more than the MDD.
- **Falsifier:** < 5 % pass, or no difference beyond the MDD.

## Deliverables

- The supply code, merged into the fork's `dan` after review.
- `preregistration/frontier-supply.md`, `run_frontier_supply.md` (≤ 400 words + figure), `numbers.md`
  § frontier-supply, `log.md`, and `STATUS.md` line `FRONTIER-SUPPLY DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/frontier-supply/{ckpts,artifacts,data}`.
- Pods deleted; `touch ~/runs/frontier-supply/executor.done`.

Never run `test_run_once.sh`.
