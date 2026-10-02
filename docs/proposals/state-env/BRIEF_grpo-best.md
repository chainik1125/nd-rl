# Run brief: grpo-best — GRPO against expert iteration on the model organism, from the same starting checkpoints

Run id `grpo-best`. Repository `~/work/grpo-best` (branch `dan_grpo-best` from the fork's `origin/dan`, made by `newrun`).
Role: executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$45**, ceiling **90 pod-hours**.
Proposal: nd-rl `docs/proposals/2026-09-29-literature-shortlist.md` item 3, on Robbie's recipe as Dan asked
(2026-10-02).

Read first:
- The `grpo-state` summary and its `preregistration/grpo-state.md` (on `origin/dan_grpo-state`): the design this
  run adapts.
- The `trajectory` and `best-state` summaries.
- `grpo_state.py` and `grpo_adv.py`, which are on `dan`.

## Why

The run4 review found GRPO acquiring the depth-3 pattern from zero where expert iteration mostly did not. That was
on the old token format, and no checkpoint survives. `trajectory` now gives the cleanest possible comparison:
- three cap-12 starting checkpoints, end of pretraining, Robbie's recipe;
- an EI ladder from each, with per-theorem, per-round results;
- a group of theorems that EI never solved (group C).

The question is whether GRPO, and in particular the variants that reward rare correct proofs, solves what EI from
the **same checkpoint** could not.

## Design

- **Starting checkpoints:** `trajectory`'s `stage1_best12_s{0,1,2}_b1200.pt`
  (`hf://buckets/dan-pandori/nd-rl/trajectory/ckpts/tj/`).
- **The EI arm is not re-run.** It is `trajectory`'s T1 ladders, with r1–r8 checkpoints and read-outs already in
  the bucket.
- **New arms, 3 seeds each:** GRPO default, unlikeliness rank penalty (β_rank 0.25), and pass@k advantage. Add the
  distinct-proof bonus only if the budget allows.
- **Same targets as the ladder** (`rl_targets.jsonl`). **Same total sampled attempts as the EI ladder**
  (1,793,960 per clean ladder in `best-state` / `trajectory`), so the arms are compute-comparable on samples.
  Record GPU-seconds and flag any arm above 1.25× EI's.
- **Model class:** `grpo_state.py` was built for the 3.2M GPT. Load Robbie's ALiBiGPT through the same
  checkpoint path that `state_ladder_ei` uses for best-state (it is on `dan`), and test the gradient on it before
  the first pod. Write down the optimizer for the RL updates.
- **Read-outs, matching `trajectory`:** the 322 theorems (textbook72 + holdout250), pass@k up to 256 with both
  sampling seeds, at the round-equivalents of EI's r2, r4 and r8. Plus the dev metric and held-out greedy at the
  end.

## Analysis

- **Support:** of `trajectory`'s group C (EI never solved it, per seed), how many does each GRPO arm solve at
  k 256? This is the headline.
- **Sharpening vs support:** pass@1 against pass@256 for each arm, by `trajectory` group.
- **Retention:** held-out greedy and group A.
- Paired per seed with the EI arm. Per-seed values, IQM, and the MDD stated in the pre-registration.

## Deliverables

- `preregistration/grpo-best.md`, `run_grpo_best.md` (≤ 400 words + figures), `numbers.md` § grpo-best, `log.md`,
  and `STATUS.md` line `GRPO-BEST DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/grpo-best/{ckpts,artifacts}`.
- Pods deleted; `touch ~/runs/grpo-best/executor.done`.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
