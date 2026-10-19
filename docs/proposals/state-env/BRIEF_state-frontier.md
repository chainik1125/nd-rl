# Run brief: state-frontier — settle `state-env`'s length question, and test the "lottery" finding

Run id `state-frontier`. Repository `~/work/state-frontier` (branch from the fork's `origin/dan`, which
now carries the state environment: `state_env.py`, `state_*`, `STATE_ENV.md`). Role: executor.
`AGENT_POLICY.md` governs. Budget **$12**, ceiling **24 pod-hours**. Authorised by Dan 2026-09-28.
Read first: the `state-env` summary in `~/nd-rl/experiment-summaries/` and `review_state-env.md` on
`origin/dan_state-env` (its "Next step" list is this run), `STATE_ENV.md`, `NOISE_FLOOR.md`.

Set `LEAN_GATE_DUMP` for every job (the review found `state-env` did not store the literal sampled text).

## Two questions

1. **Does the state reach `L_true` ≥ 13 at a real rate, or was it one lucky theorem?** `state-env`: `L*` 12
   in all six T1 runs; only 3 distinct `L_true` ≥ 13 theorems solved by any run, one of them 4 of 7 hits;
   C0 0. The transfer pool has only 23 theorems at ≥ 13.
2. **Does the state remove the depth-3 "lottery"?** All 6 state-conditioned Stage-1 checkpoints landed in
   the high depth-3 mode (0.860–0.956) against the control configuration's 0.462 rate — ≈ 0.01 by chance;
   the reviewer calls it suggestive. If it holds, `NOISE_FLOOR.md`'s widest quantities come from state
   tracking, and state-conditioned experiments need far fewer seeds.

## Design (a suggestion)

- **Re-sample the final checkpoints on the long theorems.** Every existing final T1 checkpoint (S s0/s1,
  SN-v2 s0/s1 from `state-env`; C0's T1 s0/s1 from `ds-generator`/`cap-horizon` — whichever are in the
  bucket) and their Stage-1 bases, at k = 256 per theorem, T 0.8, on the 23 `L_true` ≥ 13 transfer theorems
  and the `L_true` 11–12 bins (201). Per-theorem success counts, paired across models. The whole-proof
  models sample whole proofs; the state models sample in the environment (`Env(assign=True)` for SN-v2).
- **More seeds.** Pre-register which configuration: the reviewer's suggestion is arm **S** (the
  pre-registered primary) for the headline question — Stage-1 seeds s2, s3, then T1 + frozen ladders — and
  the forward-looking configuration is **SN-v2** (environment-assigned names, the best arm). Suggested:
  S s2/s3 through T1 + frozen (question 1 at n = 4), and SN-v2 s2–s5 through Stage-1, held-out greedy and
  a frozen ladder (question 2 at n = 6 for SN, and a base-reachability floor for the state model).
- **The lottery test:** the depth-3 slice of held-out greedy per Stage-1 seed, read as the fraction in the
  high mode (> 0.44) with a Wilson interval, against `NOISE_FLOOR.md`'s 0.462 [0.333, 0.595].

## Pre-registration (before the first pod)

Per question, the expected result and the minimum detectable difference at your n (policy). For question
2, a falsifier: e.g. "the state removes the lottery" is dead if ≥ 2 of the new SN seeds land in the low
mode.

## Deliverables

`preregistration/state-frontier.md`; `run_state_frontier.md` (≤ 400 words + a figure: per-theorem success
at `L_true` ≥ 11, state vs whole-proof; depth-3 slice per seed against the control's distribution);
`numbers.md` § state-frontier; `log.md`; `artifacts/sf2/summary.json`; checkpoints to the bucket as they
are written; `STATUS.md` line `STATE-FRONTIER DONE <UTC>`; pods deleted;
`touch ~/runs/state-frontier/executor.done`.
