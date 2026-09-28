# Run brief: state-cap12 — the two levers together: proof state and longer training proofs

Run id `state-cap12`. Repository `~/work/state-cap12` (branch from the fork's `origin/dan`, after
`long-pool` has merged its pool). Role: executor. `AGENT_POLICY.md` governs. Budget **$20**, ceiling
**40 pod-hours**. Authorised by Dan 2026-09-28. Read first: the `state-env`, `cap-horizon` and `long-pool`
summaries in `~/nd-rl/experiment-summaries/`, `STATE_ENV.md`, `data/ladder/POOLS.md`.

## Question

Two changes each moved something and neither moved `L*` past 12. **The proof state** (`state-env`) lifts
what the model reaches below the wall (frozen transfer 779–975 vs 114–158; T1 1,348–1,557 vs 890–965) but
reached `L_true` ≥ 13 only 1–3 times. **Longer training proofs** (`cap-horizon`) raise the frozen
frontier (frozen `L*` 9 → 12 from cap 6 to cap 12) but T1 `L*` stays 12 at every cap. Together, does the
frontier move — T1 `L*` ≥ 13 on `long-pool`'s pool, which can measure 13–16?

## Design (a suggestion)

SN-v2 configuration (environment-assigned names; `Env(assign=True)`), trained on `cap-horizon`'s **K12**
set (cap 12, flat over lengths 2–12, 155,000 proofs; `hf://buckets/dan-pandori/nd-rl/cap-horizon/data/kh/`)
converted to (state, action) pairs — check the environment's gates on it first (round trip, Lean's state vs
the renderer's on longer proofs, replay), since the proofs are twice as long as anything the environment has
seen. Two Stage-1 seeds; held-out greedy; ladder T1 + frozen (8 rounds × 32) on `data/ladder/rl_targets.jsonl`,
read out on **both** the original transfer pool and `long-pool`'s `transfer_long.jsonl`. Compare with
`state-env`'s SN-v2 (cap 6) and `cap-horizon`'s K12 (whole-proof, cap 12) on the same readouts
(`long-pool` re-read those models on its pool).

## Pre-registration (before the first pod)

Expected T1 and frozen `L*` on the long pool, per-bin solve rates at 13–16, and the minimum detectable
difference at n = 2. Falsifier for "the levers compound": T1 `L*` on the long pool ≤ the larger of
SN-v2-cap-6's and K12's on both seeds.

## Deliverables

`preregistration/state-cap12.md`; `run_state_cap12.md` (≤ 400 words + a figure: per-bin solve rate on the
long pool — SN-v2 cap 6, K12 whole-proof, SN-v2 cap 12 — T1 and frozen); `numbers.md` § state-cap12;
`log.md`; checkpoints to the bucket as written; `STATUS.md` line `STATE-CAP12 DONE <UTC>`; pods deleted;
`touch ~/runs/state-cap12/executor.done`.
