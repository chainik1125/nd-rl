# Run brief: long-pool — a transfer pool that can measure the length frontier

Run id `long-pool`. Repository `~/work/long-pool` (branch from the fork's `origin/dan`). Role: executor.
`AGENT_POLICY.md` governs. Budget **$8**, ceiling **16 pod-hours** (mostly CPU). Authorised by Dan
2026-09-28. Read first: `data/ladder/POOLS.md` (how the ladder's pools were built), `gen.py` (`--ladder`
and the strict long generator), `minlen.py` (the bounded prover behind `L_true`), the `cap-horizon` and
`state-env` summaries in `~/nd-rl/experiment-summaries/`.

## Why

Every length claim in this project is read off `data/ladder/transfer.jsonl`, whose `L_true` counts are
7: 300 · 8: 300 · 9: 1,010 · 10: 451 · 11: 99 · 12: 102 · **13: 13 · 14: 10** — 23 theorems at ≥ 13. With
`L*` defined as ≥ 5 solved at `L_true` ≥ L, `L*` = 13 needs 5 of those 23 and `L* > 14` cannot be
measured at all. `state-env` solved 1–3 theorems at ≥ 13 and could not say whether that was a rate or one
lucky theorem; `cap-horizon`'s `L*` was censored at 14.

## What to build

A second transfer pool, **`data/ladder/transfer_long.jsonl`**, with **≥ 100 theorems per `L_true` bin from
11 to 16** (more if cheap), from the same generator family as the ladder's pools, labelled by `minlen.py`.
- Class-disjoint (renaming class) from every training set in use (the control set, cap-horizon's K8–K14
  sets), from `data/ladder/rl_targets.jsonl`, and from the existing transfer and held-out pools. Report it.
- `minlen.py` gets expensive at these lengths: run it on a many-core CPU pod (or a GPU pod's cores),
  report the timeout rate per bin, and treat a timeout as "label unknown", never as a label. Say what
  bounds you used and how `L_true` is certified (as `data/ladder/POOLS.md` does).
- A `POOLS.md` section: generation settings, per-bin counts, shape table, the disjointness numbers.

Then **re-read the existing models on it** (no training): the final T1 and frozen checkpoints of
`state-env` (S and SN-v2, both seeds), the whole-proof C0 T1 checkpoints, and `cap-horizon`'s K12/K14 T1
checkpoints if they are in the bucket — k = 256 per theorem, T 0.8 — giving per-bin solve rates and an
`L*` that is no longer censored at 14.

## Pre-registration (before the first pod)

The per-bin counts you expect to reach and the generation budget; for the re-read, your expected `L*` per
model on the new pool.

## Deliverables

The pool and its `POOLS.md` section, merged into the fork's `dan` (it is data every later run will use);
`preregistration/long-pool.md`; `run_long_pool.md` (≤ 300 words + a figure: per-bin solve rate by model);
`numbers.md` § long-pool; `log.md`; `STATUS.md` line `LONG-POOL DONE <UTC>`; bucket
`hf://buckets/dan-pandori/nd-rl/long-pool/{data,artifacts}`; pods deleted;
`touch ~/runs/long-pool/executor.done`.
