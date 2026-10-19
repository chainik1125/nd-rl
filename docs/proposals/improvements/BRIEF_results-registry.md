# Run brief: results-registry — one results table; checkpoints to the bucket on save (proposal 15, item 5)

Run id `results-registry`. Repository `~/work/results-registry` (branch `dan_results-registry` from the
fork's `origin/dan`, made by `newrun`; starts after `fast-stage1` and `lean-prefilter` are merged, so
build on their code). Role: executor. `AGENT_POLICY.md` governs. Budget **$1**, ceiling **2 pod-hours**
(smoke tests only). Authorised by Dan 2026-09-28 (proposal 15's queue). Read first: proposal 15 §5.

## Why

Checkpoints have been lost or never uploaded in three runs (the 2026-09-22 cleanup; no Lean-format EI
checkpoint was uploaded before `support-curves`, which had to retrain two; `support-curves`' seed-1 EI
checkpoint). Results live in `numbers.md` prose across 27 branches; every reviewer re-derives from scratch
and every librarian re-reads it all.

## What to build

1. **`save_ckpt` uploads before it returns** — to `hf://buckets/dan-pandori/nd-rl/<run-id>/ckpts/…` (run id
   from `ND_RUN_ID`), records the md5 and URI, and fails loudly if the upload fails (an explicit offline
   flag for tests).
2. **`record.py`**: `record(metric, value, n=None, **labels)` appends a row — run id, arm, seed, git SHA,
   the script's full config, dataset file + hash, model checkpoint md5 + URI, metric, value, n, source
   file, UTC — to `artifacts/<run-id>/registry.jsonl`, synced to `registry/<run-id>.jsonl` in the bucket.
   `registry_merge.py` builds one table from all runs' files.
3. **Instrument the main scripts** (`train.py`, `eval_set.py`, `coverage.py`, `ladder_ei.py`,
   `expert_iter.py`, `support.py`, the `state_*` scripts if merged) to record their headline numbers.
4. **Backfill** rows for every reviewed run from its `summary.json` files, labelled `backfilled`, so the
   table starts complete.

## Acceptance (pre-register)

A smoke run end to end (a few hundred training steps, one eval, one tiny coverage job) produces registry
rows whose checkpoint URIs download to files with the recorded md5; the merged table answers "every
held-out accuracy of the control checkpoints, by run" with no reading of prose; the backfill reproduces
three chosen headline numbers from their summaries exactly.

## Deliverables

Merged into the fork's `dan` after acceptance (the reviewer follows); `REGISTRY.md` (schema, how to record,
how to query); proposed policy text for `AGENT_POLICY.md` in `QUESTIONS.md` (Dan's session adopts it);
`preregistration/results-registry.md`; `run_results_registry.md` (≤ 250 words); `log.md`; `STATUS.md` line
`RESULTS-REGISTRY DONE <UTC>`; pods deleted; `touch ~/runs/results-registry/executor.done`.
