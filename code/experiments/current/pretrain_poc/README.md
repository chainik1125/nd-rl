# Pretrain workstream: proof-of-concept loop

Status: first pass, agent-written (Claude, 2026-09-22), **not reviewed by Robbie**. One run at a
time, no autoresearch yet. Proposal and constraints:
[`docs/proposals/2026-09-20-pretrain-workstream-constraints.md`](../../../../docs/proposals/2026-09-20-pretrain-workstream-constraints.md)
(also unreviewed).

"Option A at every step": Dan's cap-6 Stage-1 file with the `abs` codec, plain expert iteration
(ladder rung T1), the ladder transfer pool with `L_true` labels. Model 4 layers, d = 256, 3.2M
params (cap 10M). Pretrain and RL are each wall-clock capped (default 300 s) on a pinned GPU
(A40). Primary number: dev-transfer theorems solved at `L_true >= 9`, frozen vs final, equal attempts.

## Files

- `pipeline.py`: pretrain → eval(frozen) → expert iteration → eval(final) → `summary.json`.
  Imports Dan's fork (`ND_FORK`) for model, tokenizer, sampler, verifier, judge; changes none of it.
- `remote.sh`: on the pod, checks out `dan-pandori/nd-takehome` at the pinned commit, unpacks the
  Stage-1 file, runs the pipeline.
- `pod.py`: rents one A40 on RunPod, ships the two files, runs, pulls artifacts to
  `artifacts/pretrain_poc/<name>/` (repo `artifacts/` is gitignored), **terminates the pod in a
  `finally`**. Needs `RUNPOD_API_KEY` (`~/repos/nd-takehome/.envrc.local`).

```bash
source ~/repos/nd-takehome/.envrc.local
python pod.py --name run1 -- --pretrain_secs 300 --rl_secs 300     # pipeline args after --
python pod.py --list ; python pod.py --kill all
ND_FORK=<fork checkout> python pipeline.py --out /tmp/smoke --smoke  # CPU plumbing check
```

## Frozen in this loop

Verifier, Stage-1 file (asserted cap 6), RL-target subsample (1,500 of 4,495, seed 0), dev half of
the transfer pool (renaming-class hash parity), EI hyperparameters (k = 32, T = 0.8, 600 ft steps,
20k retained records), seed 0. The three pools are asserted class-disjoint at start.

## Known gaps

- Sampler is Dan's per-token Python loop; it is the run-time bottleneck. Nothing is compiled.
- The RL stage stops before a round that would not fit the budget; the first round always runs,
  so `rl.over_budget` in `summary.json` says whether the cap was actually met.
- No pre-registration, single seed, no reviewer pass; this is a plumbing check, not a result.
