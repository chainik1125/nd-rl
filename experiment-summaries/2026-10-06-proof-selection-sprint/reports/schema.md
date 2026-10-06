---
written_on: 2026-10-06
written_by: agent:claude
---

# Shared data schema for the selection sprint

All heavy files live under `~/runs/proof-selection/`. Code imports Dan's environment from
`~/runs/proof-selection/nd-takehome` (`sys.path.insert(0, ...)`), Python `~/venvs/proof-state/bin/python`,
`PATH=$HOME/.elan/bin:$PATH`, `ND_SAMPLE_PATH=fast`, `LEAN_GATE_WORKERS=16`.

## 1. Pools: `pools_v2/<set>__<gen>__k<k>.jsonl` (one row per theorem)

```
{"name": str, "prompt": str, "set": "rl_targets|dev1108|rr600|holdout250|textbook72", "ref_lines": int|null,
 "gen": "s0_r16", "k": int, "sampling": {"temperature": 0.8, "max_action": 512, "max_steps": 96, "seed": int},
 "attempts": [
   {"i": int,                       # 0..k-1, generation order (independent fresh contexts; order carries no information)
    "base": int,                    # SN name base used by the environment for this attempt (needed for exact replay)
    "actions": [[int, ...], ...],   # model-emitted action token ids per step, as decoded by tok.decode_action (eos stripped)
    "states": null,                 # states are recomputable by replay; not stored
    "done": bool,                   # environment reached a finished proof (top-level exact). Text-visible, legal.
    "steps": int,
    "nd": str|null,                 # assembled ND proof text when done (what Lean judged); null otherwise
    "gen_tokens": int               # total sampled tokens (cost accounting)
   }, ...]}
```

Nothing in a pool row says whether Lean accepted the proof, and no failure *reason* strings appear in it.

## 2. Labels: `pools_v2/<same stem>.labels.jsonl`

```
{"name": str, "valid": [bool, ...], "reason": [str|null, ...]}   # aligned with attempts[i]
```

Loaded only by training/evaluation code through `load_labels()`. Selector *inputs* are built from §1 only.

## 3. Scores: `scores/<pool stem>__<scorer>.jsonl`

```
{"name": str, "scores": [float, ...]}   # aligned with attempts; higher = more likely valid; -inf allowed
```

Scorer names: `random` (uniform noise), `first` (= -i), `completion` (done ? 1 : 0), `shortest` (completion then
-steps), `loglik_mean`, `loglik_sum`, `loglik_min_step`, `entropy_mean` (teacher-forced under the frozen r16 with the
environment-assigned name positions masked), `completion_loglik` (completion, ties broken by loglik_mean),
`verifier_s<seed>` (learned head), `verifier_ens`.

## 4. Selection evaluation (`evaluate_selection.py`)

For pool size m in {1, 2, 4, 8, 16, 32, 64}: draw R=200 random m-subsets per theorem (seeded), pick argmax score,
report mean accuracy of the selected attempt, oracle pass@m on the same subsets, random expected accuracy, and the
fraction of the (oracle − random) gap recovered. Paired theorem-level bootstrap (B=1000) for differences between scorers.
Report full pools and pools with ≥ 1 valid candidate separately.

## 5. Stopping evaluation (`stopping.py`)

Policy consumes attempts in order i = 0, 1, ...; after each attempt it has the score of every attempt so far;
it stops when the best score so far ≥ τ or when i+1 = N_max; it returns the best-scoring attempt so far. Report
accuracy vs mean attempts (and vs mean generated tokens) sweeping τ, for each scorer, against fixed-k with the same
scorer, "stop at first finished", and the oracle (stop at first valid). Thresholds are chosen on dev only; test
curves are read once.
