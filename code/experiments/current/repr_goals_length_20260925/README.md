---
written_on: 2026-09-25
written_by: agent:claude
---

# Representation variants and goal notes for length generalisation: working folder

This folder holds the code of two sibling experiments as they ran on 2026-09-25. Both ask whether a change to how
proofs are written helps the prover past the lengths it was trained on. Both train on true lengths 6-11 and test at
6-18. They share one baseline: the `abs` variant of `repr/`.

- `repr/`: five representation variants (abs, rel, pos-rand, nope, pos-line), 3 seeds each. pos-rand stopped after
  pretraining, pending a decision.
- `goals/`: the prover writes its own `GOAL <formula>` notes between proof lines, 3 seeds.

Results and evidence:
[experiment-summaries/2026-09-25-representation-and-goals-length](../../../../experiment-summaries/2026-09-25-representation-and-goals-length/README.md).

**About the copies.**
- *Byte-identical.* Every file except this README is a copy of a file in `/scratch/cowifornia/nd-rl/ab3/repr/` or
  `/scratch/cowifornia/nd-rl/ab3/goals/`, checked with `cmp`. The files are not reformatted and should be excluded
  from ruff, like the other frozen snapshots.
- *Renamed on copy* (contents unchanged): each experiment's `README.md` became `repr_README.md` and
  `goals_README.md`. The generated harness of one frozen copy became `frozen_copy/harness.py`, and its manifest
  `frozen_copy/*_MANIFEST_*.json`.
- *Nothing heavy.* There are no checkpoints, data files, evaluation rows or sample dumps here.
- *Where they write.* The scripts use hard-coded scratch paths, and most put outputs next to themselves. To re-run
  anything, work in the scratch copies.
- *Not copied.* `repr/lib/` and `goals/lib/` also hold unchanged copies of `ab3/frozen/lib` (`nd_verify/`,
  `normalize.py`, `keys.py`, `prune.py`, `eval_set.py`, `train.py`). They are not repeated here. `repr/lib/tokenizer.py`
  is also unchanged from `ab3/frozen/lib` and is copied only because `rel` uses its `rel` mode.

## Files: repr/

| file | what it is |
|---|---|
| `repr_README.md` | the experiment's own README: question, design, choices and why, tests, commands, results, open questions |
| `common_repr.py` | variants, seeds (pretrain 0-2, RL 101-103), paths, model builder and loader, GPU lease |
| `lib/model.py` | the frozen GPT with a positional variant in its cfg: `pe` = rope (frozen, bit-identical), none (NoPE) or line (two-part rotary: 8 of 16 pairs by line index, 8 by offset in line); `pos_train` = rand for pos-rand; position helpers for the KV-cache samplers |
| `lib/sample.py` | frozen `lib/sample.py` with positions from the model (`init_pos`, `next_pos`) |
| `lib/posenc.py` | training positions: a sorted random subset of range(1024) per sequence (pos-rand), or the model's own positions |
| `lib/tokenizer.py` | unchanged frozen tokenizer (modes `abs` and `rel`) |
| `pretrain.py` | the pretraining team's run 039 recipe (Muon for 95%, then AdamW) at a fixed 2,911 steps; also measures the released 039 checkpoints (`--ref`) |
| `make_frozen.py` | builds `frozen/<variant>_s<seed>/`: `ab3/frozen` with the variant lib, the variant's `pretrain.pt`, and three textual edits to `harness.py` (tokenizer mode and variant read from the checkpoint) |
| `frozen_copy/harness.py`, `frozen_copy/REPR_MANIFEST_abs_s0.json` | the generated harness (identical in all 12 frozen copies, sha256 `d093189c1b47…`) and one manifest with the diff against `ab3/frozen/harness.py` |
| `rl_repr.py` | `lengthexp/rl_C.py` (arm C: targets of true length 6-11, fine-tune proofs of at most 11 lines) with representation plumbing only: model-driven positions in the samplers and packed batches, index cap for abs only. CONFIG equals arm C's |
| `evaluate.py`, `sampler.py` | lengthexp's evaluator and sampler with the variant loader, model-driven positions, `--pos-eval rand` and the frozen test-set copy `testsets/` |
| `launch.py` | pipeline driver: pretrain, frozen copy, RL, evaluation per variant and seed, at most 4 jobs on the GPU lock |
| `analyse.py` | the analysis (`uv run --script`); imports lengthexp's statistics; writes `report/report.md`, figures and `report/results.json` |
| `record_extracts.py` | CPU extracts for this record: `report/record/runs_summary.json`, `pretrain_summary.json`, `derived.json` |
| `tests/test_variants.py` | parameter counts, cfg round trip, frozen equivalence of abs, line positions, random subsets, KV-cache decoding against the full forward for every sampler and variant, packed logits |
| `tests/test_rel_tokenizer.py` | rel round trip on all cap-6 and pool proofs; verifier and checker agree on decoded rel text |
| `tests/test_meter_variants.py` | the frozen compute meter charges every variant like the baseline |
| `tests/test_equivalence.py` | repr sampler = lengthexp sampler; rl_repr = rl_C for abs; CONFIG = arm C |
| `tests/run_unit.sh`, `tests/run_smoke.sh`, `tests/with_gpu.py`, `tests/compare_smoke.py` | test runners on a locked GPU; the smoke pipeline; harness smoke comparison rl_C vs rl_repr |
| `tests/posrand_diag.py` | pos-rand held-out greedy under contiguous, random and evenly spaced positions (no training) |

## Files: goals/

| file | what it is |
|---|---|
| `goals_README.md` | the experiment's own README: question, what changes, how it is run, results, caveats |
| `DESIGN.md` | annotation rules and their justification, tokenizer, stripping, pipeline edits, token and compute impact |
| `lib/goals.py` | the annotator (agenda of owed goals, display rules), `strip`, consistency check |
| `lib/gtokenizer.py` | the abs tokenizer plus one `GOAL` token (vocabulary 100, `tok_mode` `abs_goals`) |
| `common_goals.py` | paths and constants |
| `build_data.py` | annotates and validates the cap-6 train and held-out files and the RL-found proofs; writes `data/` and `data/VALIDATION.json` |
| `pretrain.py` | `repr/pretrain.py` with four edits (annotated data, goals tokenizer, index shift in `abs_goals` mode, stripped held-out checks with max_new 768) |
| `make_frozen.py`, `frozen_copy/harness.py`, `frozen_copy/GOALS_MANIFEST_goals_s0.json` | the harness copy (goals tokenizer, stripped held-out check, annotated retention file) and one manifest with the diff |
| `make_rl.py`, `rl_goals.py`, `rl_goals.diff` | the RL file: `lengthexp/rl_C.py` plus the goals edits (verify stripped samples, train on the model's own notes, max_new 768) |
| `make_eval.py`, `evaluate.py`, `evaluate.diff` | the evaluator: `lengthexp/evaluate.py` plus stripping and annotation statistics |
| `launch.py` | pipeline driver (two seed chains at a time, every GPU job through the lock) |
| `analyse.py` | goals against repr's abs baseline and arm C; writes `report/report.md`, `fig_main.png`, `fig_hazard.png`, `results.json` |
| `analysis/mechanism.py` | conjunction theorems: which part the notes name, whether the owed part is named after the first is derived |
| `analysis/structure_stats.py` | design statistics: box and goal structure, token overhead per annotation policy |
| `tests/test_goals.py`, `tests/test_rl_goals.py`, `tests/smoke.sh` | annotation and strip tests on 20k proofs per source; RL edits; GPU plumbing smoke |

## Scratch locations (not in Git)

- **Frozen harness and data.** `/scratch/cowifornia/nd-rl/ab3/frozen/` (run-3 harness, compute meter,
  `DEFAULT_FLOPS = 3.9e15`, `data/train_cap6.jsonl` = the 039 fork's `data/train.jsonl`, sha256 `9c024cd9…`).
- **Released 039 checkpoints.** `/scratch/cowifornia/nd-rl/checkpoints/runs/039_muon-adamw-finish-5pct_s{0,1,2}/`.
- **repr/.** `/scratch/cowifornia/nd-rl/ab3/repr/`:
  - `pretrain/<variant>_s<seed>/` (pretrain.pt, summary.json, log) and `pretrain/ref039_s<seed>/`;
  - `frozen/<variant>_s<seed>/` (12 copies; none for pos-rand);
  - `runs/<variant>_s<seed>/` (final.pt, round_*.pt, metrics.jsonl, found_targets.jsonl, summary.json);
  - `eval/<variant>_s<seed>/` (per-theorem rows, unit summaries, `samples/*.jsonl.gz`);
  - `testsets/` (frozen copy of lengthexp's final draw of 2026-09-25 02:27 UTC: std sha256 `62db9018…`, matched
    `172e376e…`);
  - `report/`, `report/record/`, `report_interim/` (superseded), `tests/out/` (smoke outputs), `STATUS.md`, `logs/`.
- **goals/.** `/scratch/cowifornia/nd-rl/ab3/goals/`: `data/` (annotated cap-6 files, 152 MB), `pretrain/`,
  `frozen/`, `runs/`, `eval/`, `testsets/` (the same frozen draw), `report/`, `analysis/`, `STATUS.md`.

**Checkpoint sha256 prefixes** (full hashes in the summary's `evidence/pretrain_summary.json` and
`evidence/runs_summary.json`):

| run | pretrain.pt, seed 0 / 1 / 2 | RL final.pt, RL seed 101 / 102 / 103 |
|---|---|---|
| abs | `93066a7c1434` / `cfe7344bb5a7` / `1207a9cae8e2` | `5522950870d2` / `a0a4c3deb4f3` / `b8c70e2c58c2` |
| rel | `877158c13263` / `42c5e7a5f7e7` / `4b1b85cd7451` | `eeddf9795970` / `cd5092e5015d` / `a1114bab8e85` |
| nope | `25371b08b15c` / `44b2715c67e7` / `bcd581e39f3b` | `7b1fc34d99b3` / `64df6daf6d50` / `e56696349bd8` |
| pos-line | `faf4405306cb` / `1d4561211bc5` / `209b656099fa` | `7044ab66315c` / `d40bccc3304c` / `e752d5cdc915` |
| pos-rand | `14080ffc451e` / `9857dc03d48a` / `8a66b0050590` | not run (decision pending) |
| goals | `72ab3557d2ae` / `6b8ea27d11a2` / `02307905f3a3` | `6b799af6c82d` / `e2750fe8d555` / `118368dace10` |
| released 039 (reference) | `08e2644a8560` / `418af3f9df38` / `fab74b363784` | - |

## Reproduce

From `/scratch/cowifornia/nd-rl/ab3/repr`, with `PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python`:

```
$PY tests/with_gpu.py -- bash tests/run_unit.sh          # unit tests on one locked GPU
$PY tests/with_gpu.py -- bash tests/run_smoke.sh         # smoke pipeline, all variants; rl_C vs rl_repr comparison
$PY pretrain.py --ref 0                                  # released 039 s0 measured with our code (also 1, 2)
nohup $PY launch.py run --hold-rl posrand > logs/run.out 2>&1 &   # the whole pipeline, restartable
$PY launch.py status
$PY tests/with_gpu.py -- $PY tests/posrand_diag.py pretrain/posrand_s0/pretrain.pt
uv run --script analyse.py                               # report/report.md, figures, results.json
$PY record_extracts.py                                   # report/record/*.json (CPU)
```

From `/scratch/cowifornia/nd-rl/ab3/goals`:

```
$PY tests/test_goals.py --n 20000 && $PY tests/test_rl_goals.py
$PY build_data.py && $PY analysis/structure_stats.py
$PY make_rl.py && $PY make_eval.py                       # rl_goals.py and evaluate.py (+ .diff files)
nohup $PY launch.py all --chains 2 > logs/launch.log 2>&1 &
uv run --script analyse.py && $PY analysis/mechanism.py  # report/report.md, figures, analysis/mechanism.md
```

`goals/analyse.py` reads repr's abs evaluations from `/scratch/cowifornia/nd-rl/ab3/repr/eval/abs_s*` and arm C's
from `/scratch/cowifornia/nd-rl/ab3/lengthexp/eval/C_s*`.
