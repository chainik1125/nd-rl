---
written_on: 2026-09-25
written_by: agent:claude
---

# Length generalisation of RL-trained proofs: working folder

This folder holds the code of the length-generalisation experiment as it ran on 2026-09-24 and 2026-09-25. The
experiment compares:
- the pretrained prover;
- RL arms A, B and C, trained on targets up to 7, 9 and 11 lines;
- arm D, which is C plus constructed composition targets;
- the ANDI composition diagnosis.

Results and evidence: [experiment-summaries/2026-09-25-length-generalisation](../../../../experiment-summaries/2026-09-25-length-generalisation/README.md).

**About the copies.**
- *Byte-identical.* Every file except this README is a copy of the file in
  `/scratch/cowifornia/nd-rl/ab3/lengthexp/`, checked with `cmp`. The files are not reformatted and should stay excluded
  from ruff, like the other frozen snapshots.
- *Nothing heavy.* There are no checkpoints and no sample dumps here.
- *Where they write.* The scripts use hard-coded scratch paths, and `common.py` puts outputs next to itself (`LE =`
  its own directory). To re-run anything, work in the scratch copy, or copy this folder to scratch first. Otherwise
  `eval/`, `runs/` and `report/` are created inside the repository.

## Files

| file | what it is |
|---|---|
| `lengthexp_README.md` | the experiment's own README on scratch: question, design, how to read the plot, commands, arm D, caveats |
| `common.py` | shared constants (arms, seeds, lengths 6-18, k = 32, T = 0.8, max_new 1024, test-set sources, training files for the disjointness check), pkey, pass@k |
| `build_testsets.py` | builds `testsets/std.jsonl` (L6-18) and `testsets/matched.jsonl` (L6-14), draws 300 per length with a fixed seed, checks pkey disjointness, writes `MANIFEST.json` |
| `ndcheck.py` | per-sample verdict (nd_verify) plus the first rejected line from an incremental checker (run 2's LineChecker with reason codes) |
| `sampler.py` | compacting temperature sampler; same distribution as `frozen/lib/sample.generate` |
| `evaluate.py` | evaluates one checkpoint on every test unit through the GPU lock; writes per-theorem rows, unit summaries and every sample |
| `launch.py` | `prepare` (arm files from the seed recipe), `smoke`, `train`, `eval`, `analyse`, `all`, `status` for arms A-C |
| `rl_A.py`, `rl_B.py`, `rl_C.py` | the seed recipe `ab3/seed/rl.py` (sha256 `be380193…`) with `max_len_train` and `max_proof_lines_train` set to 7, 9 and 11 (`arms.json` has the one-line diffs) |
| `rl_D.py` | byte copy of `rl_C.py`; arm D differs only in its RL pool (`ab3/frozen_comp`) |
| `arms.json` | seed sha256, knob lines and per-arm sha256 written by `launch.py prepare` |
| `watch_d.py` | queued each arm-D run's evaluation, and D_s101's composition part evaluation, when the run finished |
| `analyse.py` | the main analysis (`uv run --script`); writes `report/report.md`, figures, `report/results.json` |
| `conclusion_D.md` | the hand-written conclusion that `analyse.py` inserts into report section 8 (D vs C) |
| `fake_finish.py` | CPU re-analysis of the saved samples: failed samples whose first rejected line writes the theorem's conclusion |
| `summary_figures.py` | `fig_attempts.png` and `fig_composition.png` from the results files |
| `record_extracts.py` | small extracts for the record: run statistics, frozen-file hashes, redraw counts, 039-base context |
| `frozen_shas.json` | output of `record_extracts.py`: the frozen-file hashes recorded by C_s101 and D_s101; only `data/targets_L6-11.jsonl` differs |
| `composition/build.py` | selects 100 ANDI theorems per joined length and recovers their two parts; builds controls (a) premises grouped by part, (b) conjuncts in premise order, (c) original atom letters; every part and control proof is checked with nd_verify |
| `composition/run_eval.py` | evaluates the part and control prompts (k = 32) through the lock; records written lines per sample |
| `composition/failloc.py` | CPU: places the first error of each failed joined ANDI sample relative to the two parts |
| `composition/analyse.py` | writes `report/composition.md` and `composition/results.json` |
| `composition/conclusion.md`, `composition/conclusion_dc.md` | hand-written conclusions inserted into composition.md (sections Conclusion and 6) |
| `tests/test_ndcheck.py` | checker vs nd_verify agreement on stored and corrupted proofs, plus hand-made cases |
| `tests/test_launch.py` | arm-file creation (one-line diffs, refusal cases) |
| `tests/standin/rl_seed.py` | a stand-in seed used only by the launcher tests |

**Versions.** These are the final versions. The ones that changed during the experiment:
- `evaluate.py`, `sampler.py`, `ndcheck.py`: unchanged since 2026-09-24 20:43 UTC. Every evaluation in the record used
  them, with batch 512. `meta.json` in each evaluation directory records the batch.
- `common.py`, `build_testsets.py`: edited 2026-09-25 02:25 UTC. They add arm D's pool to the disjointness check and
  keep the `family` and `label_kind` fields. Neither change affects a draw.
- `launch.py`: this version was edited by the monitor on 2026-09-24 21:03 UTC to set two knobs. It prepared
  `rl_A-C.py` and launched arms A-C.
- Arm D was launched by hand (below).
- `analyse.py` and `composition/*`: final versions.
  - The `--dir`, `--roles` and `--labels` options were added for the D comparison; the defaults reproduce the first
    composition run.
  - Earlier report versions came from earlier `analyse.py` versions.

## Scratch locations (not in Git)

- **Harness and base checkpoint.**
  - `/scratch/cowifornia/nd-rl/ab3/frozen/`: the run-3 frozen harness (`harness.py`, compute meter `meter.py`,
    `lib/`, `gpulock.py`), `DEFAULT_FLOPS = 3.9e15`, and the pretrained checkpoint `pretrain.pt` (sha256
    `f4409e4f82d8…`).
  - `/scratch/cowifornia/nd-rl/ab3/frozen_comp/`: the same, with `data/targets_L6-11.jsonl` =
    `ab3/data_comp/targets_L6-11.jsonl` (arm D).
- **Test-set sources.**
  - `ab2/data/reserve.jsonl` (L6-7), `ab2/data/dev_L8-11.jsonl`, `ab2/sealed/sealed_L12-14.jsonl`;
  - `ab3/sealed/sealed_L15-16.jsonl`, `sealed_L17-18.jsonl` and `matched_L6-14.jsonl` (final versions of 01:10 UTC,
    2026-09-25; sha256 in the summary's `evidence/redraw_check.json`).
- **Test sets.** `ab3/lengthexp/testsets/` holds the final draw; `testsets_predraw/` holds the earlier draw used by
  composition sections 1-5.
- **Evaluations.**
  - `ab3/lengthexp/eval/<label>/`: per-theorem rows (`<set>_L<L>.jsonl`), unit summaries, `samples/*.jsonl.gz` and
    `meta.json`.
  - `eval_predraw/` holds the earlier-draw units.
- **Training runs.** `ab3/lengthexp/runs/<label>/` contains `final.pt`, `round_*.pt`, `metrics.jsonl`,
  `found_targets.jsonl`, `summary.json`, `source/rl.py` and the `<label>.log`, 9.8 GB in all. `final.pt` sha256
  prefixes are listed below; the full hashes are in the summary's `evidence/runs_summary.json`.

  | arm | seed 101 | seed 102 | seed 103 |
  |---|---|---|---|
  | A | `6d0205019ec6` | `020d96e4f4be` | `8525671d72d9` |
  | B | `70a0f06a28d4` | `2530c356952f` | `03329e788005` |
  | C | `9ec083a96666` | `f316a3d7a3c3` | `8ebe0f591744` |
  | D | `c95556a27ebb` | `7a985c2c1266` | `369a4b9664b3` |

- **Composition.** `ab3/lengthexp/composition/` holds the earlier draw; `composition/final/` holds the final draw,
  C vs D.
- **Reports.** `ab3/lengthexp/report/` holds `report.md`, `composition.md`, `results.json`, `fake_finish.*`,
  `runs_summary.json`, `redraw_check.json`, `base039_context.json` and the figures.

## Reproduce (from `/scratch/cowifornia/nd-rl/ab3/lengthexp`)

```
PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python
$PY build_testsets.py                                  # test sets + MANIFEST.json (CPU)
$PY launch.py prepare                                  # rl_A/B/C.py from ../seed/rl.py
$PY launch.py all                                      # 9 runs (A, B, C x seeds 101-103) through the lock, then evaluation
# arm D: frozen_comp = frozen with only data/targets_L6-11.jsonl replaced; rl_D.py = copy of rl_C.py
for s in 101 102 103; do AB_WORKER=lengthexp $PY ../frozen_comp/harness.py --rl rl_D.py --out runs/D_s$s --seed $s \
    --gpu auto --no-score --gpu-wait 43200 > runs/D_s$s.log 2>&1 & done
$PY watch_d.py                                         # evaluates each D run when it finishes
$PY evaluate.py --ckpt runs/<label>/final.pt --label <label>   # one checkpoint (cached units are skipped)
uv run --script analyse.py                             # report/report.md, figures, results.json
# composition diagnosis (earlier draw: all roles; final draw: renamed parts only, C vs D)
$PY composition/build.py && $PY composition/failloc.py
$PY composition/run_eval.py --ckpt <ckpt> --label <label>      # pretrain, A_s101, B_s101, C_s101
$PY composition/build.py --dir composition/final --lmax 14 --roles part_ren
$PY composition/run_eval.py --dir composition/final --ckpt runs/C_s101/final.pt --label C_s101   # and D_s101
$PY composition/failloc.py --dir composition/final --labels C_s101,C_s102,C_s103,D_s101,D_s102,D_s103
$PY composition/analyse.py
# record extras (CPU)
$PY fake_finish.py && $PY record_extracts.py && uv run --script summary_figures.py
# tests
$PY tests/test_ndcheck.py && $PY tests/test_launch.py
```

Running `composition/failloc.py` with no arguments on the final `eval/` does not reproduce the earlier-draw rows. For
those, point `--eval-root` at `eval_predraw/`, and keep `testsets_predraw/` in place of `testsets/`.
