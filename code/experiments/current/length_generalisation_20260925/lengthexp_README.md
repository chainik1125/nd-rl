---
written_on: 2026-09-24
---
# Length generalisation of RL-trained proofs (ab3/lengthexp)

A monitor-run experiment. No hill-climbing agent reads anything here.

## Question

The prover is a 4-layer, d=256 GPT pretrained on natural-deduction proofs of at most 6 lines. RL (plain
expert iteration against the verifier) on longer proofs makes it better at longer proofs. The question is
how. Does training on longer proofs make the model more general, so that it copes better with any extra
length? Or does it only move the edge of competence, so that the model is as bad 3 lines past its training
length as the pretrained model is 3 lines past 6?

## Design

- **Arms.** The arms differ only in the training max length m (7, 9 or 11), applied twice: RL targets have true
  length (`L_true`) 6..m, and the fine-tune mix keeps only found proofs of at most m written lines (longer
  verified proofs are dropped whole, never edited; they still count as solved in the run's bookkeeping):
  - pretrain only: `frozen/pretrain.pt`, no RL, training max 6.
  - A: m = 7 (targets of length 6-7, proofs of <= 7 lines). B: m = 9. C: m = 11.
  Whether the arms use the proof cap is the PI's call (pending): `prepare --no-proof-cap` makes one-knob arms
  (targets only, as before).
- **Runs.** Seeds 101, 102 and 103 for each of A, B and C, so 9 runs.
- **Recipe and budget.** All runs use the same recipe and the same compute budget: the run-3 seed recipe
  `ab3/seed/rl.py`, copied with only `CONFIG['max_len_train']` and `CONFIG['max_proof_lines_train']` changed,
  both to m (a two-line diff, checked by `prepare` and recorded in `arms.json`), under the frozen harness's
  `DEFAULT_FLOPS`. The seed logs, per round, the sampling window by true length (`window_by_len`,
  `samples_by_len`, `accepted_by_len`, `solved_by_len_round`, `solved_cum_all_by_len` in the metrics.jsonl
  `sample` records and in `rl_stats.per_round`) and the proofs left out by the cap (`proofs_over_line_cap`).
- **How runs are launched.** Each run goes through `frozen/harness.py --gpu auto --no-score`. The GPU lock is
  shared with everyone else on the 5 GPUs. The harness's own dev and frontier scoring is off, because the
  evaluation below replaces it.
- **Test sets.** 300 theorems per length (fewer where the source has fewer). Every theorem is disjoint by
  premise-order-invariant class (`pkey`) from the RL pool, the pretraining data and the other test lengths:

  | length | source | notes |
  |---|---|---|
  | 6-7 | `ab2/data/reserve.jsonl` | only records built fresh for run 2 (the old ladder records include run 1's RL targets); 180 generator + 120 textbook each, like dev |
  | 8-11 | `ab2/data/dev_L8-11.jsonl` | all of it (180 + 120 per length) |
  | 12-14 | `ab2/sealed/sealed_L12-14.jsonl` | subsampled: L12 300, L13 300 (102 textbook), L14 all 158 (114 textbook, one schema) |
  | 15-18 | `ab3/sealed/sealed_L15-16.jsonl`, `sealed_L17-18.jsonl` | being built by the data agent; added when they appear |
  | control | `ab3/sealed/matched_L6-14.jsonl` | same construction as the long sets at short lengths; evaluated separately when present |

- **Evaluation.** Every checkpoint is evaluated at every length: k = 32 samples per theorem at temperature
  0.8 with max_new 1024.
  - **Why max_new 1024.** Correct proofs use at most ~37 tokens per line (the minimal proofs at most 43),
    so 18 lines need at most ~780 tokens.
  - **Verdict.** Each sample is judged by `nd_verify` against the prompted sequent.
  - **Where it went wrong.** An incremental line checker finds the first rejected line of each failed
    sample, or labels it early QED, out of tokens, or past the N64 line-index vocabulary. This is run 2's
    `LineChecker` with reason codes. It agreed with `nd_verify` on 8,427 stored proofs, on 50,562 corrupted
    ones and on every evaluated sample so far.
  - **Budget.** Evaluation runs outside the compute meter.

## How to read the main plot (`report/fig_main.png`)

x is d = test length minus the arm's training max (6 for pretrain, 7, 9 and 11 for A, B and C). y is pass@1,
raw in the left panel and as log-odds in the middle panel. There is one curve per arm, and the bars show the
range over seeds.

- **Curves lie on top of each other.** Each arm falls off the same way once past its own training
  length. Longer training only moved the edge.
- **Curves get flatter for arms with a larger training max.** Each extra line costs those arms less. They
  generalise more.

The report fits a logistic in d for each run over d >= 1 and gives the slope: the drop in log-odds per extra
line. It gives a seed-level 95% t-interval for each arm, and tests whether the slopes differ between arms:
- a trend of slope against training max (exact permutation over the 9 runs);
- an omnibus F test;
- pairwise Welch and permutation tests.

The same fit is also done over the common window L >= 12, where every arm is past its training max, and on
generator theorems only (see caveats). The right panel shows absolute length, which answers the practical
question of which arm is best at 15-18 lines.

## Other sections of `report/report.md`

1. pass@1, pass@4 and pass@32 per arm and length, as the mean and range over seeds.
2. The main plot, the slopes and the tests.
3. The lengths of correct proofs against `L_true`. If a correct proof is shorter than `L_true`, the label is
   too high for that theorem: the minlen search is restricted to a formula closure, which would inflate
   results at long lengths. This section also reports the median excess length.
4. Where failed samples go wrong, as histograms of the failure line per arm and test length. It also shows the
   per-line hazard, P(fail at line p | lines 1..p-1 valid). A spike at each arm's own training max would
   point to a length-specific skill. A flat hazard means errors are spread evenly. The section also has the
   first-line-index distribution, because proofs that start high run out of N1..N64.
5. Training side, from each run's `metrics.jsonl`, `found_targets.jsonl` and `summary.json`: the cumulative
   solve rate by `L_true` over rounds, rounds and samples, compute used, and the held-out check. It also
   gives the written lengths of the proofs the run actually trained on.
6. The matched-construction control, when present: pass@1 on the matched set against the standard sets at
   the same lengths.

## Commands

Run everything from `/scratch/cowifornia/nd-rl/ab3/lengthexp` with
`PY=/home/cowifornia/code/research/nd-rl/code/.venv/bin/python`.

```
$PY build_testsets.py              # (re)build testsets/ + MANIFEST.json; re-run when the L15-18 or matched files appear
$PY launch.py prepare              # rl_A.py / rl_B.py / rl_C.py from ../seed/rl.py (only max_len_train and
                                   # max_proof_lines_train differ, both = m; arms.json); --no-proof-cap: targets only
$PY launch.py smoke --arm A        # harness --smoke --no-score on one arm file (1e12 FLOPs, ~10 s)
$PY launch.py train --dry-run      # print the 9 harness commands
nohup $PY launch.py all > logs/all.out 2>&1 &   # prepare, 9 runs queued on the lock, each run evaluated when it
                                   # finishes (pretrain too), then the analysis
$PY launch.py status               # runs, evaluation units, GPU locks
$PY launch.py eval                 # (re)evaluate pretrain + every finished run; cached units are skipped
$PY launch.py analyse              # = uv run --script analyse.py  -> report/report.md, report/*.png, report/results.json
```

Single pieces:
```
$PY evaluate.py --ckpt runs/A_s101/final.pt --label A_s101          # through the lock (--gpu auto)
$PY evaluate.py --ckpt ../frozen/pretrain.pt --label pretrain --limit 20   # smoke slice -> eval_smoke/
uv run --script analyse.py --eval-root eval_smoke --out report_smoke --train-max n092=11 \
    --run-dir n092_s101=/scratch/cowifornia/nd-rl/ab2/monitor/samecard/runs/n092_s101
$PY tests/test_ndcheck.py          # checker vs nd_verify agreement
$PY tests/test_launch.py           # arm-file creation
```

When the L15-18 files appear: run `build_testsets.py`, then `launch.py eval`. Only the new lengths are
sampled, because the existing lengths keep the same theorems and their cached results are reused. Then run
`launch.py analyse`.

`train` and `all` refuse to start while the harness's `DEFAULT_FLOPS` is None. They skip runs whose
`summary.json` says ok, and they move a failed run's directory aside before re-running it.

## Layout

| path | contents |
|---|---|
| `common.py` | constants (arms, seeds, lengths, k, T, max_new, sources), pkey, pass@k |
| `build_testsets.py`, `testsets/` | the test sets and `MANIFEST.json` (counts, source hashes, disjointness checks) |
| `ndcheck.py` | per-sample verdict + first-error classification (incremental checker) |
| `sampler.py` | compacting temperature sampler (same distribution as `lib.sample.generate`) |
| `evaluate.py`, `eval/<label>/` | per-theorem rows, per-unit summaries, all sampled proofs (`samples/*.jsonl.gz`) |
| `launch.py`, `rl_{A,B,C}.py`, `arms.json`, `runs/<label>/` | arm files and training runs (harness output) |
| `analyse.py`, `report/` | report and figures |
| `tests/` | checker and launcher tests; `tests/standin/rl_seed.py` is a stand-in seed for tests only |
| `eval_smoke/`, `report_smoke/` | test evaluations (pretrain slice, run-2 n092 slices); not part of the experiment |

## Arm D: composition-augmented RL pool (added 2026-09-25, PI-approved)

**What D is.** Arm D is arm C exactly, except for its RL target pool:
- Same as C: the recipe (`rl_D.py` is a byte copy of `rl_C.py`: `max_len_train=11`, `max_proof_lines_train=11`), the
  compute budget (`DEFAULT_FLOPS` 3.9e15) and the seeds 101-103.
- Different: the pool is `ab3/data_comp/targets_L6-11.jsonl`. It holds run 2's 15,738 targets plus 6,744 constructed
  ANDI/chain targets at L_true 6-11, 30% of each length bin; see `data_comp/POOLS_COMP.md`.

**How the pool is swapped in.** It works the way `frozen_039` did. `ab3/frozen_comp` is a copy of `ab3/frozen` in which
only `data/targets_L6-11.jsonl` differs (checked file by file with sha256). Its `data/SHA256SUMS` was removed so that the
harness recomputes the data hashes.

**Commands.** The runs were launched with:
```
AB_WORKER=lengthexp python ../frozen_comp/harness.py --rl rl_D.py --out runs/D_s10x --seed 10x --gpu auto --no-score
```
`watch_d.py` then queues each finished run's evaluation, and the composition part evaluation for D_s101
(`composition/final`).

**How D is reported.**
- D is not part of the A/B/C length-trend tests.
- `report.md` section 8 compares D with C.
- `composition.md` section 6 compares D_s101 with C_s101 on the final draw.

**Final test draw.** The data agent finished the sealed files at 01:10 UTC, and `build_testsets.py` was then re-run.
- *What changed.* The L15-18 draws changed, and so did the matched draw: the matched file keeps 3,599 records, but only
  1,707 of the 2,700 matched theorems drawn before survive in it. The standard L6-14 draws are unchanged.
- *New checks.* The builder now also checks disjointness against arm D's pool, and none was found. The drawn test theorems
  share no reserve part with D's constructed targets (0 of 6,211 parts).
- *Archive and re-evaluation.* The earlier draws and their evaluation units are archived in `testsets_predraw/` and
  `eval_predraw/`. The changed units (matched L6-14 and std L15-18) were re-evaluated for pretrain and A/B/C × 3.
- *Stale units.* `analyse.py` leaves out any unit whose theorems differ from the current `testsets/` and lists it.

## Caveats

The report quantifies each of these where it can.

- **Without the proof cap, the training max in lines is not `max_len_train`.** Expert iteration trains on the
  model's own verified proofs, and those are often longer than `L_true`. Run 2's n092 had targets of `L_true` <= 11
  but found (and trained on) proofs of up to 15-17 lines, and 26-28% of its found proofs were longer than 11 lines.
  `max_proof_lines_train` = m closes that gap (it drops those proofs from the fine-tune mix, so it also removes
  some training data from each arm; `proofs_over_line_cap` counts them per round). Section 5 reports the lengths
  actually trained on for every run; with `--no-proof-cap` arms, d should also be read against those.
- **The test mix changes with length.**
  - Textbook share: 40% at L6-12, 34% at L13, 72% at L14. At L14 all the textbook theorems come from one
    schema (demorgan_nand_to_or), and there are only 44 generator theorems. Run 2 found no schema
    instances above 14.
  - Construction: the L15-18 sets come from a different generator setting. The matched control addresses
    this if it arrives.
  - What the report does about it: it reports generator-only fits alongside the full ones.
- **The pretrain slope cannot be estimated.** Pretrain solves 0 of 9,600 samples at every length from 8 to
  14 (pass@1 0.14 at L6 and 0.011 at L7). Only one length with d >= 1 has any success, so its fit over
  d >= 1 has no finite slope. It serves as an anchor, not a curve.
- **Slopes over different ranges.** Each arm's own-window fit covers a different range of d (A: L8-18, C:
  L12-18). A curved log-odds profile would then produce slope differences by itself. The common window
  L >= 12 avoids this.
- **Three seeds give little power.** An exact pairwise permutation test cannot go below p = 0.10, and the
  95% t-interval uses t = 4.30. The trend test over all 9 runs (1,680 relabelings) is the sharpest test
  between arms. Within a run, the theorem-bootstrap SE of a slope was ~0.2-0.3 on 100-theorem slices. All
  arms see the same theorems, so arm comparisons are paired.
- **The arms differ in more than length.**
  - Target counts: arm A's pool (L6-7) is ~3,500 targets and is used whole. B and C draw 6,000 stratified
    by length. So the arms have different numbers of targets per length.
  - Compute: under a fixed FLOP budget, longer targets cost more per sample, so C gets fewer rounds or
    samples. Section 5 shows these counts.
- **Line indices.** The `abs` tokenizer has only N1..N64.
  - Pretrain starts its proofs anywhere in N1..N59, and 21% of its samples could not fit 18 lines.
  - The seed caps the start (`index_room=24`), so RL-trained models have room for 24 lines.
  - Failures past N64 are counted separately.
- **Line count and token position are confounded.** Longer proofs also sit at token positions (RoPE) not
  seen in training. This design cannot separate generalisation in line count from generalisation in
  position.
