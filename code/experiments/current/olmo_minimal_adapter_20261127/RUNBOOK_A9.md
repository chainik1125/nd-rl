---
written_on: 2026-11-28
---

# Runbook for Amendment 9: a smaller adapter and fewer examples, sampled and read

What runs: one training cell (the full-size adapter on 16 examples), one sampling batch of three cells (the rank-1
adapter on 64 examples, the full-size adapter on 16 and on 4 examples; 48 reach prompts × 8 draws each), and one
reader round (round 5) on serrano. The design, the reading rule and the limits are in `PREREG.md`, Amendment 9, with
the changes after the audit in 9(j). The audit and the replies are in `discourse/stage1_report_audit.md`.

Paths: `W` is this folder; `V2=/scratch/cowifornia/nd-rl/olmo_elicit/v2`. All times below are pod time counted from
the driver's first command.

## Who does what

- The coordinator creates the pod, adds its ssh alias, and deletes it. Nothing in this folder creates, stops or
  deletes a pod.
- The orchestrator runs the driver on serrano, watches it, runs the reader round, and reports.
- The pod is shared with the weight-interpolation job (`olmo_weight_interp_20261128`). One job on the GPU at a time,
  and never two set-ups at the same time (both install the same pinned packages for the whole container).

## Pod

| item | value |
|---|---|
| GPU | 1 × H200 (141 GB), secure cloud, $5.29 per hour (the other job's pod; nothing more is needed for this job) |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, as every pod of 27 Nov |
| container disk | 50 GB (pip packages) |
| volume at `/workspace` | this job adds about 2 GB under `/workspace/mini`; the shared files are listed below |
| ssh | an alias `rp-<name>` in `~/.ssh/config` on serrano; the alias must be new (no `V2/runs_rp-<name>` yet) |

Where things are on the pod:

| path | whose | what |
|---|---|---|
| `/workspace/mini/` | this job only | code, pools, plan, staged adapters, `runs/dpo/` (training, gate, samples, logs, state) |
| `/workspace/models/dpo/` | shared, read-only for us | the `dpo` weights in Dan's `fetch.sh` layout: commit 97604023, 14 shards, 64,467,127,296 bytes, with `.sha` and `.complete`. **One download serves both jobs**: whichever job starts first fetches them, the other finds `.complete` and skips. This job never deletes them (`DELETE_WEIGHTS` stays 0). |
| `/workspace/models/tok/` | shared | the tokenizer |
| `/workspace/nd-takehome/` | shared | Dan's code at 8e744a89 (with the worktree `/workspace/nd-answeronly` at 93d36de7) |
| `/workspace/.hf/` | shared | the Hugging Face cache |
| `/workspace/wi/` | the other job | not read or written by this job. The one exception: while a chain of this job runs, it holds a lock on `/workspace/wi/gpu.lock` if that file exists (opened for reading; never created or changed), so the other job refuses to start on the GPU. `A9_OTHER_LOCK=` (empty) turns this off. |

## Everything that is copied to the pod

Nothing sealed and no credential: no textbook72, test_13_20, test_21_30 or 832-theorem file, no key and no token.
The driver checks this twice with `a9.py scan` (once on the staged push, once on the staged batch) and stops if a
name or a token-like string is found. The models and Dan's repository are public and are fetched by the pod itself.

1. **Code and inputs** (`pod/mini_pod.sh push`, 94 files, 35 MB when staged on 28 Nov; md5-checked on both sides):
   - this folder's top-level `*.py`, `*.sh` and `*.md` except `test_*`, and `pod/*.sh`, `pod/*.py`;
   - `olmo_edl.py` and `olmo_template.py` from `olmo_elicitation_20261125/`;
   - `stage2/pids_L.txt` (the 48 reach prompt ids with 8 draws each) and `stage2/seeds_L.csv`;
   - `V2/pools/` (training and held-out answers of `dpo` and `s2300` on non-reach prompts: `dpo/{final,final25,own,unsel}`,
     `s0100`, `s0500`, `s1100`). Only `dpo/final` is used here; the rest is copied because the push copies the folder.
   - the plan `V2/plan_dpo_a9.json` (md5 c70e2838…).
2. **The staged batch** `V2/stage/batch_a9/` (`stage_cells.py`, md5-checked): for each sampled cell the adapter
   (`adapter_config.json`, `adapter_model.safetensors`), `gate.npz` and `cell.json`. Two full-size adapters of 537 MB
   each and the rank-1 adapter of 2.6 MB. The 16-example adapter is trained on the pod, fetched, and staged from the
   fetched copy.

## Before the pod exists (done in preparation, 28 Nov)

- `V2/plan_dpo_a9.json` is built (`python3 a9.py plan --out ...`; `--check` prints `PLAN_OK`).
- `python3 a9.py verify --ref` prints `VERIFY_OK`: the rank-1 and 4-example adapters and the reference are DONE, from
  our pool, selected by held-out loss, with the md5 recorded at the stage-1 fetch.
- Tests and ruff: see `log.md`.

## Commands, in order

Run from `W` on serrano. One command starts everything on the pod:

```bash
cd /scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code/experiments/current/olmo_minimal_adapter_20261127
V2=/scratch/cowifornia/nd-rl/olmo_elicit/v2
A=rp-<name>
nice -n 10 timeout 11400 bash pod/drive_a9.sh "$A" > "$V2/logs/drive_a9_$A.log" 2>&1 &
```

- `A9_T0=<epoch seconds>` if this job's pod time started before the driver (for example the pod was created for this
  job some minutes earlier). The money stop counts from it.
- A watchdog beside it, as for every unsupervised job: check the log at 30 minutes, 1 hour, 2 hours; `STALL` (a
  heartbeat older than 10 minutes), `EXIT` and `OVERRUN` lines are the driver's own alarms.
- Status at any time: `bash pod/mini_pod.sh status "$A" dpo`.

| step | what the driver does | expected | deadline | line in the log when it is done |
|---|---|---|---|---|
| 0 | checks on serrano: plan file, reused adapters' md5, no `stage/batch_a9`, new alias, pod has no `/workspace/mini`, GPU under 2,000 MiB in use, scan of what a push would copy | under 1 min | | `PLAN_OK`, `VERIFY_OK`, `SCAN_OK` |
| 1 | push and set-up (Dan's pinned stack; a pod the other job has set up is reused) | 3 to 6 min | 30 min | `SETUP_DONE`, then `setup done; ... so far` |
| 2 | budget decision before training | | | `decision before training: CELLS=...` |
| 3 | weights if absent, then training the 16-example adapter; fetch | 0.5 to 4 min, then 24 to 31 min; fetch 2 min | minute 46 of pod time: after it three cells no longer fit, so training is stopped and the two reused cells go on | `train_plan_dpo_a9: DONE`, `fetch dpo (try 1): FETCH_OK` |
| 4 | budget decision before sampling: three cells, or the rank-1 and 4-example cells, or none | | | `decision before sampling: cells=...` |
| 5 | staging, upload, LoRA gate, sampling, grading | 3 min, 3 min, 75 to 110 min | gate 15 min; sampling until minute 165 | `sampling started: 3 cells, seed 660009`, then `sample_batch_a9: DONE` |
| 6 | last fetch with md5 on both sides | about 3 min | | `final fetch OK`, then `A9 DONE on <alias>` |

Total: about 1 h 55 min to 2 h 40 min, $10.0 to $14.3. Hard stop $15 (170 minutes).

How the driver ends, and what each ending means:

| last line | exit | meaning | what to do |
|---|---|---|---|
| `A9 DONE on <alias>` | 0 | everything sampled, graded and fetched | "After the fetch" below. Tell the coordinator this job no longer needs the pod. |
| `A9 STOPPED rc=8` | 8 | nothing fitted inside the money stop; nothing was sampled; all fetched | report |
| `A9 STOPPED rc=7`, "the money stop arrived during sampling" | 7 | the driver stopped this job's chain at minute 165 and fetched what exists | "If sampling was cut" below, then report |
| `A9 STOPPED rc=4` | 4 | the GPU was in use by something else when a step of ours was about to start; what exists is fetched | tell the coordinator. If it happened at step 0, run the driver again when the GPU is free. If it happened later, see "Resuming by hand". |
| `A9 STOPPED rc=3` | 3 | an input check failed on serrano (plan, adapter md5, an existing batch, a forbidden name) | fix the input; nothing ran on the pod |
| `EXIT rc=... NEEDS A DECISION` | 2, 5, 6, 7 | a step failed, a chain was refused on the pod, or a fetch is not `FETCH_OK` | read the lines above it; fetch by hand first (`bash pod/mini_pod.sh fetch "$A" dpo`); the coordinator decides about the pod |

The money stop, in the driver's own terms: it decides before training and again before the engine starts whether the
planned batch ends inside $15 (`a9.py decide`; the decisions are saved as `V2/a9/decide_{train,sample}_<alias>.json`).
It drops whole cells, in the pre-registered order: the 16-example cell first, then nothing is sampled. It does not use
a deadline to cut draws in a healthy run. The pod-side caps (`TO_SAMPLE`) are 20 minutes later than the driver's own
stop, so only the driver's stop can act.

## Resuming by hand (only with the coordinator's go)

The driver has no resume mode: it refuses an alias it has used. If it stopped after training (GPU in use before
sampling, a failed sample stage, a lost connection), the batch can be started again by hand. The chain skips stages
that are DONE, and the sampler goes on from its own files; draws taken after a resume come from a newly seeded
engine, which is a deviation to report.

```bash
# only if V2/stage/batch_a9/cells_manifest.json lists the cells that were decided (V2/a9/decide_sample_<alias>.json)
GPU_FREE_MIB=2000 ALSO_LOCK=/workspace/wi/gpu.lock TO_GATE=900 TO_SAMPLE=<seconds left before minute 165, plus 1200> \
  bash pod/mini_pod.sh sample "$A" dpo "$V2/stage/batch_a9" --seed 660009
bash pod/mini_pod.sh status "$A" dpo      # until sample_batch_a9 reads DONE
bash pod/mini_pod.sh stop "$A" dpo        # at minute 165 if it is still running: the money stop is yours to keep now
bash pod/mini_pod.sh fetch "$A" dpo       # must read FETCH_OK
```

## Fetch and md5

- `pod/mini_pod.sh fetch <alias> dpo` copies `/workspace/mini/runs/dpo/` to `V2/runs_<alias>/dpo/`, computes md5 of
  every file on the pod and on serrano, and prints one verdict: `FETCH_OK` (all equal, no chain running),
  `FETCH_PARTIAL` (equal, but a chain is still running) or `FETCH_MISMATCH`. The two md5 lists stay in
  `V2/pod_stage/<alias>.fetch.dpo.{pod,lab}.md5`.
- The driver fetches twice: after training (the new adapter is staged from this copy) and after sampling. Each must
  read `FETCH_OK`; it tries twice.
- The pod may be given up by this job only after the last fetch reads `FETCH_OK`. The driver says so in its last line.
- Record in `log.md`: the md5 of the new adapter's `adapter_model.safetensors`, and of each cell's `samples.jsonl.gz`
  and `graded.jsonl`.

## After the fetch (serrano, CPU only, at most 4 cores, `nice -n 5`)

```bash
PYX=/scratch/cowifornia/venvs/olmo-edl/bin/python
B=$V2/runs_$A/dpo/sample/batch_a9          # one directory per sampled cell
R5=$V2/handcheck/round5; RD=$V2/hc5_read   # the key stays in R5; readers see only RD
nice -n 5 $PYX handcheck5.py prep --out $R5 --items-dir $RD --batch-dir $B
```

1. Look at the training record of the new cell first (`V2/runs_<alias>/dpo/train/cells/dpo__final__A16__n16/cell.json`):
   the selected lr and step, the held-out loss, and the flags `did_not_train` and `unconverged`. Record them.
2. `prep` prints, per cell, draws, strict passes, screened passes and items, and writes the batches. It builds once
   and refuses to rebuild. Nobody opens an answer of the new cells before the readers have finished.
   - It also adds the 100 drift items and every round-4 item of the final model (258) and of the one-pass adapter
     (149). `--no-reread` leaves the last two out (the first design). The coordinator kept the re-read (28 Nov).
   - It also adds every strict pass of the four weight-interpolation cells (coordinator's note, PREREG 9(k)): files
     under `/scratch/cowifornia/nd-rl/olmo_elicit/interp/`, read only, md5-checked against their own list.
     `--no-interp` leaves them out.
   - Expect about 1,250 to 1,400 items: about 80 to 88 batches for each of the four reads.
3. Dispatch, as in round 4 (PREREG "Round-4 hand-check protocol", item 7): one dispatcher at a time; fresh Sonnet
   subagents at ordinary effort (`subagent_type: general-purpose`, `model: sonnet`, no effort setting), in waves of
   at most 8; each is given only its batch file `RD/batches/<tag>.prompt.md` and its output path
   `RD/reads/<tag>.json`, nothing about cells, rounds or this experiment. The reader prompt is round 4's, word for
   word: "Read the file <batch file> and do what it says. Write the JSON to <output path>. Do nothing else: open no
   file other than that batch file and the item files it lists. Reply with one line." Order: primary read 1 (`a..`), primary read 2 (`b..`), task read 1 (`ta..`), task read 2 (`tb..`).
   - After every wave: `$PYX handcheck5.py status --out $R5 --items-dir $RD --tags <the wave's tags> --quarantine`.
     A file that fails validation is moved to `R5/failed_reads/` and its batch is dispatched once more. A second
     failure is reported; its items end as "unread".
4. Third reads on splits: `$PYX handcheck5.py third --out $R5 --items-dir $RD --kind primary` and `--kind task`; each
   prints the new batch tags (`c..`, `tc..`). Dispatch them the same way and validate.
5. Copy `RD` into `R5` (`cp -a $RD/. $R5/`), then
   `nice -n 5 $PYX handcheck5.py table --out $R5 --items-dir $RD` (10,000 bootstrap reps, seed 660054).
   It writes `R5/round5_summary.json` and `R5/labels.csv` and prints the cells, the differences against the
   reference with their readings, the shares of the final model's rate, and the drift results.
6. `$PYX handcheck5.py spot --out $R5 --items-dir $RD` writes 40 items with their labels to `R5/spot.md` for the
   dispatcher's own look.
7. Figure: `MPLBACKEND=Agg $PYX fig_a9.py` writes
   `experiment-summaries/2026-11-27-olmo-minimal-adapter/figures/a9_size_examples.{pdf,svg,png}`.
8. Write the result into PREREG (a dated section under Amendment 9), `log.md` and the summary README, with the limits
   of 9(f) and 9(j) beside every sentence. Percentages with intervals; "holds" only as "holds (within 12 points of
   the reference)"; "below the reference" whenever `interval_below_zero` is true.

## If sampling was cut

The pod grades only after the sampler has ended, so a batch that was stopped comes back with `samples.jsonl.gz` files
and no `graded.jsonl`. `handcheck5.py prep` refuses such a batch. Then:

```bash
S=$V2/a9/batch_a9_salvaged_$A
nice -n 5 $PYX handcheck5.py salvage --batch-dir $B --dest $S     # whole requests only; the fetched batch is not changed
cd /scratch/cowifornia/nd-rl/olmo_elicit/ext/nd-takehome           # Dan's code at 8e744a89
export NLTK_DATA=/scratch/cowifornia/nd-rl/olmo_elicit/venv_grade/nltk_data
for d in $S/*/; do
  [ -f "$d/graded.jsonl" ] || nice -n 5 /scratch/cowifornia/nd-rl/olmo_elicit/venv_grade/bin/python \
    olmo_ifomega/grade.py "$d/samples.jsonl.gz" olmo_ifomega/problems.jsonl "$d/graded.jsonl"
done
cd - && nice -n 5 $PYX handcheck5.py prep --out $R5 --items-dir $RD --batch-dir $S
```

- This local grading reproduces the stored grading of the reference (384 rows, 166 passes).
- `table` then reports each cut cell three ways, as 9(h) says: as sampled; every unfinished draw a fail; every
  unfinished draw an accepted pass. A reading that changes under either bound is reported as not robust to the cut
  (`robust_to_the_cut` in the summary). The figure draws a cut cell hollow, with the bounds as its whisker.
- A cell with no finished request is reported as not sampled.

## The optional fourth cell

Not run: the coordinator decided against it on 28 Nov. Only if the coordinator approves it and raises the stop: `REF2=1 A9_CAP_USD=18.5 bash pod/drive_a9.sh ...`
samples the reference adapter again in the same engine (about 35 minutes, $3.1). `handcheck5.py` then uses it as the
primary reference and reports the stored one beside it.

## Checks that were not possible on CPU

- vLLM, the gate, the sampler and the trainer are not exercised by the tests. They are the code of the 27 Nov runs,
  unchanged apart from two opt-in guards in `pod/mini_chain.sh`.
- The driver was run only against stand-ins for `ssh`, `mini_pod.sh` and the clock (`test_a9.py`).
- The lock on `/workspace/wi/gpu.lock` was tested with a local file, not with the other job running.
