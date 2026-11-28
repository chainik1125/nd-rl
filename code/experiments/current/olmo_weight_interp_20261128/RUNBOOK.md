---
written_on: 2026-11-28
written_by: agent:claude (orchestrator of this measurement, Fable 5.2)
---

# Runbook: weight interpolation between `dpo` and `s2300` on one shared pod

What is measured and how it is read: `PREREG.md` (with Amendments 1 and 2). Audit and replies: `discourse/audit.md`.
This file is the order of operations. W is this folder; `WI_DATA` is `/scratch/cowifornia/nd-rl/olmo_elicit/interp`.
Nothing here creates, stops or deletes a pod: the main thread does that.

## 1. Pod to request

| item | value | why |
|---|---|---|
| GPU | 1 x H200 (141 GB), secure cloud, $5.29 per hour | one 32B model in bf16 (64.5 GB) with room for 192 parallel draws; the pod type of metrics 2 |
| image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | the image the pinned stack (vLLM 0.11.0, torch 2.8.0+cu128) was installed on yesterday |
| minCudaVersion | 12.8 | torch 2.8.0+cu128 (accepted by the coordinator) |
| container disk | 50 GB | pip packages |
| volume | 500 GB at `/workspace` | see "Disk" below (accepted by the coordinator) |
| ports | 22/tcp only | ssh and rsync; no web port is used |
| environment | none | no token, no secret; weights and code come from public sources |

After creation I need: the pod's creation time in UTC (it starts the cost ledger), its public IP and ssh port. I add
`Host rp-wi` to `~/.ssh/config` on serrano (the driver only accepts aliases of the form `rp-<name>`).

Disk. `dpo` and `s2300` are 64.5 GB each (129 GB). Each merged checkpoint is another 64.5 GB. In the default
schedule the three writes run while the cells sample, so up to three merged checkpoints can exist at once:
129 + 3 x 64.5 = 322 GB. A merged checkpoint is deleted when its cell is complete and counted; one whose cell was cut
stays until the release step. `df` on the pod shows the whole storage cluster (350 TB free yesterday), so the write
step sums the sizes of all files under `/workspace`, whoever wrote them, and refuses a write that would pass the
500 GB (`WI_VOLUME_GB`) less a 20 GB margin.

## 2. The pod is shared: what this run touches

Another orchestrator's job uses the same GPU after this run. This run therefore keeps to the following.

| path on the pod | whose | what this run does |
|---|---|---|
| `/workspace/wi/` | this run only | code, inputs, `runs/` (per-draw files, counts, reports, logs), `meta/`, locks, the merged checkpoints in `models/interp_<tag>`, and the marker `GPU_FREE` |
| `/workspace/models/dpo` | shared | downloaded here by Dan's unedited fetch script (it has no other destination); verified; **left in place at the end** |
| `/workspace/models/s2300` | this run, in a shared folder | downloaded and verified; removed at the end, and only if this run downloaded it |
| `/workspace/models/tok`, `/workspace/nd-takehome`, `/workspace/nd-answeronly`, `/workspace/.hf`, pip packages | shared set-up | created by the unedited set-up scripts of 2026-11-25 and metrics 2 if absent, reused if present; never removed |
| anything else | not this run's | never read, written, stopped or removed |

- Every stop command signals only the process groups recorded by this run's own jobs. Nothing uses `pkill`.
- A cell does not start while GPU memory is in use by anything (it waits 2 minutes, then fails with exit 5 without
  touching the other process). The two runs do not share a lock, so the other job must not start GPU work before
  this run's `GPU_FREE` marker exists.
- Set-up installs pinned packages for the whole container. Two set-ups must not run at the same time.

## 3. What is copied to the pod

Exactly the 20 files of `pod/STAGE_LIST.txt` plus the 4 files of `$WI_DATA/inputs/` (24 files, about 180 KB):

- from W: `interp_write.py`, `cell_count.py`, `pod/wi_cell.sh`, `pod/wi_rm.sh`, `pod/wi_download.sh`,
  `pod/wi_release.sh`;
- from metrics 2, unedited: `sample_wrap.py`, `rank1/expo_write.py`, `rank1/common.py`, `tools/grade_job.sh`,
  `tools/gz_repair.py`, `tools/ledger_ok.py`, `pod/m2_job.sh`, `pod/m2_chain.sh`, `pod/m2_stop.sh`,
  `pod/m2_status.py`, `pod/m2_download.sh`, `pod/m2_verify_weights.py`, `pod/m2_setup.sh`;
- from the 2026-11-25 run, unedited: `pod/oe_setup.sh`;
- inputs: `pids_reach.txt` (the 48 prompt ids), `spec_reach4.txt` (4 draws per prompt), `prompt_sha256_reach.json`
  (sha256 of each prompt's token ids), `MANIFEST.json`.

Nothing sealed (textbook72, test_13_20, test_21_30, the 832 theorem sets) and no credential is in that list, and the
push refuses any file whose name looks like one. On the pod, set-up clones Dan's two pinned public checkouts, installs
pinned packages and fetches the `dpo` tokenizer files; the two checkpoints come from the public hub without a token.

## 4. Commands, in order (serrano, from W)

```bash
W=/scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code/experiments/current/olmo_weight_interp_20261128
export WI_DATA=/scratch/cowifornia/nd-rl/olmo_elicit/interp
PY="nice -n 5 uv run --offline --project /scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code --no-sync python"
cd $W

# 0. inputs (done 2026-11-28; repeat only if $WI_DATA/inputs is missing)
$PY make_inputs.py

# 1. push (md5 of all 24 files on both sides) and set-up; prints CODE_OK, VERSIONS, GRADER_OK, GPU_OK, SETUP_DONE
bash pod/wi_pod.sh setup rp-wi

# 2. which schedule fits this pod (read only): prints MODE lanes | MODE sequential | MODE none, and why
bash pod/wi_pod.sh podcheck rp-wi

# 3. the chains, with the pod's creation time (UTC) for the cost ledger
$PY make_chain.py --pod-start-utc 2026-11-28TXX:XX:XXZ --cap-usd 21          # add --mode sequential if step 2 says so
bash pod/wi_pod.sh chain rp-wi prep  $WI_DATA/chains/chain_prep.txt
bash pod/wi_pod.sh chain rp-wi cells $WI_DATA/chains/chain_cells.txt        # after prep, within 3 minutes
# (sequential: one file and one command instead: bash pod/wi_pod.sh chain rp-wi interp $WI_DATA/chains/chain_interp.txt)

# 4. watchers (log only, they stop nothing) and status
(timeout 4h bash pod/wi_watch_chain.sh rp-wi prep  $WI_DATA/chains/chain_prep.txt  &)
(timeout 4h bash pod/wi_watch_chain.sh rp-wi cells $WI_DATA/chains/chain_cells.txt &)
bash pod/wi_pod.sh status rp-wi          # one line per job and chain; tail $WI_DATA/logs/watch_rp-wi.log

# 5. when chain:cells is DONE or FAILED: fetch everything with md5 on both sides
bash pod/wi_pod.sh fetchall rp-wi        # must print FETCHALL_OK rp-wi: <n> files md5-equal

# 6. analysis and figure (CPU), on the fetched files
$PY analyze.py                            # $WI_DATA/analysis/results.{json,txt}, cells_manifest.json
$PY fig_interp.py                         # $WI_DATA/analysis/fig_interp.{pdf,svg,png}

# 7. release the shared pod (refused without step 5's FETCHALL_OK): prints the GPU_FREE line
bash pod/wi_pod.sh release rp-wi
```

Schedule (the coordinator chose the default; the fallback is mine to take and is logged in `log.md`):

- Default, `--mode lanes`: two chains side by side. `prep` uses CPU and disk only: `download_s2300`, `download_dpo`
  (each with the hub's sha256 for every shard), `preflight` (same tensor names, shapes and dtypes; config files),
  `write_a050`, `write_a075`, `write_a025` (each write followed by its checks; the last one also counts bf16 steps
  between the inputs and looks at the writer's arithmetic at alpha 1). `cells` uses the GPU: `cell_a100`,
  `cell_a050`, `rm_a050`, `cell_a075`, `rm_a075`, `cell_a025`, `rm_a025`. Each cell first waits for the `prep` job it
  needs (`preflight` for `a100`, `write_<tag>` otherwise) and gives up at once if that job failed or the `prep` chain
  ended without it.
- Fallback, `--mode sequential`: one chain (`chain_interp.txt`): write, sample, delete, next, one merged checkpoint
  at a time (194 GB at most). It is 35 to 50 minutes of pod time slower, because the GPU waits during each write and
  its check.
- Rule (`wi_pod.sh podcheck`, `cell_count.schedule_mode`): sequential if the container's memory limit is under 96 GB
  or cannot be read, or if fewer than 345 GB of the 500 GB volume are unused when the run starts. If fewer than
  215 GB are unused, neither fits and I report before starting anything. The 96 GB is a cautious guess, not a
  measurement: the write holds one shard of three checkpoints plus fp32 temporaries (about 35 GB) beside a loading
  engine.

The verdict names are Amendment 2's: "below proportional at the midpoint", "about proportional", "above
proportional", "unclear at this size". `results.txt` always prints the rule's verdict and the second line (against the
smooth reference), and names a switch only when the verdict is "below proportional at the midpoint" and the second
line is "sharper than the smooth reference".

## 5. Expected durations and cost

Speeds for sampling are yesterday's on the same pod type (192 to 416 requests: 1,170 to 1,650 tokens per second,
4,100 to 6,000 tokens per draw, model load 2.3 minutes). The write check has only been timed on serrano at 4 threads.

| step | expected | notes |
|---|---|---|
| set-up | 2 to 10 min | pip installs dominate; shorter if the pod already holds the stack |
| two downloads with sha256 | 3 to 5 min | 20 to 40 s each yesterday, plus hashing 129 GB |
| preflight | under 1 min | reads headers only |
| each cell | 16 min (10 to 25) | 2.3 min load, then about 1 million tokens; timeout 90 min |
| each write | 4 min | measured yesterday for the same writer |
| each write check | 8 to 15 min; the last one 15 to 25 min | not measured on the pod |
| fetch, analysis, release | 5 min | a few MB; the release deletes at most 258 GB |

Default schedule: the alpha 1.0 cell starts about 5 minutes after the chains and every later cell starts when the
previous one ends, or a few minutes later if its write check is still running. About 1.3 hours of chains
(1.1 to 1.9), about 1.6 pod-hours with set-up and fetch: about $8.5 (range $7 to $12) for this run's share of the pod.
Sequential mode: about 2.2 pod-hours, $11.5.

Cost gate. Before sampling, every cell computes hours since pod creation x $5.29 plus a full 90-minute cell ($7.94)
and refuses to start if that passes $21 (exit 9, the chain stops, the coordinator decides). So a cell starts only
within the first 2 hours 28 minutes of the pod's life. On the expected schedule the last cell starts at about
1 hour 15 minutes. The ledger counts the pod from its creation; if another job used the pod before this run, I pass
that spend as `--prior-usd` or the coordinator gives me a later start time.

Check-ins while it runs: every 30 minutes (`status` and the watch log), as for any job under an hour.

## 6. Markers and exit codes

Per job, under `/workspace/wi/runs/<job>/state/`: `RUNNING`, then `DONE` (exit 0) or `FAILED` (with `rc=<n>`).
Per chain, under `runs/_chains/<name>/state/`: `RUNNING`, then `DONE` or `FAILED <job> rc=<n>`; `rcs` lists every
job's exit code. A re-submitted chain skips jobs that are `DONE`. At the very end: `/workspace/wi/GPU_FREE`.

| job | exit | meaning | chain |
|---|---|---|---|
| `download_*` | 0 | commit, shard count, bytes and every sha256 match the hub; `out/origin.txt` says whether the folder existed before this run | goes on |
| | other | fetch or verification failed | stops |
| `preflight` | 0 | the two checkpoints have the same tensor names, shapes and dtypes; config differs only in listed fields | goes on |
| | 6 | they do not | stops |
| `write_<tag>` | 0 | written to `/workspace/wi/models/interp_<tag>`; all 707 tensors bit-equal to a fresh recomputation and inside the two invariants; exact check passed on the sample; marker `WI_WRITE_OK` written | goes on |
| | 4 / 5 / 6 | a tensor failed a check / disk budget / census or config | stops; the cell that needs it exits 21 |
| `cell_<tag>` | 0 | 192 rows sampled and graded, settings as registered (`out/count.json` says `complete`) | goes on |
| | 124 | the sampler hit its own 90-minute timeout: rows kept and graded, missing draws listed | goes on for a merged cell; stops for `a100` |
| | 9 | not started: the cost gate | stops |
| | 21 | not started: the weights it needs are not verified, or the job it waits for failed | stops |
| | 5 | not started: GPU memory was in use by another process | stops |
| | 31 / 32 | rows missing, extra or duplicated / a setting differs from the registered one | stops |
| | 2 | not started: rows of an earlier attempt exist (never resumed silently), or usage | stops |
| | other | the sampler's own exit code (for example an engine failure) | stops |
| `rm_<tag>` | 0 | `interp_<tag>` removed after its cell was `DONE` and counted complete (or already gone) | goes on |
| | 11 | kept: the cell is not complete | goes on |
| | 2 | refused: the folder does not carry the write marker | stops |

Files a cell leaves in `runs/cell_<tag>/out/`: `samples.jsonl.gz` (one row per finished draw), `graded.jsonl`
(strict checker verdict per row), `meta.jsonl` and `wrap.json` (the sampler's own record of its settings),
`count.json` (planned, present, missing and duplicated keys, settings checks), `cell_status.txt`, and
`samples_complete.jsonl.gz` only if the gzip was cut and repaired. A write leaves `runs/write_<tag>/out/check.json`
and `MANIFEST.json`.

## 7. If something goes wrong

- The alpha 1.0 cell is not complete: the chain has stopped and nothing can be read. I report; no re-run without the
  coordinator, because a re-run fills only the missing draws (PREREG "Cut cells").
- A merged cell was cut (exit 124): the chain went on, the merged checkpoint is still on the pod, and the analysis
  reports the cell as sampled and with every missing draw a fail and a pass. Resuming it is the coordinator's call
  (`WI_ALLOW_RESUME=1` exists for that and nothing sets it).
- A write failed: `runs/write_<tag>/out/check.json` lists the tensors and the reason. I report before changing
  anything.
- The cost gate skipped a cell: I report the spend so far and what is missing. A new chain file with a higher
  `--cap-usd` is the only change needed if the coordinator agrees; finished jobs are skipped.
- A cell exits 5 (GPU memory in use): something that is not this run's is on the GPU. I report and stop nothing.
- To stop this run: `bash pod/wi_pod.sh stop rp-wi --chain cells` (and `prep`). It signals this run's own process
  groups only. Draws in flight are lost and the cell counts as cut.
- A watcher line `STALL` or `OVERRUN` is a prompt to look, not an action. A cell that is waiting for its write, or
  whose remaining draws are all long, writes nothing for a while.

## 8. Fetch, md5, release

1. `wi_pod.sh fetchall rp-wi` copies every folder of `/workspace/wi/runs` (per-draw files, counts, write reports,
   logs, chain logs) and `/workspace/wi/meta` to `$WI_DATA/runs/` and `$WI_DATA/pod_meta_rp-wi/`. For every file it
   compares the md5 computed on the pod with the md5 of the fetched copy and writes the list to
   `$WI_DATA/fetch_rp-wi.md5`. It prints `FETCHALL_OK rp-wi: <n> files md5-equal` only if every file matches and
   nothing is still running (`FETCHALL_PARTIAL` or `FETCHALL_MISMATCH` otherwise). Weights are never fetched.
2. I run the analysis on the fetched files and check that `results.txt` lists four cells with their row counts.
3. `wi_pod.sh release rp-wi` (refused unless step 1 printed `FETCHALL_OK`) runs `pod/wi_release.sh` on the pod. It
   - refuses while any job or chain of this run is running;
   - removes the merged checkpoints still under `/workspace/wi/models` (those of cut cells);
   - removes `/workspace/models/s2300` only if `runs/download_s2300/out/origin.txt` says this run downloaded it and
     the folder is the pinned commit; otherwise it leaves it and prints why;
   - leaves `/workspace/models/dpo` (64.5 GB, commit `97604023...`), the tokenizer folder, Dan's checkouts and
     everything that is not this run's;
   - writes `/workspace/wi/GPU_FREE` (one line with the UTC time and the path of `dpo`) if no GPU memory is in use.
     If memory is in use it writes no marker, prints the processes and stops nothing.
   The report is printed, kept on the pod as `/workspace/wi/meta/release.txt` and on serrano as
   `$WI_DATA/release_rp-wi.txt`.
4. I report to the main thread: the `GPU_FREE` line, the path of `dpo`, and what was removed. The main thread decides
   when the pod itself is deleted.

## 9. Tests and checks (serrano, CPU)

```bash
cd $W && OMP_NUM_THREADS=1 nice -n 5 uv run --offline --project /scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code \
  --no-sync --with safetensors --with transformers python -m pytest --confcutdir=. -p no:cacheprovider -q
cd /scratch/cowifornia/nd-rl/worktrees/olmo-elicitation/code && UV_PROJECT_ENVIRONMENT=/scratch/cowifornia/nd-rl/worktrees/two-authors/code/.venv \
  uv run --offline ruff check experiments/current/olmo_weight_interp_20261128 && UV_PROJECT_ENVIRONMENT=/scratch/cowifornia/nd-rl/worktrees/two-authors/code/.venv \
  uv run --offline ruff format --check experiments/current/olmo_weight_interp_20261128
```

The tests cover: the alpha arithmetic on tiny bf16 checkpoints (alpha 1.0 equals `s2300` and alpha 0 equals `dpo`
bit for bit; 0.25, 0.5 and 0.75 against an independent exact reference); the chains run with stubs (order, markers,
skip on re-run, cut cell, failed yardstick, failed preflight or download, cost gate, deletion guards, stop); the
release step (only this run's weights go, `dpo` and other files stay, no marker while something runs or the GPU is in
use); the serrano driver through a stub `ssh` (push, schedule check, chains, watcher, status, fetch with md5,
analysis and figure on the fetched files, release); and the analysis on synthetic cells reaching each verdict and
each second line.

What the tests cannot cover: the real vLLM engine, the real checkpoints (no weights on serrano), set-up on the pod,
the container's memory limit, and the speed of the write check on the pod.
