---
written_on: 2026-10-13
written_by: agent:claude
---

# GRPO v2 launchers, reconstructed after serrano's crash

serrano crashed on 2026-10-11 at about 11:48Z (4:48 am PDT) and stayed down. Its local `/scratch` held the run data
and these launcher scripts, which were never in git. They are rebuilt here from the session transcripts
(`~/.claude/projects/-home-cowifornia-code-research-nd-rl/99f17bd8-…jsonl` and its `subagents/`) and the old NFS
backup (`/home/cowifornia/backups/nd-rl-think/`), for the restart on ynez. The runtime paths are serrano's, unchanged.

## Files and install paths

| file here | installed at (runtime path) | bytes | md5 | serrano evidence |
|---|---|---|---|---|
| `line.sh` | `/scratch/cowifornia/nd-rl/think/runs/grpo_v2/line.sh` | 11,227 | `daaea130…eb27a` | byte exact |
| `queue.sh` | `/scratch/cowifornia/nd-rl/think/runs/grpo_v2/queue.sh` | 851 | `8acdfe27…008c` | byte exact (commit `718188f`); ynez adds one line, below |
| `lib.sh` | `/scratch/cowifornia/nd-rl/think/runs/followup_20261007/lib.sh` | 2,245 | `6f2aa471…ca95` | byte exact |
| `watchdog.py` | `/scratch/cowifornia/nd-rl/think/runs/watchdog.py` | 3,786 | `2442d2b5…ab8a` | byte exact |
| `backup.py` | `/scratch/cowifornia/nd-rl/think/runs/backup.py` | | | new on ynez (below) |
| `dry/` | `/scratch/cowifornia/nd-rl/think/dry/` | | | the dry-run harness (below) |

`grpo_v2/preflight.py`, which `line.sh` runs, was already in git (`../preflight.py`).

### How each file was rebuilt

- **`line.sh`** (the final version, with the think-ban flags `--ban_restore_check --ban_stop 0.005`, `ARM_b` and the
  `ARM_h` arms at lr 7.5e-5; moved over `line.sh` at 2026-10-11 09:11Z). Two independent routes give the same bytes:
  1. the last full `cat` of `runs/grpo_v2/line.sh` on serrano, 2026-10-11 10:57:50Z (subagent `ab8873244e7e40e6`,
     50 minutes before the crash; no command touched the file after it). Its length, 11,227 bytes, is the size
     `ls -la` printed for `line.sh` in the same output and at the 09:10:57Z swap;
  2. the `cat` of the previous `line.sh` at 08:28:33Z (9,579 bytes, md5 `f6d9b46d…`, the md5 serrano printed for it
     at 08:45Z), plus the six successful `Edit`s subagent `ad6a202caf48dd21` made to `line.sh.new` from 08:58:47Z to
     08:59:20Z, in order.

  It also agrees with the 10:50:45Z `diff` against the harness's `line.sh.new.dry` (only the `source` and `PY=` lines
  differ) and with the live `grpo_think_h_b_s1_L3` process (see the dry run below).
- **`queue.sh`**: the heredoc that created it at 2026-10-10 03:19:22Z; serrano listed it at 851 bytes with that
  mtime until the end, printed its md5 (`8acdfe27…`) at 2026-10-11 05:03Z, and `cat`s at 10-10 11:36Z, 16:19Z,
  10-11 04:58Z and 10:40Z show the same text.
- **`lib.sh`**: the NFS backup copy (`/home/cowifornia/backups/nd-rl-think/followup_20261007/lib.sh`, mtime
  2026-10-08 04:50Z, the time of its last edit, which added `evdone`). A full `cat` on serrano at 2026-10-11 07:05Z
  and a `head -60` (the whole 28 lines) at 11:25Z, 23 minutes before the crash, match it byte for byte (the tool
  strips the final newline); no command wrote it after 10-08 04:50Z. Mode 755 as in the backup.
- **`watchdog.py`**: the `Write` of 2026-10-06 19:03Z equals the NFS backup (`watchdog.py`, Oct 6). No later command
  wrote it; reads on 10-08 09:01Z (all 103 lines), 10-08 21:38Z (lines 1-80) and 10-09 21:11Z (1-30) match.

Not rebuilt: `runs/restart3_grpofix/launch.sh`, which only `lib.sh`'s `train()` calls; the v2 line and queue never
call `train()` (or `ev()`). A backup of it from 2026-10-06 is at
`/home/cowifornia/backups/nd-rl-think/restart3_grpofix/launch.sh`.

## Inputs the line reads under `runs/` (rebuilt separately)

`stage1/{S1g3,T0S}_s<seed>/model.pt`; `data/{S1g3,T0S}.pkl` (EI); `pools/curriculum/L{2,3}/pool.jsonl` and
`test_1k.jsonl`; `pools/curriculum_3p/L{2,3}_3p.jsonl` (`_vs3`); `pools/canary_long.jsonl`;
`pools/charles_1002/{candidate_v0,candidate_v1,batch3,hand_proved_14}.jsonl`;
`eval/matched/sets/{t1320,t2130}_n1000.jsonl`; and the preflight's sealed sets `pools/test_13_20/test_13_20.jsonl`,
`pools/test_21_30/test_21_30.jsonl`, `restart2_hard12/hard12_test_1k.jsonl`, `pools/hard712/{test_1k,dev_500}.jsonl`.
The preflight only prints `(skip …: not built)` for a missing sealed set and still passes (the `_vs3` extra-targets
check fails on it), so check its log after the first line on ynez. Outside `runs/`: textbook72 in the `nd-takehome`
clone (`/scratch/cowifornia/nd-rl/think/nd-takehome/data/eval_only/textbook72/textbook_{dev,train}.jsonl`), the
code worktree `/scratch/cowifornia/nd-rl/think/nd-rl-v2` on `leon/grpo-v2` (line.sh refuses any other branch) and
the env `/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv`. `lib.sh` also needs `runs/logs` and `runs/tmp` (`TMPDIR`).

## Dry-run harness (`dry/`, runs from `/scratch/cowifornia/nd-rl/think/dry/`)

Rebuilt from serrano's `scratchpad/dry{,_b}/make.sh` and `run.sh` (printed in the transcripts on 2026-10-11). It
runs the real `line.sh` and `queue.sh` in a sandbox with python, `timeout`, `nvidia-smi`, `sleep`, `rm`, `python3`
and `git` stubbed to record every call (argv, cwd, `CUDA_VISIBLE_DEVICES`, redirections, the environment).

    cd /scratch/cowifornia/nd-rl/think/dry
    bash make.sh [DIR]      # DIR with line.sh, lib.sh, queue.sh; default the installed copies
    bash all.sh             # every arm x seeds 0-2 x normal/SMOKE=1, refusals, a busy GPU, queue.sh (~4 min)
    python3 check.py [--expect-hook]
    python3 extract_evidence.py   # regenerates evidence.json from the transcripts

`check.py` compares the records with a model of the recipe written from the README and with the transcript
evidence in `evidence.json`: the ps / `/proc` capture of the live `grpo_think_h_b_s1_L3` process (2026-10-11 ~10:42Z),
serrano's harness records (full commands and per-call records of three runs, per-arm call counts), every launcher
log line of a v2 arm (`launched …`, stage exits, `read …`, `grpo_v2 line done …`, queue lines) and every ps line of a
v2 process.

## ynez (2026-10-13)

Paths are serrano's. Changes, all of them:

1. `queue.sh` runs a backup pass after each arm:
   `python3 $R/backup.py >> $L/backup.log 2>&1 || true` (one line; `line.sh`, `lib.sh` and `watchdog.py` are
   serrano's bytes).
2. `backup.py` is new (below).
3. The install creates `runs/logs` and `runs/tmp` (`lib.sh` exports `TMPDIR=$R/tmp`; the queues write their logs to
   `runs/logs`).
4. `leon/grpo-v2-sched` (2026-10-13, not yet installed): `line.sh` gains the lr-schedule arms `ARM_c`, `ARM_w` and
   `ARM_wb` (`../README.md`). `dry/all.sh` and `dry/check.py` cover them, and every earlier arm's dry-run commands
   are unchanged. From then on `line.sh` is no longer serrano's bytes.
5. `leon/grpo-v2-pilots` (2026-10-13, not yet installed): `line.sh` adds the L3 pilot arms `ARM_rfb`, `ARM_ulb` and
   `ARM_3pb`, and the full arm `grpo_think_vs3` (`../README.md`, "L3 pilots").
   - `ARM_rfb` and `ARM_ulb` need `grpo.py`'s pilot options in the code worktree. `line.sh` still runs
     `nd-rl-v2` on `leon/grpo-v2`, and it refuses these two arms until that worktree has them, for example by
     fast-forwarding `leon/grpo-v2` to this branch. The default path is digest-identical.
   - Arms with 3-premise rows are refused with a clear message while a `curriculum_3p` file is missing.
   - `dry/all.sh` and `dry/check.py` cover all of this. Every earlier arm's dry-run records are byte-identical to
     c4b8548's harness output: `exit`, stdout, call records, trees, logs, and the sandbox md5s apart from the
     launcher copy itself.

Checked and left alone:

- **4 GPUs, not 5.** Nothing assumes a count: the GPU is `queue.sh`'s and `line.sh`'s first argument, `waitfree`
  and `watchdog.py` query `nvidia-smi -i GPU`, and `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes that index the CUDA index.
  On ynez start queues on GPUs 0-3 only.
- **24 GB, not 48.** No script sets a memory size; batches are 1,024 for GRPO, EI and eval, as on serrano. On the
  A40s the v2 jobs mostly showed 9-15 GB in `nvidia-smi` (reserved, not peak allocated), and the thinking EI L3
  stages up to 20.7 GB (2026-10-10 11:30Z), which leaves about 3 GB on a 3090: watch the first EI line.
- **`waitfree`** is unchanged: under 500 MiB used and no compute process on that GPU, polled each minute for up to
  3 h, after which the line stops. On 2026-10-13 GPUs 0 and 1 held another user's 23 GB training processes.
- **`ND_LEAN=$HOME/.elan/bin/lean`** (`lib.sh`) works on ynez: `~/.elan` is on NFS (Lean 4.34.0, default
  toolchain), and the gate runs core Lean on one temp file per chunk, with no Mathlib, lake project or oleans of
  its own. A 3-theorem `lean_gate.check_sources` run on ynez returned `[True, False, True]` as expected.

## Install (done 2026-10-13)

    L=/scratch/cowifornia/nd-rl/think/nd-rl-v2/code/experiments/current/thinking_traces_20261005/grpo_v2/launch
    R=/scratch/cowifornia/nd-rl/think/runs
    mkdir -p $R/grpo_v2 $R/followup_20261007 $R/logs $R/tmp
    install -m 664 $L/line.sh $L/queue.sh $R/grpo_v2/          # serrano's modes
    install -m 775 $L/lib.sh $R/followup_20261007/
    install -m 664 $L/watchdog.py $L/backup.py $R/

## Backups (`backup.py`)

One pass mirrors the small, expensive artifacts of `runs/` to `/home/cowifornia/backups/nd-rl-think-ynez/` (NFS),
under the same relative paths, capped at 30 GB: launch scripts and logs, the ynez rebuild's scripts and notes
(`runs/rebuild_ynez`), per-run args/metrics/dashboards, eval rows and summaries (`eval/v2`, `eval/matched`,
`eval/tts`, `eval/charles`), final pools and test sets, Stage-1 models, stage-final checkpoints (`*_it39.pt`, EI
`*_r1.pt`) and data pickles under 2 GB. Never EI `found_*.jsonl`, work dirs, smoke dirs, registries, Lean gate
logs, or rolling and alarm checkpoints. It deletes only files it manages (`.manifest.json`: what it copied, and
any file already there that matched its source) once they leave the selection, so copies others put in the backup
dir stay. The docstring has the rules: tiers under the cap, the deletion guard against a lost `/scratch`, the
2-minute wait for fresh checkpoints, atomic copies, one pass at a time. Its log is `runs/logs/backup.log`;
`.last_pass.json` in the backup dir has the last pass's counts.

The queue runs a pass after every arm. An hourly loop, each pass bounded at 3 h:

    nohup setsid bash -c 'while true; do timeout 10800 nice python3 /scratch/cowifornia/nd-rl/think/runs/backup.py >> /scratch/cowifornia/nd-rl/think/runs/logs/backup.log 2>&1; sleep 3600; done' > /dev/null 2>&1 < /dev/null &
