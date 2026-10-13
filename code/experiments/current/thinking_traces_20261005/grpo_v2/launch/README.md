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
| `queue.sh` | `/scratch/cowifornia/nd-rl/think/runs/grpo_v2/queue.sh` | 851 | `8acdfe27…008c` | byte exact |
| `lib.sh` | `/scratch/cowifornia/nd-rl/think/runs/followup_20261007/lib.sh` | 2,245 | `6f2aa471…ca95` | byte exact |
| `watchdog.py` | `/scratch/cowifornia/nd-rl/think/runs/watchdog.py` | 3,786 | `2442d2b5…ab8a` | byte exact |
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
  wrote it; reads on 10-08 09:01Z (all 104 lines), 10-08 21:38Z (lines 1-80) and 10-09 21:11Z (1-30) match.

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
