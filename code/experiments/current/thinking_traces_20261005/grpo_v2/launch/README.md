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
| `pull_queue.sh` | `/scratch/cowifornia/nd-rl/think/runs/grpo_v2/pull_queue.sh` | | | new on ynez: one worker per GPU on a shared job file |
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
    bash all.sh             # every arm x seeds 0-2 x normal/SMOKE=1, refusals, a busy GPU, queue.sh (~20 min)
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
6. `leon/grpo-v2-prior` (2026-10-13, not yet installed): `line.sh` adds the L3 pilot arm `ARM_pb`, which is
   `grpo.py --unseen_prior mean` (`../README.md`, "Unseen-pair prior"). It has `ARM_rfb`'s checks and refusals, and
   refuses until the code worktree's `grpo.py` has `--unseen_prior` (the guard checks `UNSEEN_PRIORS`).
   `dry/all.sh` and `dry/check.py` cover it. Every earlier arm's dry-run records match 0de954e's harness output in
   the same way as above.
7. `leon/grpo-v2-merged` (2026-10-14, not yet installed): `line.sh` adds `ARM_mb`, the merged recipe
   (`../README.md`, "Merged recipe"). Its averaging is a `"$PY" -` call, so `dry/make.sh`'s python stub now also
   accepts `-` and writes the output file. `dry/all.sh` and `dry/check.py` cover `ARM_mb`: its arms and refusals, a
   missing rolling checkpoint, and a rerun that does not average again (section D). Every earlier arm's dry-run
   records match 7523c16's harness output in the same way as above.
8. `leon/grpo-v2-feedback` (2026-10-14, not yet installed): `line.sh` adds the L3 pilot arm `ARM_fbb`. It is
   `grpo.py --feedback context` with `--max_rej 2` retries per step (`../README.md`, "In-context verifier
   feedback"), followed by feedback-arm reads on t1320 and charles173.
   - It applies only to thinking arms with the verifier, and has `ARM_rfb`'s other checks and refusals.
   - It refuses until the code worktree's `grpo.py` and `eval.py` have the feedback options. The guard checks
     `FEEDBACK_MODES` and `READ_ARMS`.
   - `read1` takes an optional eighth argument, `--max_rej`, which defaults to the earlier 10.
   - `dry/all.sh` and `dry/check.py` cover it. Every earlier arm's dry-run records match ccfa1c7's harness output in
     the same way as above.
9. `leon/grpo-v2` (2026-10-15): `line.sh` adds the knob pilots `ARM_mb_kKNOB`. Each one is `ARM_mb`'s L3 (the merged
   recipe) from the same L2 checkpoint, with the same checks and refusals, and one setting changed:

   | KNOB | change from `ARM_mb` |
   |---|---|
   | `x66` | `--targets_extra_frac 0.66` (`XFRAC`, 0.33) |
   | `g64` | `--prompts 64 --group 64` (256 x 16: the same attempts per iteration; `SMOKE=1`: 2 x 8 for 4 x 4) |
   | `t12` | `--temperature 1.2`, training only (grpo.py's default is 1.0; the reads stay at 0.8) |
   | `kl0`, `kl1` | `--kl_anchor_coef 0`, `0.1` (0.02) |
   | `ch4` | `--clip_eps_high 0.4` (grpo.py's default is 0.28) |
   | `p3` | `--rej_penalty 0.3` (0.1); refused for an arm without the step penalty |
   | `i78` | `--iters 78 --save_every 78` in L3 (39). The WSD cooldown is the last 20% of the longer run, the reads use its `_it78` checkpoint, and the L2 checkpoint it starts from stays `_it39`. The stage timeout is 12 h instead of 6 h, and the watchdog expects 7 h instead of 3.5 h. |
   | `rr` | nothing: `ARM_mb`'s L3 again, as a noise control (GPU runs are not bit-reproducible) |

   - The knob replaces the recipe's value, so every flag is in the command once. `--kl_anchor_coef`'s 0.02 became
     `KLC`, and L3's iterations are `IT3` (`ITERS` otherwise). `t12` and `ch4` add a flag the recipe leaves at
     grpo.py's default. All of this is in one block of `line.sh`.
   - Reads: plain only, on t1320 k16, t2130 k8 and charles173 k64, in that order, under the usual names
     (`eval/v2/<set>_<arm>_s<seed>_plain_k<k>/`). Knob selection must never use textbook72, so there is no tb72
     read, and no guided, hand14 or averaged-checkpoint read.
   - Parsing: `_mb_k` must be followed by one knob from the table and nothing else. Every earlier suffix rule is
     applied to the arm without `_kKNOB`, so `_mb` combined with any other suffix is still refused.
   - `dry/all.sh` and `dry/check.py` cover all nine knobs on `grpo_think_vs_mb` and some on other `ARM_mb` arms.
     `check.py` compares each knob arm's L3 command with `ARM_mb`'s: only the knob's flags may differ. It also checks
     that no flag appears twice in any grpo or eval command, covers the refusals, and checks the 78-iteration stage
     (section K). Every earlier arm's dry-run results match 4796bd0's harness output: all 1,370 result dirs (exit,
     stdout, call records, trees, logs). For this comparison, pids were normalised, and the stage's two background
     calls (`sleep 5` and the watchdog launch), which race, were compared as a multiset. The only md5 differences are
     the sandbox launcher copy and the queue's logs, which hold pids.
   - `t12` and the safeguards at T = 1.2:
     - The canary is always scored at T = 1, and so is its base.
     - The KL anchor tempers both pi_ref and pi.
     - The unlikeliness reward ranks attempts under pi_old at the sampling temperature.
     - The clip-low and grad-norm alarms compare a run with its own history.

     So no alarm compares a tempered value with a base taken at another temperature. The think-ban mass (alarm
     1e-3, `--ban_stop` 0.005) is an absolute threshold on the tempered full-vocabulary mass.

     A CPU check (2026-10-15) ran one iteration from `grpo_think_vs_s0_L2_it39`, 24 groups x 8 attempts. On the
     same T = 1.0 samples, scoring at T = 1.2 raised the mean mass from 2.0e-4 to 3.1e-4. On T = 1.2 samples it was
     7.8e-4 scored at T = 1.0 and 8.6e-4 at T = 1.2. Most of the rise comes from the states T = 1.2 sampling
     reaches, not from the scoring, so grpo.py is unchanged. `t12` starts L3 near the alarm, though, so expect
     think-ban alarms (with the restore check, possibly rollbacks) in its guard lines.
10. `pull_queue.sh` (2026-10-15): the backup pass after each job is opt-in (`BACKUP=1`). Its destination on `/home`
    was removed on purpose (minimal data duplication on `/home`), and a pass would recreate it. `queue.sh` still
    runs one after each arm.
11. `leon/grpo-v2` (2026-10-16, not yet installed): `line.sh` adds the EI knob pilots `ARM_kKNOB`, for an EI arm
    (`ei_think_v_kKNOB` asked for; `../README.md`, "EI knob pilots"). Each is ARM's L3 from ARM's finished L2
    checkpoint (`ARM_sSEED_L2_r1.pt`; refused without it), with one setting changed in the knob block:

    | KNOB | change from ARM's L3 |
    |---|---|
    | `rr` | nothing (the noise control) |
    | `3p` | `--targets` the fixed L3 + 3-premise mix `runs/pools/curriculum_3p/L3_mix_ei.jsonl` (`T3`; `SMOKE=1`: its first 8 rows) |
    | `ret0` | `--retain 0` (`RT3`; the default passes none, ei.py's 20,000; `SMOKE=1` passes 64) |
    | `t10` | `--temperature 1.0` (`ET`, training samples only; 0.8) |
    | `r2` | `--rounds 2 --k 16 --ft_steps 300` (`RD3`, `K3`, `FS3`: one round's attempts and fine-tune steps; `SMOKE=1`: k 1, 1 step a round). The stage ends at, and the reads take, `_r2.pt` (`ck` and `ei_stage`) |

    - Every flag is in the command once: the smoke `--ft_steps 3` and `--retain 64` moved out of `FT` into `FS` and
      `RET` (in the same order), so the existing EI commands are unchanged. `ei_stage` takes stage L3's rounds, k,
      temperature, retain and fine-tune steps from the knob block, and L2's are as before.
    - `3p` is refused while the mix is missing, or while its pass record
      (`L3_mix_ei.preflight_pass.txt`, from `grpo_v2/build_mix_ei.py`) does not name the mix's md5. At launch the
      mix also gets the extra-targets preflight (L2 pool vs the mix; only the band line may fail).
    - Reads and `KNOB_FULL=1` are as for `ARM_mb_kKNOB`.
    - Parsing: an EI arm's first `_k` starts the knob. It must be one of `rr 3p ret0 t10 r2` with nothing after it,
      and the part before it must be an EI arm with no other suffix. `_mb_k` on an EI arm keeps `_mb_k`'s parse and
      refusal. An EI knob on a GRPO arm (`grpo_think_vs_kret0`) is refused, and any other `grpo_*_kX` stays an
      unknown arm, as before.
    - `pull_queue.sh` runs these arms unchanged: its job lines are `[a-z0-9_]+:[0-9]+`, and `ei_think_v_kret0:0`
      is one.
    - `dry/run.sh` passes `DRY_KNOB_FULL` on as `KNOB_FULL`. `dry/all.sh` and `dry/check.py` cover the ten EI knob
      arms and their refusals (section B), each knob command against ARM's L3 command (only the knob's flags may
      differ), and section J: `_kr2`'s `_r1` / `_r2` checkpoints, reruns of finished and part-read pilots, and
      `KNOB_FULL=1` on finished EI pilots, on a fresh one and on a finished GRPO knob pilot. That is 1,940 result
      dirs, and all checks pass. All 1,628 result dirs of 7c19dac's harness match its output (exit, stdout, call
      records, trees, md5s, logs), with pids normalised, a stage's two racing background calls compared as a
      multiset, and the sandbox launcher copy's md5 dropped.
    - Smoke (`SMOKE=1`, CPU, a scratch runs root with the inputs linked read-only): `ei_think_v`, then each of the
      five knobs, then `KNOB_FULL=1` and two reruns on `ei_think_v_kr2`, and four refusals. All ran as described.
    - Follow-up (2026-10-16): `r2` adds `--ft_steps 300` (`FS3`), so its two fine-tunes total one round's 600 steps.
      In the dry run, all checks pass. Of 17cd1e6a's 1,940 result dirs, only the 18 where an `r2` L3 stage runs
      differ, in its command and launch line alone (`--ft_steps 300`; `SMOKE=1`: 1 instead of 3). A CPU smoke of
      `ei_think_v_kr2`, then `KNOB_FULL=1` and a rerun, ran 2 x 1 fine-tune step.
12. `leon/grpo-v2` (2026-10-16, not yet installed): the mini runs and the checkpoint reads (`../README.md`, "Mini runs").
    - `grpo_think_vs_mini[_kKNOB]`: L2-mini then L3-mini, 10 iterations each, from `S1g3_s0` at every seed, on
      `runs/pools/mini` (`grpo_v2/build_mini.py`; refused without the subsets or with a subset that differs from its
      pass record). Both commands come from `grpo_stage`.
      - L2-mini is `grpo_think_vs`'s L2 and L3-mini `grpo_think_vs_mb_kkl0`'s L3. They differ only in `--iters` /
        `--save_every`, the subsets, `--save_rolling 2`, the 2 h timeout (watchdog 1 h) and L3-mini's init (L2-mini's
        `_it10`).
      - New variables keep every earlier command byte for byte: `TO2` / `WH2` (stage L2's timeout and watchdog, the
        former literals), `KL3` (stage L3's KL coefficient when set), `ROLL` (`--save_rolling`, empty), `$mk` (the mini
        knob). `init0` is fixed to `S1g3_s0` for the mini arms. In `SMOKE=1` the 3-premise subset gets its head even
        without an L2 extra file.
      - Knobs: `rr`, `t07` `t085` `t115` `t13`, `lpNN` / `lpmNN` (`--len_penalty` ±NN/100), `kl1` (L3-mini only), `g64`.
      - Reads: `MINI_READS` of L3-mini's last checkpoint (`l2test` k8, `l3test` k8, t1320 k8, t2130 k8, charles173
        k64). `MINI_ROLL` (env) adds the `MINI_PANEL` reads of rolling checkpoints. After each stage the line logs its
        `ALARM` lines and rollback count (`stage_alarms`).
    - `ARM_mb_kKNOB_ckreads`: `CKR_READS` of a finished GRPO knob pilot at it0 (its L2 init), it10 / it20 / it30
      (rolling) and it39 (its own names), skipping the reads already there. It writes `ckreads_commit.txt`, runs no
      preflight and trains nothing.
    - Refused: another base arm, another or a second knob, `lp00`, anything after the knob, `KNOB_FULL=1`; for
      `_ckreads`: a non-knob or EI arm, `_ki78`, `KNOB_FULL=1`, no finished L3, no L2 checkpoint.
    - grpo.py: `--len_penalty`'s length-only groups (no outcome signal; on top of the loss's divisor; `row['len_only']`).
      At lp 0 nothing changes: c4b8548's digests hold for `plain` and for the new `mixed0`.
    - `pull_queue.sh` runs these unchanged (`grpo_think_vs_mini_klpm10:0` and `grpo_think_vs_mb_kkl0_ckreads:0` match
      `[a-z0-9_]+:[0-9]+`).
    - `dry/run.sh` passes `DRY_MINI_ROLL` on as `MINI_ROLL`. `dry/all.sh` and `dry/check.py` cover all of it in sections
      M (every mini arm at seeds 0-2 and the reference at 3, against `expect()` and against the full recipe's commands),
      N (reruns, `MINI_ROLL`), Q (`_ckreads`) and B (the refusals). That is 2,272 result dirs, and all checks pass.
      All 1,940 result dirs of 765161f9's harness match its output (exit, stdout, call records, trees, md5s, logs),
      with pids and the two harness homes normalised, the racing background calls compared as a multiset, and the
      launcher copy's md5 and the records' environment hash (it holds the harness home) dropped.
    - Smoke (`SMOKE=1`, CPU, a scratch runs root): the reference at seeds 0 and 3 (both from `S1g3_s0`), each knob, a
      rerun (nothing runs), `MINI_ROLL` (its 2-iteration stages have no rolling checkpoint: all three skipped), and four
      refusals. All ran as described. The `len it` lines show the length-only groups (`lpm10` it 1: 1 of 4 groups, no
      unlikeliness ranking there), `kl1`'s L3 has the anchor at 0.1 and its L2 at 0.02, and `t13`'s L3 raised the think
      ban alarm at its first iteration (mass 2.0e-3 against a restore point at 2.1e-3: no rollback, as the restore check
      says).
13. `leon/grpo-v2-dev` (2026-10-16, launcher only: nothing imported by `code.thinking.grpo` / `eval` changes, so it can
    be fast-forwarded while runs are going): the mini forks, the pool knobs and the generalised checkpoint reads
    (`../README.md`, "Mini runs").
    - Forks `grpo_think_vs_mini_kkl1` and `_kg64` (now L3-mini only): `src` is the reference, whose `_L2_it10` at the
      same seed is the init and KL reference. A fork waits (once a minute, up to 3 h, then FAILED) until that
      checkpoint is there and the reference's job log has `GRPO grpo_think_vs_mini_sS_L2 exit 0:`, and refuses unless
      the reference's L2 log ends at `lr x1 (0 rollbacks)`. `g64b` is the both-stage `g64`.
    - `pool250` / `pool500`: `MPN` picks the nested subsets (`L2_mini250.jsonl`, …), checked against the pass record as
      the 1,000-row ones. `size` is refused until its data exists.
    - A mini stage that ends without its last checkpoint logs `GRPO <stage> FAILED: exit N (--ban_stop: …)`.
    - Reads: `MINI_TRAIN` (`l2train`, `l3train`: the training subsets as read, `L2T` / `L3T`, before `SMOKE=1` takes
      their heads) for the reference, `rr` and the pool arms; the T arms' `MINI_PANEL` at their T (`read1`'s 9th
      argument: `--temperature` and the `_T07` suffix); `MINI_PANEL` is now `l3test` k8, t1320 k8, t2130 k4,
      charles173 k16. `reads_of` runs a read named in two of its lists once. Follow-up: a rerun of a finished mini
      line (its L3-mini's last checkpoint there) writes HEAD to `ARM_sS.reads_commit.txt` and keeps the training's
      `base_commit.txt` (`bc`, as `KNOB_FULL=1`); `dry/check.py` section N checks both records. In the dry run all
      41,450 checks pass; against c00f45a6's 2,374 result dirs only the six finished-line reruns differ (their record
      file and call), the harness home aside. A smoke rerun of a finished `grpo_think_vs_mini` (`MINI_ROLL=L3:0`) kept
      its `base_commit.txt` and wrote `reads_commit.txt`.
    - `ARM_ckreads` for any full GRPO L3 that started from another arm's L2 (`_ki78` included: `CKR_ITS` is every 10
      below `IT3`), each iteration from the nearest earlier rolling checkpoint, recorded in `ARM_ckreads_sS.ckpts.txt`,
      then `CKR_FINAL` of the last. `ARM_l2_ckreads` (lower case, as pull_queue.sh's job lines must be) and
      `s1g3_ckreads` read a base arm's last L2 and the Stage-1 init (`CKR_PANEL` and `MINI_READS`). HEAD goes to
      `<job>_sS.base_commit.txt`. A missing checkpoint is refused before anything is written.
    - `SMOKE_WT` (with `SMOKE=1` only): a dev worktree for the line (`WT`, and its branch `leon/grpo-v2-dev`). Without
      `SMOKE=1` the line stays pinned to `nd-rl-v2` / `leon/grpo-v2`.
    - `dry/`: `make.sh`'s grpo stub exits 3 without a checkpoint for a stage name containing `DRY_GRPO_STOP` (`run.sh`
      passes it on). `dry/all.sh` and `dry/check.py` cover the forks, `g64b`, the pool
      arms, the training-subset and T reads, the fork's wait (180 one-minute waits, then FAILED) and rollback
      refusal, `--ban_stop`'s FAILED line, the new `_ckreads` jobs (kl0; t12 with holes; i78; `ARM_mb`; fbb without
      rolling checkpoints; the shared L2; s1g3) and their refusals. That is 2,374 result dirs, and all 41,438 checks
      pass (`check.py --expect-hook`). All 1,940 result dirs of 765161f9's harness match its output at the same
      harness home, environment hash included (the commit records' contents aside: another `nd-rl-v2` HEAD).
      Against 74864c28's harness only mini and `_ckreads` dirs differ. The `lp` arms' calls and stdout are unchanged
      (only the setups' nested subsets are new), and the reference's commands are unchanged but for its two added
      reads.
    - Smoke: `SMOKE=1`, CPU, a scratch runs root, `SMOKE_WT` the dev worktree. The reference at seed 0 (L2,
      L3, seven reads); `g64b`, `t07` (four `_T07` reads), `pool250` (its 250-row training reads); `MINI_ROLL="L2:0
      L3:0 L3:6"` on a rerun (eight reads of the two real `_it0_rolling.pt` files, L3:6 skipped); `ksize` refused; a
      fork at seed 1 with no reference waits (stopped by its 90 s timeout). Both forks at seed 0 refused, as they
      should: the smoke reference's 4 x 4 L2-mini rolled back once at a clip-low alarm. At seed 2 (no rollback) both
      ran L3-mini only from `grpo_think_vs_mini_s2_L2_it2.pt`, which is also their KL reference (`kl1`: coefficient
      0.1; `g64`: 2 x 8 at 0). Then `grpo_think_vs` and `_kkl0`, `_kkl0_ckreads` (`CKR_FINAL` only: a 2-iteration L3
      has no `CKR_ITS`), `grpo_think_vs_l2_ckreads` and `s1g3_ckreads` (five reads each: in `SMOKE=1` every k is 2,
      so `MINI_READS` repeats `CKR_PANEL`'s dirs and runs once), and a rerun of `_kkl0_ckreads` that read nothing.
14. `leon/grpo-v2-dev` (2026-10-16): grpo.py's mini-run knobs and their arms. This one changes `code.thinking.grpo`:
    fast-forward it only once no run that imports `nd-rl-v2`'s grpo.py is going.
    - grpo.py: `--loss_tokens all|action|think` and `--adv_sign both|pos|neg`, opt-in (`argparse.SUPPRESS`: left out of
      args.json unless given). `policy_loss(tokens=)` trains one phase's scored tokens, with its divisor over them;
      the entropy and the ban mass stay over every scored token (`LossStats.tokens_all` weights the logged
      entropies). `group_items(adv_sign=)` zeroes one sign of the outcome advantage after the deductions; the step
      penalty's −P stays.
    - Tests: `tests/thinking/test_grpo_mini_knobs.py` (each phase's loss and gradient equal the full loss on masks
      that keep that phase, with or without the PPO ratio; budget 0 under `think`; the signs, after the unlikeliness
      deduction; the step penalty kept; args.json; the defaults given change nothing but args.json against c4b8548's
      `plain` digest). All 267 GRPO / EI tests pass (10 new), and ruff, ruff format and pyright are clean over `code/`.
    - line.sh: `grpo_think_vs_mini_klossact` / `_klossthink` / `_kadvpos` / `_kadvneg`, both stages, refused unless
      the worktree's grpo.py has `LOSS_TOKENS` and `ADV_SIGNS`. In the dry run (sections A-Q and the
      four arms at seeds 0-2, both modes, in M) all 42,170 checks pass over 2,398 result dirs. All 1,940 of
      765161f9's match, and against 0923b391's 2,374 only the 17 refused mini names' stdout differs (the knob list in
      the message), the harness home aside.
    - Smoke: `SMOKE=1`, CPU, `SMOKE_WT` the dev worktree. `lossact`, `lossthink` and `advpos` ran both stages and
      the five reads (args.json records the knob). `advneg`'s 4 x 4 L3-mini at seed 0 hit `--ban_stop` at its second
      iteration (think-ban mass 0.0113) and logged the FAILED line with no reads; at seed 2 it ran through. Without
      `SMOKE_WT` (`nd-rl-v2`'s grpo.py, which has neither option) `lossact` is refused.
15. `leon/grpo-v2-dev` (2026-10-18, audit 3): the adv knobs reworked, crash-safe reruns, the T-panel reference job.
    Changes `code.thinking.grpo` (`--adv_sign neg` only), so fast-forward it between jobs.
    - grpo.py: `--adv_sign neg` decides by outcome. It keeps the failed attempts' advantages (at most 0) and zeroes
      every solved attempt's, also one the unlikeliness deduction made negative. `pos` is unchanged (by sign).
    - line.sh: `advpos` and `advneg` are L3-mini forks of the reference's L2-mini at the same seed, with the same wait
      and guard as `kl1` / `g64`; `advpos` runs at `--rej_penalty 0` (`P`, in its launch line as `P 0` and `knob advpos:
      --adv_sign pos --rej_penalty 0`). `lossact` / `lossthink` stay both-stage.
    - A mini line writes `base_commit.txt` once; a rerun that trains again records HEAD in `rerun_commit.txt` (a
      finished line's rerun: `reads_commit.txt`, as before).
    - `grpo_stage`: a stage dir with `args.json` or `metrics.jsonl` but no last checkpoint (a crash, a stop) is moved
      aside with its log to `*.crashed_<UTC>` before the stage reruns; the line logs it. Every GRPO arm (EI stages are
      unchanged). Dirs without either file (the dry harness's stub rolling checkpoints) stay, so earlier result dirs
      are unchanged.
    - `grpo_think_vs_mini_tpanel` (`ckr=tpanel`): MINI_PANEL of the reference's L3-mini last at `TPANEL_TS` (0.7,
      1.3), under the reference's names with `_T07` / `_T13`; refused for another arm or without the L3-mini.
    - Tests: `test_grpo_mini_knobs.py`'s sign test covers neg by outcome (a solved attempt made negative by the
      deduction is zeroed; with a length penalty that takes the mean below 0, a failed attempt's positive advantage is
      capped at 0). All 267 GRPO / EI tests pass; ruff, ruff format and pyright are clean. Dry run (`dry/`: the adv forks, tpanel
      and its refusals, a crashed stage's rerun for the mini reference's L3-mini and for `grpo_think_vs`'s L2): all
      42,479 checks pass over 2,440 result dirs. All 1,940 of 765161f9's match, and of 344d80b0's 2,398 only the
      12 adv-arm dirs differ (now forks).
    - Smoke: `SMOKE=1`, CPU, `SMOKE_WT` the dev worktree, 4 threads. The reference at seed 2, then `advpos` (L3-mini only
      from its `_L2_it2`, which is also its KL reference; `--rej_penalty 0`, the launch line `P 0`) and `advneg`; then
      `tpanel` (eight reads, `_T07` / `_T13`). A crash: the reference at seed 5 killed (its whole process group) 35 s
      into L2-mini, with `args.json` and a one-row `metrics.jsonl` there. The rerun moved the dir (its `_it0_rolling.pt`
      too) and the log aside to `*.crashed_20261018T024716Z`; the new `metrics.jsonl` holds only the rerun's rows;
      `base_commit.txt` was kept and `rerun_commit.txt` written.
16. `leon/grpo-v2-dev` (2026-10-18): the size knob. Launcher only (nothing `code.thinking.grpo` / `eval` imports).
    - `grpo_v2/build_size.py` builds the size-twin mix into `runs/pools/mini/` (`L{2,3}_mini_size.jsonl`,
      `L3_3p_mini_size.jsonl`, `size.json`, `preflight_size*.log`, its own `preflight_pass_size.txt`); built, and a
      rebuild gives the same bytes.
    - `grpo_think_vs_mini_ksize`: both stages on the mix (`MPN=_size`, checked against `preflight_pass_size.txt`; the
      refusal names `build_size.py`), knob text `the size-twin mix (a fixed half of the rows as their big-statement
      twins)`; reads `MINI_READS`, then `SIZE_READS`.
    - `grpo_think_vs_mini_szpanel` (`ckr=szpanel`): `SIZE_READS` of the reference's L3-mini last; refused for another
      arm, without the L3-mini.
    - `SIZE_READS` = `l2test_sz`, `l3test_sz` (the test_1k twins) and `l2test_om`, `l3test_om` (their matched
      originals), k8, from `runs/pools/size_twins/` (`SZ_SETS`, checked before anything is written).
    - Dry run (`dry/`: `_ksize` in M at seeds 0-2, both modes, `szpanel`, the refusals): all 42,893 checks pass over
      2,482 result dirs. All 1,940 of 765161f9's match, and of 80eaca21's 2,440 only the six `_ksize` dirs differ
      (refused before). `_ksize` against the reference: L2 `--targets` `L2_mini` -> `L2_mini_size`; L3 `--targets`
      `L3_mini` -> `L3_mini_size`, `--targets_extra` `L3_3p_mini` -> `L3_3p_mini_size`, and the names and init.
    - Smoke: `SMOKE=1`, CPU, `SMOKE_WT` the dev worktree, 4 threads: `_ksize` at seed 0 (both stages on the mix's heads,
      five of the first eight L2 rows twins; `MINI_READS` and the four size reads), the reference at seed 2, then
      `szpanel` (four reads) and its rerun (nothing left to read).
17. `leon/grpo-v2-dev` (2026-10-18): the mini20 variant, launcher only. `grpo_think_vs_mini20[_kKNOB]` parses as a mini
    arm with `MV=20`: `ITERS=$MINI20_ITERS` (20), `TO2` / `TO3` 10800 and `WH2` / `WH3` 2; a fork's `src` is
    `grpo_think_vs_mini20`, so it waits for and starts from that reference's `_L2_it20`; `_tpanel` / `_szpanel`
    accept `grpo_think_vs_mini20` too. Nothing else in the mini path changes.
    - Dry run (`dry/`: every mini arm and fork again as mini20 at seeds 0-2, both modes, its tpanel and szpanel,
      three refused names): all 46,799 checks pass over 2,632 result dirs, and every result dir of 765161f9's
      (1,940) and 0a7d2cc4's (2,482) is unchanged. The mini20 reference against the mini reference: in both stages
      `--iters` and `--save_every` 10 -> 20 and the timeout 7200 -> 10800 (watchdog 1 -> 2 h); in L3-mini `--init` and
      `--kl_anchor_ref` its own L2-mini's `_it20`; the names.
    - Smoke: `SMOKE=1` (2-iteration stages whatever the variant), CPU, `SMOKE_WT` the dev worktree, 4 threads: the mini20
      reference at seed 2 (seven reads), its `kl1` fork (L3-mini from `grpo_think_vs_mini20_s2_L2_it2.pt`, also
      the KL reference, at 0.1), its `tpanel` (eight reads) and `szpanel` (four), and `ksize` at seed 0 (nine).

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

`queue.sh` runs a pass after every arm, and `pull_queue.sh` only with `BACKUP=1` (since 2026-10-15; the destination
was removed on purpose, item 10 above). An hourly loop, each pass bounded at 3 h:

    nohup setsid bash -c 'while true; do timeout 10800 nice python3 /scratch/cowifornia/nd-rl/think/runs/backup.py >> /scratch/cowifornia/nd-rl/think/runs/logs/backup.log 2>&1; sleep 3600; done' > /dev/null 2>&1 < /dev/null &
