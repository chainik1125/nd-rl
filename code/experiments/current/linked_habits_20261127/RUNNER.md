---
written_on: 2026-11-27
written_by: agent:claude (runner build agent)
---

# Linked habits version 2: measurement, gate, stage 2 and the Runpod steps (runner)

This note covers the runner of `SPEC2.md` with PREREG.md Amendments 1-6 (Amendment 3 applies the design audit,
`discourse/design_audit.md`; Amendments 4 and 6 trim the plan to the budget; Amendment 5 adds early snapshot steps): pretraining on the four corpora and `c0n`, the measurement of 24 habits, the performance
control, the gate, the stage-2 inputs and arms, the readout, and the pods. The corpus agent wrote the corpora, the
detector (`lh_profile.py`, `lh_habits.py`) and the twins (`lh_twin.py`); the third agent wrote the trainers
(`lh_tune.py`, `lh_dpo.py`, `lh_sft.py`, `lh_rm.py`). No GPU job ran and no pod was created or contacted: everything
below ran on serrano's CPU (at most 8 cores, at most 3 since the coordinator's limit at about 10:20 UTC) as a unit
test, a CPU smoke or `--dry-run`.

Labels: **[obs]** observed in this build, **[interp]** my reading, **[guess]** untested. W = this folder, W2 =
`../two_authors_20261111` (commit 9618763), LH = `/scratch/cowifornia/nd-rl/linked_habits`,
PY = `/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python`.

## Files of the runner

| file | what |
|---|---|
| `lh_core.py` | names and paths: LH (and `TA = AT = LH` for W2's modules), the corpora (env `LH_CORPORA`, default c24,c9,c4,c0), the extra base `c0n` on the c0 pod (`LH_EXTRA_BASES`), the Nov 14 bases, the step limit 51, the line cap 16, the raters (`reit`, `atom` = `note_atom`), the step grid, the arms, the task list and dependencies per corpus, `linked_of`, the detector's tables |
| `lh_launch.py` | jobs `pretrain`, `measure` (+ strict 106), `prep2`, `stage2`, `rl`, `status`, `wait`, `steps` (`--extra c0n`), `nov14`; entry points `_pretrain`, `_line`, `_tasks` (the scheduler: GRPO lines alone, other tasks two at a time, each with its share of the CPUs), `_task`, `_rm`, `_offline` (train; probe a rated run), `_full` (the bracketing snapshots and the last, measured), `_job`. W2's `authors_launch` imported and patched, never copied |
| `lh_measure.py` | `run` (W2's measurement with the 24-habit counts, step limit 51, pass@1 / pass@8, near misses), `probe` (a rated run's snapshots in step order, untagged marker_plain, one process, stopping past +30 and then the last), `strict106` (Dan's read at the limit, profiled), `sets` |
| `lh_pairs.py` | `anchor` (the KL anchor's canary/1 states and the trainers' KL states), `sample` (the base's proofs on Dan's RL pool), `build` (balanced pairs, the two fine-tune files, the reward-model probe twins unless `--no_probe`) |
| `lh_rlvar.py` | the two GRPO arms through W2's RL-variant hook (argv changes; the reward model's bonus), and `pool` (the RL pool's largest premise class) |
| `lh_stats.py` | per-theorem count columns and the bootstrap over theorems |
| `lh_gate.py` | G0, G1, G2 (primary: other habits), the performance control with the pass@1 flag, PREREG's decision check |
| `lh_read.py` | the readout: tables, the +30 read by interpolation, seed means, P1 / P2 (S, S_before, S_after, first sites, site split) / P3, chain prediction, reward-model marks, composition prediction, form-type groups, the readings table |
| `lh_driver.sh`, `lh_pod.sh`, `lh_pod_setup.sh` | the pod: the driver (both stages); setup / stage / copy / check-env / start / go / nogo / job / check / fetch from serrano |
| `test_lh_{rlvar,measure,pairs,launch,gate,read,pod}.py` | tests (below) |

Nothing in W2, `../author_tags_20261110/`, `../knockout_danbase_20261104/`, Dan's tree or TA was edited by hand; see
"Incidents" for a formatting pass on other agents' files and one log line appended in Dan's tree copy.

## Artifacts written to LH by this build

| file | what |
|---|---|
| `LH/data/pretrain_steps.json` | the step counts: the four corpora at the top (`lh_launch.py steps`), `c0n` under `extra.c0n` (`steps --extra c0n`) |
| `LH/data/pretrain_steps_v1.json` | version 1's step file (l-corpora), renamed so it cannot be read by mistake |
| `LH/bases/stage1_{authors,control}_half_s0.{pt,tag.json,pt.args.json}` | `lh_launch.py nov14`: byte copies of TA/bases (sha256 checked), staged only to the c4 pod |
| `LH/data/rewarded_class.json`, `rl_pool_premises.json`, `LH/sets/*` | from version 1, unchanged and still used (the note rater's class; the measurement sets) |
| `LH/smoke_v2/`, `LH/smoke_v2_sample/`, `LH/smoke_v2_rm/` | CPU smoke trees (disposable) |
| `LH/pod_stage/rp-lhaudit{,4,0}*` | stage copies for the closure audit (disposable, never sent anywhere) |

## Numbers fixed before any pod

**Step counts** [obs] (`lh_launch.py steps`; tokens the trainer encodes: W2's loader at cap 16, shown tags; state ids +
action ids + eos per pair):

| corpus | records | pairs | tokens | tokens / authors_half | seed-0 steps |
|---|---:|---:|---:|---:|---:|
| authors_half | 155,000 | 1,276,494 | 126,508,088 | 1 | 25,607 (Dan's) |
| c24 | 155,000 | 1,626,467 | 219,579,069 | 1.735692 | 44,600 |
| c9 | 155,000 | 1,626,467 | 219,495,936 | 1.735035 | 44,600 |
| c4 | 155,000 | 1,626,268 | 219,765,838 | 1.737168 | 44,600 |
| c0 | 155,000 | 1,626,710 | 219,897,925 | 1.738212 | 44,600 |
| c0n | 155,000 | 1,344,877 | 137,753,401 | 1.08889 | 28,000 |

The four c bases share the largest factor (1.738212, c0): seeds 0, 1, 2 train 44,600, 41,800, 42,400 steps. `c0n`
uses its own factor (equal passes over its own data; the coordinator's decision under Amendment 3 point 9): 28,000,
26,200, 26,600. Rounded up to the logging interval 200. Each entry records its corpus's sha256; `pretrain` refuses a
corpus that changed after the count. **Line cap 16**: the loader runs at 16 (tag_pretrain passes 12; the cap is only
an assert in Dan's loader), the census refuses any record over 16 lines, and a training record the loader skips stops
the run. Census [obs]: c24 and c0 at most 16 lines; c0n at most 15, no notes.

**The note rater's class**: `atom` (24.7 % of Dan's RL pool theorems have an atom premise; unchanged). **Sets**: W2's
`marker_plain` (500) and `route_unseen` (300); no `note_plain` (79.4 % of marker_plain has at least 2 premises).

## The measurement (`lh_measure.py run`)

W2's measurement (Dan's sampler at T 0.8, k 8, paired noise seeds, the Lean gate; tags A, B and none on the bases and
on the last snapshot of each corpus's seed-0 reiteration DPO run, untagged on every other tuned snapshot: Amendment 4)
with:
- **step limit 51** in every sampler call of the process (`state_sample.env_generate` wrapped; W2's `MAX_STEPS` = 51 in
  every summary; each row has `lh.max_steps` and `lh.step_cap`, the theorem's attempts that ended at the limit);
- **the 24 habits** of every accepted proof through `lh_profile.profile`. Row field `lh`, counts only: `hab` {habit:
  [sites, A, B, other, none]} (other: sites in neither form, e.g. a both-halves near miss, reported beside the A-share
  A / (A + B)), `with_site`, `first`, `later`, `link` and `linko` {"X|Y": [AA, AB, BA, BB]} for X, Y in notes / natural
  / all (the proof's first site of group X against later sites of group Y; `linko` only later sites of other habits,
  other classes for notes: the primary G2), `pair` {"h1|h2": ...}, `cond` {R: {h: [hA|RA, hB|RA, hA|RB, hB|RB]}} and
  `condn` {R: [first R site A, B]} for R = reit, note_atom (sites of other habits before the proof's first R site:
  the GRPO arm's chain prediction), `near` (lh_profile's near-miss counters, summed), `proofs`, `unread`, `lines`,
  `seen`. Left out of the site sequence (first / later / link / linko / pair / cond): the route, case split and ex
  falso (Amendment 3 point 8);
- **summaries** gain `lh_totals`: accepted % of attempts, pass@1, pass@8, mean lines, % at the step limit, the A-share
  and `other_pct` of every habit, near misses (total and per proof), the tables' sha256.

`probe` (Amendments 3 and 4): `run` untagged on marker_plain at k 8 for a rated run's snapshots in step order in one
process (the same sampler, step limit, Lean gate and detector), each into its folder with DONE; with `--stop_habit H
--stop_above X` it stops after the first snapshot whose A-share of H is above X (the base's share + 30) and then probes
the last snapshot; probe.json lists every probed snapshot (A-shares, sites, accepted %, pass@1, share at the limit), where
it stopped and what it skipped. The acceptance rule is the full measurement's (Lean), not nd_verify.

`strict106`: Dan's read of the strict 106 (tag_danread.py --tag none, k 64, batch 128) in the same process, so at the
limit 51, then W2's profile of its proofs; Dan's gate log goes to the measure folder. The `measure` job runs both for
every base: the four c bases, `c0n` (on the c0 pod), and on the c4 pod `authors_half_s0` and `control_half_s0`.

## Stage 2 (`prep2`, then `stage2`; PREREG Amendments 3 and 4)

**prep2** (one job): `lh_pairs.py anchor` (CPU, about a minute): 2,000 records drawn (seed 20261127) from the corpus's
untagged records, every step of each proof as canary/1 rows by `code.thinking.canary.proof_row` (the row builder of the
GRPO snapshot's collapse_audit/build_canary.py; that script reads fixed long-proof pools and sealed-set indexes, so it
is not run) -> LH/data/anchor_<c>.jsonl (about 21,000 steps [obs on c24: 200 records gave 2,085]); the first 256 records
also go to LH/data/klstates_<c>.jsonl (the offline trainers' logged KL). `sample` (GPU): the base, untagged, on Dan's
4,495 RL-pool theorems x 8, step limit 51 -> LH/pairs/<c>_s0/sampled.jsonl.gz (raw). `build --rater R` (CPU): every
distinct accepted proof read by the detector; a proof with all sites of the rated habit in one form gets
`lh_twin.twin`, read again (rated habit flipped at every site, every other habit unchanged), all twins checked by Lean
in one batch. Split by whole theorems (seeded) into held-out (until both sampled sides have 100 pairs) and train; then:
- `train.jsonl`, `heldout.jsonl`: each split subsampled (seeded per corpus, rater, split) to equal numbers of A-sampled
  and B-sampled pairs (held-out 100 + 100 at most). DPO, its swap control and the reward model use these.
- `sft_pref.jsonl` / `sft_sampled.jsonl`: the preferred / the sampled member of every train pair, before the balancing
  (written for both raters; only the reiteration rater's are trained on).
- `probe_twins.jsonl` (reiteration rater only; `--no_probe` for the note rater, which has no reward model): from the
  held-out theorems' sampled proofs, twins of every other flippable habit (the 15 notes, restate, halves, conj,
  disjunct), same checks and Lean, `preferred` = the A member, at most 400 per habit (seeded).
- `summary.json`: counts (why each proof gave no pair, before / after the balancing, probe twins per habit) and the
  A-share of every habit in sft_pref, sft_sampled and all the base's readable sampled proofs.

**stage2** (one job holding the GPU's lock; `_tasks` runs the tasks in list order, each in its own process under its
own `timeout`, 120 s stagger, a task waiting for the tasks it depends on). A GRPO line takes both slots of the card
(runs alone); the other tasks run two at a time. Each running task gets its share of the pod's CPU quota as Lean workers
and threads (the quota divided by the tasks running, never above the job's 8).

| task | what |
|---|---|
| `rl_<c>_rlhfreit_s0` | `_rm`: `lh_rm.py fit --max-tokens 32768` (rm.pt only at held-out accuracy >= 0.90; `_rm` checks it again); `lh_rm.py probe` on probe_twins.jsonl -> LH/arms/rm_<c>_reit_s0/probe.json (never a gate); then the GRPO line. Under 0.90 (`fit` exits 3 with `rm.pt.failed`): LH/arms/rl_<c>_rlhfreit_s0.not_run.json with the reason, the line is **not run and never retried** (a later driver run skips it), listed in `<c>.NOT_RUN`, not `<c>.FAILED`; the readout prints it. A crash of the fit is not a decision: the next driver run fits again |
| `rl_<c>_anc_s0` | the GRPO line, anchor and no learned reward |
| `dpo_<c>_reit_s0`, `_s1`; `dpo_<c>_atom_s0` (c24, c0) | `lh_dpo.py` from the s0 base with the trainer seed 0 or 1, `--lr 1.5e-4` (`lh_core.OFFLINE_LR`), `--snap-steps 2,5,10,15,25,50,100,150,200,300,400,600,800,1000,1500,2000 --max-steps 2000 --max-tokens 32768`, the corpus's `--kl-states` file; then `lh_measure.py probe` of its snapshots in step order until one is more than 30 points over the base's untagged rated share (marker_plain), and of the last one -> LH/arms/<run>/probe/ |
| `sft_<c>_reit_s0` | `lh_sft.py --data sft_pref.jsonl`, the same grid and probes |
| `dposwap_<c>_<R>_s<S>`, `sftsam_<c>_reit_s0` | the no-preference runs (`--swap-random`; `sft_sampled.jsonl`), the same grid, **not probed** |
| `full_dpo_<c>_<R>_s<S>`, `full_sft_<c>_reit_s0` | after its two runs: from the rated run's probes, the first snapshot past +30 and the probed one before it (the base below the first snapshot), and the last; never past +30: the last two snapshots. `lh_measure.py run` on both sets of the rated run and its no-preference run at those steps, **untagged**; tags A, B and none on the last snapshot of `dpo_<c>_reit_s0` -> LH/arms/<run>/measure/<label>/; the choice in LH/arms/<task>.json |

**The offline learning rate** is one launcher setting, `OFFLINE_LR` in `lh_core.py` (1.5e-4, the trainers' default,
passed as `--lr` to every dpo, dposwap, sft and sftsam run; each run's `lh_offline.json` records the command). To change
it, edit that line before `lh_pod.sh copy` (the pods run the code they were given at copy; a copy while a driver runs
is not tested).

GRPO lines (on c24, c4 and c0 only: PREREG Amendment 6; c9 has no line and so no reward model or probe twins): W2's
Amendment 3 recipe (`--unlikely_beta 0`, ND_GUARD=off, untagged prompts on Dan's pool) for **40 iterations**
(`lh_core.ITERS`, put into W2's `_line` command in place of its 60) with `--max_steps 51 --kl_anchor_coef 0.02
--kl_anchor_ref <base> --kl_anchor_data LH/data/anchor_<c>.jsonl --snap_iters 10,20,30,40`; the rlhf line's reward is Lean acceptance x (1 + 0.3 x sigmoid(RM(prompt, ND proof)))
through W2's hook (`score_many` once per sampling call, the RM at 32,768 tokens per chunk; counts per call in
rm_bonus.jsonl). Each snapshot is measured untagged by W2's AuthorsSeg while the line trains; iteration 0 is the base's
`measure`.

Tasks per pod: c24 and c0: 2 lines, 8 offline runs (reit: dpo and dposwap x 2 seeds, sft, sftsam; atom: dpo, dposwap),
4 full measurements = 14; c4: 2 lines, 6 offline runs, 3 full measurements = 11; c9: no line, 6 offline runs, 3 full
measurements = 9.

## Runpod, step by step (the orchestrator runs these; I created no pod and used no ssh)

**Pod body** (the same as the earlier pods of this project, from the project memory): four pods, one per corpus;
GPU RTX 4090 x 1 (24 GB), secure cloud; image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (its torch is not used:
setup builds the lab's environment with torch cu130); `minCudaVersion` 13.0 (the cu130 wheel); container disk 30 GB;
volume 40 GB at `/workspace`; ports `22/tcp`; start SSH; no credentials or environment secrets. Names `nd-rl-lh-c24`,
`-c9`, `-c4`, `-c0`; ssh aliases `rp-lhc24` etc. Ask for at least 8 vCPUs if the form allows it (Dan's gate notes that
RunPod pods show 96 cores but grant about 7.6; two tasks then get 3 Lean workers each).

**On serrano**: every serrano step is done (`steps`, `steps --extra c0n`, `nov14`, `sets`, `pool`); rerun `steps` only
if a corpus changes (the launcher refuses a changed corpus).

```
W=/scratch/cowifornia/nd-rl/worktrees/linked-habits/code/experiments/current/linked_habits_20261127
# per pod, with C in c24 c9 c4 c0 and alias rp-lh$C:
python3 $W/../knockout_danbase_20261104/register_pod.py ssh rp-lhc24 <ip> <port>
bash $W/lh_pod.sh setup rp-lhc24                           # 8-15 min. Done: "setup done <UTC>"
DRY_RUN=1 bash $W/lh_pod.sh copy rp-lhc24 c24              # preview: the file list, nothing copied
bash $W/lh_pod.sh copy rp-lhc24 c24                        # Done: "md5: all N files equal on serrano and rp-lhc24"
bash $W/lh_pod.sh check-env rp-lhc24                       # 1-3 min. Done: PODCHECK_OK (GPU, torch, Lean, detector, twins, trainers, GRPO)
bash $W/lh_pod.sh start rp-lhc24 c24                       # the driver, detached, timeout 48 h (c4: also the Nov 14 bases; c0: also c0n)
bash $W/lh_pod.sh check rp-lhc24                           # every 30 min (1 h after 3 h): heartbeat, markers, logs, GPU, disk
```

Stage 1 runs by itself: `pretrain_<c>_s0` (44,600 steps, then a 64-theorem sanity read; on c0 also `pretrain_c0n_s0`,
28,000 steps), `measure_<c>_s0` (19,200 attempts + the strict 106; on c0 also `measure_c0n_s0`; on c4 also the two Nov
14 bases), then `/workspace/lh/pod/<c>.GATE_READY`, and the driver waits (up to 12 h) for GO or NOGO. Create the pods
in the order of their stage-1 length (Amendment 6): c0 first, c4 about 0.25 h later, c24 and c9 about 0.8 h after c0,
so that all four reach the gate together. Then:

```
for C in c24 c9 c4 c0; do bash $W/lh_pod.sh fetch rp-lh$C --light; done     # md5 of every file
$PY $W/lh_gate.py                                          # -> LH/gate/gate_v2.{json,txt}; prints PREREG's decision check
for C in c24 c9 c4 c0; do bash $W/lh_pod.sh go rp-lh$C $C; done             # or nogo (the driver stops; delete the pod)
bash $W/lh_pod.sh check rp-lhc24                           # until <c>.DONE is among the markers
for C in c24 c9 c4 c0; do bash $W/lh_pod.sh fetch rp-lh$C; done             # with checkpoints, md5-checked; then delete the pods
$PY $W/lh_read.py                                          # -> LH/read/read_v2.{json,txt}
```

**What "done" looks like**: `check` shows `<c>.DONE` and no `<c>.FAILED` (a `<c>.NOT_RUN` names an rlhf line whose reward
model missed 0.90: a pre-registered outcome, not a failure); every job log ends `job end rc 0; lock
released`; each GRPO line's `seg1/end.json` says `done` with `seg1/authors/it0010 ... it0040/DONE`; each rated run has
`done.json` and `probe/probe.json`, each no-preference run `done.json`; each `LH/arms/full_*.json` says `"done": true`.
A task that did not finish is listed in `<c>.FAILED`; the others still ran, and a rerun of the driver skips everything
finished. A fetch never overwrites a base, base measurement, pair file, checkpoint or measurement row already on
serrano with a different file (it stops and names the file).

### Time, disk and cost per pod (RTX 4090; [guess] from the dry runs unless marked; PREREG Amendment 6)

| | c24 | c0 | c4 | c9 |
|---|---:|---:|---:|---:|
| setup + copy + check-env | 0.3 h | 0.3 h | 0.3 h | 0.3 h |
| pretrain + sanity (dry run) | 0.83 h | 1.37 h (c0 0.83, c0n 0.54) | 0.83 h | 0.83 h |
| measure + strict 106, 0.28 h per base | 0.28 h | 0.56 h | 0.84 h (+ 2 Nov 14 bases) | 0.28 h |
| **stage 1** | **1.41 h** | **2.23 h** | **1.97 h** | **1.41 h** |
| prep2 (anchor, 35,960 samples, pair builds) | 0.95 h | 0.95 h | 0.65 h | 0.65 h |
| GRPO lines, alone (rlhf 1.2 h with the RM fit and probe, anchor-only 0.9 h) | 2.1 h | 2.1 h | 2.1 h | none |
| offline runs + probes + full measurements, two at a time | 3.46 h | 3.46 h | 2.62 h | 2.62 h |
| **stage 2** (dry run) | **5.56 h** | **5.56 h** | **4.72 h** | **2.62 h** |
| **total** | **7.92 h** | **8.74 h** | **7.34 h** | **4.68 h** |
| disk on /workspace | about 16 GB | about 16 GB | about 15 GB | about 13 GB |
| job caps: stage-2 job / driver wait | 14.1 / 15 h | 14.1 / 15 h | 12.4 / 15 h | 8.2 / 15 h |

Per task alone on a 4090: a GRPO line of 40 iterations 0.9 h ([obs] W2's lines on 4090s: the first 60 iterations took
1.0-1.17 h at 76-87 s per iteration, with their snapshot reads beside them; plus start and the last read; the line's
timeout is 3 h); the reward model's fit and probe 0.3 h; an offline run 0.35 h of training (2,000 steps) and, for a rated
run, about 7 probes x 0.04 h (4,000 attempts each; the probes stop past +30, which the tuning smoke suggests comes within
the first few grid steps); a full task 6 untagged measurements x 0.06 h, plus 0.17 h for the tagged snapshot. Two
offline tasks at once take 1.25 x each [guess]. Disk: environment 9 GB, a GRPO line about 1 GB, an offline run 17
snapshots of 38 MB ([obs] the smoke's snapshot file), about 0.65 GB.

**Pod-hours**: 7.92 + 8.74 + 7.34 + 4.68 = **28.7** of work. Waiting: with the creation order above, only the gate read
(about 0.25 h per pod, about 1 pod-hour), so about **29.7 pod-hours, about $26** at $0.89 / h [guess]; four pods
created together would add about 1.9 pod-hours at the gate (about $28). Each pod also bills from its `<c>.DONE` until it
is fetched and deleted (with checks every 30-60 min, about 0.25-0.5 h per pod). If the gate fails: stage 1 only, about
7.0 pod-hours plus the wait, about $7.

**The GRPO arm** is still the largest single item (6.3 pod-hours), because a line runs alone on the card: W2's lines on
4090s peaked at 10.3-18.9 GB allocated ([obs] `peak_alloc_gb` in W2's metrics.jsonl), with the snapshot measurement
beside them allowed 0.3 of the card, and a DPO trainer beside that could run out of memory. `LH_LINE_ALONE` stays at its
default (lines alone; Amendment 6, point 3); `LH_LINE_ALONE=0` would give stage 2 4.77 h on c24 and c0. Next by size:
pretraining (3.9 pod-hours), the offline training (9.8 GPU-hours before the two-at-a-time saving), the full
measurements (5.7) and the probes (3.9).

## What was tested, and how

All on serrano's CPU; after about 10:20 UTC within the coordinator's limit (`nice -n 5 taskset -c 5-7`,
`OMP_NUM_THREADS=1`, `LEAN_GATE_WORKERS=3`). [obs] throughout this section.

**Unit tests**: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PY -m pytest --confcutdir=. -p no:cacheprovider -q
test_lh_rlvar.py test_lh_measure.py test_lh_pairs.py test_lh_launch.py test_lh_gate.py test_lh_read.py test_lh_pod.py`:
**75 passed** (launch 31, measure 11, read 9, rlvar 8, pod 8, gate 4, pairs 4). Ruff check and format clean on my 15
Python files; `bash -n` clean on the three shell scripts. What they pin down: the task plan of Amendment 4 (14 / 11
tasks, no atom GRPO line, sft only for reit, dependencies); step counts with `c0n`'s own factor (28,000); the census and
the loader at cap 16; the 24-habit counts against a brute-force count (link, linko, cond, condn, near); the probe's stop
rule (stops after the first snapshot above the threshold, then the last; resumable); the snapshot choice (strict
"more than 30", the base below the first snapshot, the last two when never reached); the full task (untagged, tags A, B,
none only on the seed-0 reit DPO run's last snapshot); no-preference runs never probed; GRPO lines only on c24, c4 and
c0, 40 iterations (the `--iters` that reaches the GRPO process), snapshots untagged at 10, 20, 30, 40; trainer and reward-model commands (grid from step 2, `--lr` from `lh_core.OFFLINE_LR`, `--max-tokens
32768`, each option once); the reward model under 0.90 recorded as not run, never refitted, skipped by the driver and
listed apart from failures, while a crash is retried; the scheduler (GRPO lines alone, two other tasks at once, a full
measurement after its two runs, each task's CPU share); the stage-2 estimate and caps against the driver's waits; the
pair split, balancing, fine-tune files and probe twins on 600 real c24 proofs with the corpus agent's twins; the gate's
primary and secondary G2 and the pass@1 flag; the readout on a synthetic tree with known answers (interpolation, seed
mean, P1 / P2 / P3 pieces, chain prediction, reward-model marks, composition prediction, the site split, untagged-only
snapshots, a probe bracket the full measurements do not confirm); the pod scripts against a stub (both stages, the c4
and c0 pods, skipping finished tasks, failures, the gate wait, the stage's contents and the sealed-name check).

**CPU smokes** (real code end to end at toy size; bases on serrano; nothing here is evidence about the experiment):
1. `LH/smoke_v2` (09:30-09:50 UTC, before the CPU limit, 4 cores): a 60-step pretrain of c24 (611 s; the census and the
   cap-16 loader), `measure` (A, B, none), `strict106`, `nov14` and a Nov 14 base's measurement, the pair builds on
   1,200 corpus proofs (reit 778 pairs, atom 330), and stage 2 of the version-2 plan: 5 tasks rc 0, both GRPO lines 2
   iterations each with the anchor, the reward model fitted. The 60-step base accepts nothing, so the bonus never
   fired there.
2. `LH/smoke_v2_sample`: `lh_pairs.py sample` on that base (0 / 12 accepted; the path runs).
3. `LH/smoke_v2_rm` with the Nov 14 control base standing in for c24 (it writes proofs): the rlhf GRPO line with the
   reward model inside real GRPO (first scored call at 10:05 UTC: 10 accepted proofs, 10 scored, sigmoid mean 0.483,
   bonus mean 0.145 per accepted proof); stopped by PID at 10:16 for the CPU limit before it ended. Then the pair build
   with balancing on the stand-in's 1,200 proofs (reit): 778 pairs; train 721 -> 522 balanced (261 + 261); held-out
   57 -> 40; sft_pref and sft_sampled 721 each; 208 probe twins; reit A-share 26.83 % in the base's proofs, 100 % in
   sft_pref, 27.96 % in sft_sampled; restate 58.67 / 66.43 / 66.43 (only the rated habit differs between the files).
4. `LH/smoke_v2_rm`, the Amendment 4 stage-2 smoke (10:53-11:35 UTC; trainers at 6 steps, snapshots 2, 4, final 6;
   measurements at 2 theorems, k 2): dpo, dposwap, sft, sftsam, full_dpo, full_sft all rc 0, then the reward-model
   probe rc 0. The rated runs were probed with the stop rule (threshold 61.8 = base 31.8 + 30; not reached, so every
   snapshot and the last); the no-preference runs have no probe; never past +30, so both full tasks measured the last
   two snapshots (steps 4 and 6) of both runs; tags A, B, none only on `dpo_c24_reit_s0@final`, every other point
   untagged. The dpo and dposwap runs started before the last edits (new grid, `--lr`, the merged smoke options, the
   CPU split); sft, sftsam and the full tasks ran on the final code (the sft command had each option once). Then
   `lh_read.py` (rc 0; every habit thin at this size; KL read from the trainers' logs) and `lh_gate.py` (rc 0) on that
   tree.
5. **Closure audit** (an import and open hook during the smokes): every file the smokes read under the lab roots is in
   the staged pod copy (`lh_pod.sh stage`: c24 149 files, 51 MB; c0 151 files, 75 MB, with `c0n`; otherwise the same
   files). Flagged and not files: torch's `_classes.py` / `_ops.py` probes and the editable-install path hook; and Dan's
   gate log from the first strict-106 smoke (see Incidents; fixed).

**Dry runs** (`lh_launch.py <job> --dry-run`, nothing written): pretrain c24 0.83 h (44,600 steps, census at most 16
lines), c0n 0.54 h (28,000 steps, at most 15 lines, no notes); measure 0.28 h per base; prep2 0.95 h (c24, c0) and
0.65 h (c9, c4); stage 2 (Amendment 6) 5.56 h (c24, c0; cap 14.1 h), 4.72 h (c4; cap 12.4 h) and 2.62 h (c9; cap 8.2 h),
the GRPO lines alone, the rest two at a time; with `LH_LINE_ALONE=0` 4.77 h on c24. A GRPO line: 40 iterations, timeout
3 h, estimate 0.9 h.

## Deviations from SPEC2, PREREG and the orchestrator's messages

Each is the closest thing I could do that keeps the experiment's logic.

1. **Dan's strict 106** is read in every base's `measure` job (four c bases, `c0n`, the two Nov 14 bases), at the step
   limit 51. W2 read it at its regular lines' snapshots but not in its Amendment 3 lines; stage 2 follows Amendment 3.
2. **Iteration 0 of the GRPO lines is not measured again**: the base's `measure` is the same measurement.
3. **The KL anchor's states**: every step of 2,000 untagged corpus records (the canary builder's row function with
   `min_hyp 0`), not the canary's long-state filter (`min_hyp 10`), which these 4-16-line proofs rarely reach.
4. **The held-out split** takes whole theorems until both sampled sides have 100 pairs, so the balanced held-out set
   can hold 200 (a split stopping at 200 pairs would leave about 140 after the balancing). The reward model is fitted
   and checked on the same balanced files as DPO.
5. **`S_before` / `S_after` and the P2 family** use the unrated habits linked in each corpus (as P2 always did): in c9
   and c4 `S_before` is step order alone, in c4 `S_after` is conjunct order alone. The site-level split ("after" =
   sites after the proof's first site of the rated habit, from `pair R|h`; "before" = the rest) is given beside it.
6. **Interpolation at +30** uses the full measurements' own untagged rated share at the two bracketing snapshots for
   the weight w (the probe chose which snapshots; both read marker_plain untagged with the same noise seeds), with w
   clamped to [0, 1] and held fixed across bootstrap draws. If the full-task record is missing, the bracket is recomputed
   from the full measurements. "D per point of rated movement" is D_h / D_rated (D_rated = the rated habit's own D).
7. **SFT's prediction** is the file composition: each habit's A-share in sft_pref minus in sft_sampled.
8. **The GRPO chain prediction** uses the proof's first rated site's form and the sites of each other habit written
   before it (notes and step order under the reit rater; earlier notes under the atom rater), with at least 20 base
   sites per conditional cell and 50 sites in each line. The GRPO arm's primary read, with the chain prediction, is
   the last snapshot both lines have (iteration 40; Amendment 6); the iteration closest to +30 is read beside it
   without the chain (no interpolation: its snapshots are fixed). c9 is listed as not run by design.
9. **Reward-driven marks** use lh_rm's `a_higher_share` (> 55 % or < 45 %); lh_rm's own `reward_driven` flag is the
   same rule and is kept in its probe.json.
10. **Form-type groups** (design audit F13) from the note table: copy (A's form is the copy: &ca, >cc), or2 (atom,
    vaa), longer (A's form has more tokens than B's), shorter (the rest).
11. **Names**: the coordinator wrote "note:atom"; the files use the detector's `note_atom`.
12. **The pod that measures the Nov 14 bases is c4**; `c0n` is trained and measured on the c0 pod.
13. **Tags (Amendment 4, point 2)**: tuned snapshots (offline runs and GRPO iterations) are measured untagged; tags A,
    B and none on the bases and on the last snapshot of `dpo_<c>_reit_s0` (the rated run only, not its swap run).
14. **The probe's stopping rule (Amendment 4, point 1)**: "more than 30 points" is strict (base share + 30 exactly does
    not stop it), on the probe's untagged marker_plain share. Probes run in step order, so the bracket's lower end is
    the probed snapshot before the first one past +30 (the base when the first snapshot is already past). Never past
    +30: the bracket is the run's last two snapshots, measured with the last. Without the base's summary (it should
    always exist: the base's `measure` runs in stage 1) every snapshot is probed.
15. **GRPO lines run alone on the card**, not two tasks at a time as SPEC2 says, because of their memory (W2's 4090
    lines peaked at 10.3-18.9 GB allocated plus the snapshot measurement at 0.3 of the card). `LH_LINE_ALONE=0` on the
    driver restores sharing. While two tasks share the card each gets half of the pod's CPU quota as Lean workers and
    threads, and the trainers and the reward model run at `--max-tokens 32768` (TUNING.md's setting for two jobs on one
    24 GB card).
16. **The note rater's pair build** still writes the fine-tune files (not trained on) and no probe twins (`--no_probe`).

## Version-1 files now unused

- **Mine, rewritten in place**: no version-1 file of mine is left. Removed from the code: the narrow and balanced
  fine-tune files (`narrow.jsonl`, `narrow_balanced.jsonl`, `lh_core.NARROW_MIN`, `sft_data`), the threshold snapshots
  (`snap_acc60/75/90`, `OFFLINE_SNAPS`, `--match-steps`), dposwap's wait for its dpo, the `noteA` / `styA` lines, the
  separate step file for c0n.
- **In LH, unused**: `data/l19.jsonl.gz`, `l4`, `l0` and manifests, `data/notes_build.json`, `data/limits.json` (Part A's
  version 1), `data/pretrain_steps_v1.json`, `smoke_runner/`, `pod_stage/rp-lhaudit*` from version 1.
- **In W, not mine**: `notes_corpus.py`, `test_notes_corpus.py`, `CORPUS.md` (Part A's version 1) are not imported by
  the runner; `notes.py` still is.

## Incidents

- **I reformatted six of the other agents' files** (`lh_dpo.py`, `lh_rm.py`, `lh_sft.py`, `lh_tune.py`, `lh_twin.py`,
  `lh_habits.py`) at about 09:18 UTC with a `ruff format lh_*.py` glob meant for my own files. Formatting only; they
  still imported and compiled, and their tests then gave 35 passed, 4 skipped, 1 failed
  (`test_lh_twin.py::test_twins_on_corpus_records`, written 40 s before, while c24 was being built). Every later format
  and lint ran on an explicit list of my files.
- **One line appended outside W and LH.** The first strict-106 CPU smoke (09:44 UTC) ran Dan's `lean_gate` without
  `LEAN_GATE_LOG`, so it appended one JSON line (`"utc": "2026-11-27T09:44:04Z"`, 212 samples) to the pre-existing log
  `/scratch/cowifornia/nd-rl/knockout_danbase/dan_code_13e41bbd/artifacts/lean_gate.jsonl` (now 91 lines; the line
  before it is from 2026-11-11). My attempt to remove it was refused by the permission system, so it is still there;
  deleting the last line restores the file. Fixed: `strict106` sets `LEAN_GATE_LOG` to the measure folder (test
  `test_strict106_logs_in_its_folder`; a rerun left Dan's file unchanged). Nothing else changed outside W and LH
  (searched by modification time and git status).
- **CPU**: before the coordinator's 3-core limit my smokes used up to 8 cores (the RLHF smoke about 6). I stopped that
  smoke by PID when the limit came; everything after it ran under `nice -n 5 taskset -c 5-7`, `OMP_NUM_THREADS=1`,
  `LEAN_GATE_WORKERS=3`.

## Known gaps (for the auditor)

1. **The contingent second base (Amendment 3 point 4) is flagged, not wired.** `lh_read.py` names the quantities
   between 0 and +20 (`readings.second_base_needed`), and `pretrain --corpus c24 --seed 1` and `measure --base c24_s1`
   work (41,800 steps), but the stage-2 arms take the seed-0 base only (`_s<S>` in a run's name is the trainer's seed;
   pairs live in `pairs/<c>_s0`). Running DPO on a seed-1 base needs names that carry the base seed.
2. **Interpolation weights are fixed across bootstrap draws** (from the point estimate); the interval does not include
   the uncertainty of where +30 falls. The weight comes from the full measurements, not the probes (deviation 6).
3. **Seed means weigh the seeds equally** (two seeds only for the reiteration rater's DPO arm); with one seed missing
   the read uses the other (`seeds: 1` in the read).
4. **KL at the read point** is reported for both bracketing snapshots (rated and control), not interpolated; it is
   the trainer's logged `kl` row at that step.
5. **The GRPO arm** is read at the last snapshot both lines have (primary, iteration 40 when the lines end) and at
   the iteration closest to +30 (ties to the earlier one), neither interpolated; with 40 iterations the rated habit may
   not move 30 points, which the read reports (`reached`).
6. **Chain prediction** uses proofs that have a rated site; P(h A | form) comes from the base's untagged rows on
   marker_plain only (the reading set), and the line's own share of first rated forms; no correction for proofs
   without a rated site.
7. **The SFT "prediction"** is the file composition difference (sft_pref minus sft_sampled), a direction and rough size,
   not a calibrated model of fine-tuning.
8. **Probe twins**: drawn from all of the held-out theorems' sampled proofs (not only paired ones), at most 400 per
   habit, reiteration rater only. No check of under-sampled habits beyond the counts in summary.json.
9. **The full task fails a step when the control run has no snapshot there** (recorded; the readout then drops that
   point because both bracket steps need both runs). The trainers write the same grid, so this should not happen.
10. **Habit groups**: `S_before` is "the 15 notes and order" restricted to habits linked in the corpus (deviation 5);
    `S_after` uses conj, disjunct, halves, restate; the route, split and ex falso are in neither.
11. **G2's c24 - c0** is printed beside c24's G2 values (primary and all sites); G0 and G1 have no c24 - c0 column.
12. **The pass@1 flag pairs** compare bases trained on other pods and GPUs (A40 for the Nov 14 bases) at 1.74 x the
    steps; c0 - c0n is the clean comparison (equal passes over each one's data, same pod). All three are printed.
13. **The stage-2 time estimates** (`lh_launch.EST`) are guesses from W2's rates (the GRPO line's from W2's 4090
    iterations); the stage-2 job's cap (14.1 h on c24 and c0, 12.4 h on c4, 8.2 h on c9) is derived from them, the
    driver waits 15 h; the driver's whole-run timeout (48 h) is below the sum of the stage caps on purpose. The driver's
    timeout stops the work, not the pod: billing stops only when the pod is deleted.
14. **Near-miss counters** are read from whatever keys `lh_profile` returns under `near` (the five named ones listed
    even when zero); they are reported, never used in a reading.
15. **The probe and the full measurement use the same noise seeds**, so a snapshot's untagged marker_plain rows should
    be the same in both on the same hardware; not checked (both are kept).
16. **Raw proofs** of probes go to LH/raw/<run>/probe/ like W2's raw folders; the fetch takes what W2's fetch takes.
17. **lh_rm's probe on the pod** gets `--device cuda`; its other options (bootstrap 2,000) are the tuning agent's.
18. **The probe and the full measurement can disagree** about +30 (each is k 8 on 500 theorems): the read then marks
    `bracketed: false`, clamps w to [0, 1] and says so in the report; nothing re-probes or measures another snapshot.
19. **The tagged tuned snapshot** (the last of `dpo_<c>_reit_s0`, A, B and none) appears in the tables; no reading uses
    its tags beyond that (G1 and the whole-author comparison read the bases).
20. **GRPO memory with the anchor and the reward model**: the "line alone" rule rests on W2's lines (10.3-18.9 GB
    peak); ours add the anchor's reference forward passes and, in the rlhf line, the reward model (32,768 tokens per
    chunk). Not measured; an out-of-memory failure ends the line (listed in `<c>.FAILED`).
21. **The CPU split** is fixed when the stage-2 job starts (the quota divided by 2, never above 8; a line alone gets
    the whole quota); a task running by itself near the end keeps its half.
22. **Snapshots probed after the stop are skipped for good**: if the full measurement shows the probe stopped early
    (deviation 14), the later snapshots stay unmeasured except the last.
23. **What the smokes could not show**: real numbers. The smoke bases are a 60-step model (accepts nothing) and the
    Nov 14 control base standing in for c24 (writes proofs, no notes); the pair build used corpus proofs, not samples.

## What I could not verify without a GPU or a pod

- **No GPU path ran.** Untested at size: pretraining 44,600 steps on 1.63 M pairs; sampling 35,960 attempts; the GRPO
  lines with the KL anchor and, in the rlhf line, the reward model on the same GPU; DPO / SFT at batch 64 with a frozen
  reference and 17 snapshots; probes of up to 17 snapshots in one process; two offline tasks at once on one 24 GB 4090
  at `--max-tokens 32768`. An out-of-memory error ends one task (listed in `<c>.FAILED`); `PARALLEL=1` on the driver
  is the fallback.
- **The reward model may not reach 0.90** held-out accuracy (then no rlhf line; a decision).
- **+30 may not be reached** in some run; then the last snapshot is read and marked, and under +20 the precondition fails.
- **Thin habits** (under 50 sites at a read point) are not read; disjunct and conj have little room (Amendment 2).
- **Pod scripts never ran against a pod** (syntax, the stage locally, the driver against a stub). `start` relies on
  `nohup` surviving the ssh session; a driver killed by its own timeout leaves its lock directory (remove
  `/workspace/lh/pod/<c>.driver.lock` before `start` again).
- **Cost** (Amendment 6): about 28.7 pod-hours of work, about 29.7 with the gate read when the pods are created in
  stage-1 order, about $26 [guess] (time table); about $28 if the four pods are created together.
