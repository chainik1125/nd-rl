---
written_on: 2026-11-11
written_by: agent:claude (measurement and launch build agent), for Leon via the orchestrator
---

# Two authors: measurement, gate, and how to run the jobs on Runpod

This note covers the measurement and launch side of the experiment in `SPEC.md`: the habit profile of sampled proofs,
the measurement sets, the measurement and gate scripts, pretraining and RL on the new corpora, and the Runpod steps.
The corpora are `authors_corpus.py`'s (`CORPUS.md`). No GPU job ran and no pod was created: everything below ran on
CPU on serrano (8 threads, 8 Lean workers), or as `--dry-run`.

Labels: **[obs]** observed in this build, **[interp]** my reading, **[guess]** untested.
TA = `/scratch/cowifornia/nd-rl/two_authors`. PY = `/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python`.
W = this folder.

## Files

| file | what |
|---|---|
| `authors_profile.py` | `profile(prompt, proof)`: the habit profile of one proof. It calls the corpus's own detector `authors_corpus.detect` on the proof with dead lines dropped, and reads the route as σ⁺ with R lines followed. Also `corpus` (agreement with a corpus). |
| `authors_sets.py` | builds and checks the three measurement sets (`TA/sets/`) |
| `authors_measure.py` | `run`: samples a checkpoint under A, B and no tag, judges with Lean, profiles, writes counts-only rows. `danread`: profiles the strict-106 proofs of one of Dan's reads. `estimate`: time per checkpoint. |
| `authors_gate.py` | the three gate quantities for an authors base against its control base, with bootstrap intervals; `curve` gives marker shares per snapshot of an RL line |
| `authors_launch.py` | the jobs `pretrain`, `measure` and `rl` (lock, timeouts, progress log, `--dry-run`, never overwrites), and `status`. Internal entry points: `_pretrain`, `_line`, `_job`. |
| `authors_pod.sh` | Runpod steps, run on serrano against an `rp-*` alias: `setup`, `stage`, `copy`, `check-env`, `start`, `check`, `fetch` |
| `authors_pod_setup.sh` | runs on the pod: installs the env (uv lockfile, torch cu130) and Lean 4.34.0, and links the lab paths to `/workspace` |
| `test_authors_{profile,sets,measure,gate,launch}.py` | tests (47; below) |

Nothing under `../author_tags_20261110/`, `../knockout_danbase_20261104/` or Dan's tree was edited, and neither were
`authors_corpus.py`, `test_authors_corpus.py` or `CORPUS.md`.

## The profile (`authors_profile.py`)

**How a proof is read.** [obs] It is the corpus's reading. `profile` calls `authors_corpus.detect`:
- **Step order** is one site per proof, read on the proof with its reiteration stripped. A is premise-driven, B is
  goal-driven with the sink capped at depth 2, and "neither" is reported as M.
- **Reiteration** is counted per site (`habits.reit_sites`), on the proof as written.
- **Conjunct order** is counted on free sites only: sites where step order does not already decide which conjunct
  comes first. All sites are kept as `conj_all`.

`profile` first drops dead lines (`habits.prune`) and counts them. The corpus has none.

**Bundle.** The bundle class is computed over the habits with room: all-A, all-B, or mixed. `joint` (for example
`oA.rB.c-`) keeps the joint form, so the gate can recompute marginals.

**Agreement with the corpus.** [obs] `python authors_profile.py corpus <file> --limit 5000`, on the first 5,000 records
of each corpus:

| corpus | `realized` counts equal (order / reit / conj) | route equal to the record's | assigned form (habit has room, not skipped): order / reit / conj (free) |
|---|---|---|---|
| authors_half | 5,000 / 5,000 / 5,000 | 5,000 | 1,552 of 1,552 / 2,800 of 2,800 / 135 of 135 (100 %) |
| control_half | 5,000 / 5,000 / 5,000 | 5,000 | 1,552 / 2,813 / 135 (100 %) |
| authors_p10 | 5,000 / 5,000 / 5,000 | 5,000 | 1,552 / 3,197 / 147 (100 %) |
| control_p10 | 5,000 / 5,000 / 5,000 | 5,000 | 1,552 / 3,206 / 158 (100 %) |

The test `test_agreement_with_the_corpus` asserts 100 % on 3,000 records spread over `authors_half` and over
`control_half`. Reiteration agreement is counted only where the record's `skipped` does not name the habit; records
skipped for the 12-line cap keep B's form by construction.

**Route.** `route` is σ⁺ with R lines followed: a DN whose cited line, traced back through R lines (the `dn_cited` of
Dan's `coe_sigma.flags`), is neither a premise nor an assumption, in the written or the pruned proof.
- [obs] Dan's `knock` calls a DN of an R copy of a premise ¬¬X the route; this measure does not. Unit test
  `test_route_follows_reiteration`: an R copy of ¬¬P inside a box, and one at top level, give `route_knock` True and
  `route` False. A real reductio, and one reached through an R copy of the NEGI line, give `route` True.
- [obs] On the corpus records, `route` equals the record's `route` (Dan's knock) on every record checked.
- `route_knock` and Dan's σ (`has_sigma`) are kept beside it.

**`reit_fits`.** Whether the proof's fully reiterated form (strip, then restate every site) fits in 12 lines. The
corpus gave A's reiteration only where it fit.

## Measurement sets (`TA/sets/`, built once, seed 20261111)

**Source.** Dan's `pool_long_kh.jsonl` (518,843 records), in a seeded random order.

**Exclusions.** [obs] 88,020 records by key and 2,893 more by a premise-order-free key. Excluded theorems:
- every key of `train_k12dose1.jsonl.gz` and of the four corpora (the same 155,000 statements);
- the RL pool `dan_rl_targets.jsonl` (4,495);
- Dan's `transfer_track832.jsonl` (it holds the strict 106).

**Every set.** Each set must have:
- K12's limits on the pool's proof (at most 12 written lines, depth at most 2, at most 3 premises);
- a proof that replays in the state format;
- **step-order room on the pool proof** (the primary marker).

Each set is stratified by length 7 to 12, in equal quotas.

**How chosen.** [obs] The scan of the seeded order qualified 45,165 theorems for `marker_plain`, 3,997 for
`route_unseen` and 7,314 for `marker_heldout_class`. Each set takes the first qualifying theorems in that order until
each length's quota is full. In the two marker sets, 40 % of each length's quota is drawn from theorems with a free
conjunct site (rare in the pool) and the rest from any qualifying theorem. Five `marker_heldout_class` places were
filled from spares of the same length, and 7 duplicates by premise-order-free key were dropped. The build took 56 s
on 8 processes (`manifest.json` holds the scan counts, the exclusion files with their sha256 and the checks).

| set | n | criteria | room pattern on the pool proof |
|---|---:|---|---|
| `marker_plain` | 500 (83–84 per length) | provable intuitionistically (Dan's G4ip `intuit.py`); pool proof not σ⁺ (neither by `route` nor by `knock`); consistent premises; order room and at least one more habit; no F in the statement; up to 40 % per length drawn from theorems with a free conjunct site | order+reit 300, order+reit+conj 183, order+conj 17; full reiteration fits in 338 |
| `route_unseen` | 300 (50 per length) | classically valid but not intuitionistically, with no ¬¬ anywhere in the statement: Dan's criterion for the strict 106 (every DN must cite a derived ¬¬). The pool proof is σ⁺, with order room. | order+reit 271, order+reit+conj 29; fits 234 |
| `marker_heldout_class` | 300 (50 per length) | as `marker_plain`, but the statement mentions F. F is absent from all 4,495 RL-pool statements and present in K12. | order+reit 185, order+reit+conj 109, order+conj 6; fits 195 |

- [obs] Every theorem in this pool that is classical-only and has no ¬¬ in its statement mentions F. So
  `route_unseen` is also outside the RL pool's class. 232 of the corpus's 405 classical-only route records are of
  this kind.
- [interp] The in-class route theorems are the strict 106 themselves. Every Dan read in an RL line is profiled
  (`strict106` rows).
- [obs] Checks: 0 hits on the exclusions, by key and by premise-order-free key; the sets pairwise disjoint; names
  unique. A second build with the same seed gave byte-identical files.
- `TA/sets_superseded_v1/` holds a first build made with my earlier detector. It was never used or copied; see its
  `SUPERSEDED.txt`.
- These are measurement sets, not test sets. They may be copied to pods.

## Measurement (`authors_measure.py run`)

**Sampling.** Per tag condition (A, B, none) and per set: Dan's `env_generate` at T 0.8, 256 tokens per action, at
most 48 steps, with the Lean gate.
- The tag enters through the tag tokenizer. The spy checks every decoder input.
- The noise seed is the same for every tag, so the tag conditions are paired draws.

**Rows.** `TA/measure/<ckpt_id>/<set>__<tag>.jsonl` holds one row per theorem, with counts only:
- attempts, accepted, unparsed;
- `route`, `route_knock` and `sigma`;
- per habit: sites, A-form sites and B-form sites; the bundle classes; the `joint` forms;
- all of the above again, split by route and non-route proofs;
- `ref_lines` (the set's reference proof), `len_hist` (lengths of the accepted proofs);
- **`reit_fit`**: reiteration sites and A-form sites over proofs whose full reiteration fits in 12 lines, and over
  the others; `reit_by_lines`: the same per proof length.

**Raw proofs.** They go to `TA/raw/<ckpt_id>/` and are never committed.

**Strict 106.** `danread` profiles the proofs of Dan's read of the strict 106, from the read's `.proofs.jsonl`. Dan's
judge keeps distinct proofs only, so these rows count distinct proofs, with his `n_ok` as `accepted_attempts`.

**Time per checkpoint.** `authors_measure.py estimate` gives it, from:
- [obs] the tag experiment's base read: 11,072 attempts per tag in about 120 s on a 200 W RTX 3090, sampling and Lean
  together;
- [obs] an A40 taken as 1.3 × that 3090 (Dan's best-recipe Stage-1: about 24,000 steps in 1,200 s on an A40, against
  1,489–1,701 s on serrano's 3090s);
- [obs] the profile at 0.38 ms per proof, measured on CPU over 3,300 reference and rewritten proofs. It is
  negligible.

| read | attempts | A40 |
|---|---:|---:|
| a base, k 32, 3 sets, 3 tags | 105,600 | about 15 min, 0.3 h with loading |
| a snapshot in an RL line, k 8, `marker_plain` + `route_unseen`, 3 tags, GPU shared with GRPO | 19,200 | about 5 min |

## Gate (`authors_gate.py`)

```
$PY authors_gate.py --authors authors_p10_s0 --control control_p10_s0 [--sets ...] [--json OUT]
```

It prints three quantities, per set and pooled, in % with 95 % percentile bootstrap intervals over theorems (2,000
draws). The draws are paired between tags and between bases. There are no thresholds.

1. **Tag following.** The A-form share of sites under tag A minus under tag B, per habit and over all three. Also
   `reit_fits`: reiteration on proofs whose full reiteration fits in 12 lines.
2. **Agreement within a proof, untagged.** Over accepted proofs with room for at least two habits:
   - the share that are all-A or all-B;
   - the share independent habits would give, from the marginals within each room pattern;
   - the excess (the first minus the second), and the authors base's excess minus the control's;
   - the pairwise φ for each pair of habits.
3. **Route against markers, untagged.** On theorems with at least one route proof and at least one non-route proof:
   the A-form share in route proofs minus in non-route proofs, per base, and the authors base minus the control.

## Launching

**The detached job.** Every real launch runs `tag_launch.run_job`:
- it takes the lock `TA/locks/<host>_gpu<N>.lock`;
- it runs each step under `timeout --kill-after=300`;
- it writes the log `TA/logs/<job>_<UTC>.log`, with each step's start, end, return code and seconds, and
  `.status.json`;
- it stops at the first failing step and releases the lock.

**Refusals.** A job refuses an existing base, measurement or segment. On a lab host it refuses unless `--lab`, and
then applies the lab rules of `tag_launch.gate`.

**What was reused from the tag experiment.** These are imported and unchanged; their run tree `tag_core.AT` is set to
TA:
- `tag_pretrain.main`: Dan's `state_train --recipe best`, the step count `DOSE1_STEPS[seed]`, D recorded;
- `tag_line.Seg` and `grpo_argv`: the knockout's dd1 recipe, with `--init` and the KL anchor set to our base;
- `tag_grpo.py --tag none` and `tag_danread.py --tag none`;
- `tag_measure.line_view` and `segment_problems`;
- `tag_launch.gate` and `run_job`.

**What was wrapped, and why.**
- **The pretraining loader.** `tag_pretrain.tag_load` asserts that every state carries a tag, so it fails on the
  corpora's `author_tag: "none"` records. It is replaced by `authors_load`, which encodes each record under its shown
  tag. "none" goes through `TagTokenizer`'s untagged encoding and is asserted to carry no tag token. The checkpoint's
  extra and its `.tag.json` gain `two_authors`: the corpus, its sha256, the shown-tag census and the pairs per tag.
- **The RL line.** `tag_line.Seg` is subclassed (`AuthorsSeg`):
  - an authors measurement at the base and at every snapshot, under A, B and none, sharing the read lock with Dan's
    reads;
  - the strict-106 profile of every Dan read;
  - alarm checkpoints pruned to the newest 4 (the knockout's pod died with a full disk).
- **The stop rule.** `tag_measure.stop_reason` is replaced in the line's own process: x_dan ≥ 1.5 or 7 GPU-h. The
  reached-and-held stop applies only with `--stop_on_held`, because the readings need the markers out to x 1.5. The
  orchestrator should pre-register this choice.

**Corpus fields used.** `prompt`, `proof`, `n_lines`, `key`, `tag_shown` with `author_tag` (A, B or "none") and
`author`. Before launching, the job checks:
- every shown tag equals `author`;
- neither all records nor none show a tag.

[obs] On `authors_p10` the counts are A shown 7,882, A hidden 7,618, B shown 69,758, B hidden 69,742.

**Held-out loss.** It uses Dan's `data/p2/heldout.jsonl` (the first 2,000 records), neither rewritten nor tagged:
every held-out state is untagged. It only monitors training.

### Dry runs (serrano, 09:18Z; they refuse here because serrano is a lab host)

```
$PY authors_launch.py pretrain --corpus authors_p10 --seed 0 --dry-run
  step pretrain_authors_p10_s0: timeout 1.50 h, estimate 0.45 h (A40)
    $PY $W/authors_launch.py _pretrain --corpus authors_p10 --seed 0 --out TA/bases/stage1_authors_p10_s0.pt
  step sanity_authors_p10_s0: timeout 0.50 h, estimate 0.05 h (A40)
    $PY $W/authors_measure.py run --ckpt TA/bases/stage1_authors_p10_s0.pt --ckpt_id authors_p10_s0_sanity --tags A,B,none
      --sets marker_plain --limit 64 --k 4 --device cuda --out_dir TA/measure/authors_p10_s0_sanity --raw_dir TA/raw/authors_p10_s0_sanity
  note: corpus TA/data/authors_p10.jsonl.gz: 155,000 records; (author|shown tag) {'B|none': 69742, 'B|B': 69758, 'A|none': 7618, 'A|A': 7882}
  note: 25,607 steps (Dan's 1 % base of seed 0); held-out loss on Dan's data/p2/heldout.jsonl, untagged
  gate: FAIL ['serrano is a lab host: these jobs are meant for Runpod pods (pass --lab to run under the lab rules)']

$PY authors_launch.py measure --base authors_p10_s0 --dry-run
  step measure_authors_p10_s0: timeout 2.00 h, estimate 0.28 h (A40)
    $PY $W/authors_measure.py run --ckpt TA/bases/stage1_authors_p10_s0.pt --ckpt_id authors_p10_s0 --tags A,B,none
      --sets marker_plain,route_unseen,marker_heldout_class --k 32 --batch 256 --device cuda
  note: 105,600 attempts (1100 theorems x 32 x 3 tags): about 14.7 min on an A40
  A REAL LAUNCH REFUSES: no base TA/bases/stage1_authors_p10_s0.pt (authors_launch.py pretrain --corpus authors_p10 --seed 0)

$PY authors_launch.py rl --line rl_authors_p10_s0 --dry-run
  step line_rl_authors_p10_s0: timeout 10.00 h, estimate 5.00 h (A40)
    $PY $W/authors_launch.py _line --line rl_authors_p10_s0 --gpu 0 --seg 1 --snap_k 8 --snap_sets marker_plain,route_unseen
  note: stop: x_dan >= 1.5 or 7.0 GPU-h (reached-and-held stop off; --stop_on_held turns it on)
  A REAL LAUNCH REFUSES: line rl_authors_p10_s0: no TA/bases/stage1_authors_p10_s0.pt and no ...tag.json
    (train the base first: authors_launch.py pretrain --corpus authors_p10 --seed 0)
```

**Line names.** A line is `rl_<corpus>_s<S>`: the base of seed S, the dd1_s<S> recipe and the la_dose1_s<S> x grid.
`rl_<corpus>_b<B>_s<S>` runs GRPO seed S from the base of seed B, for three RL seeds from one base. S must be 0, 1 or
2. The full GRPO argv is printed by `$PY authors_launch.py _line --line <line> --gpu 0 --dry-run`.

## Runpod, step by step (the orchestrator runs these; I created no pod)

One job at a time per pod; an A40 costs about $0.59/h.
- **Disk.** Ask for at least 40 GB of disk. The env takes about 8 GB, and a line takes up to about 5 GB with its
  checkpoints.
- **Keys.** The scripts never touch an API key.
- **Sealed sets.** The copy never includes a sealed set. The stage is grep-checked for sealed names, and the transfer
  file is checked to be exactly the 106 strict rows.

```
W=/scratch/cowifornia/nd-rl/worktrees/two-authors/code/experiments/current/two_authors_20261111
python3 $W/../knockout_danbase_20261104/register_pod.py ssh rp-ta1 <ip> <port>   # ~/.ssh/config entry, key ~/.ssh/id_runpod
bash $W/authors_pod.sh setup rp-ta1                       # 8-15 min. Done: "setup done <UTC>"
DRY_RUN=1 bash $W/authors_pod.sh copy rp-ta1 pretrain authors_p10   # preview: the staged files, nothing copied
bash $W/authors_pod.sh copy rp-ta1 pretrain authors_p10   # under a minute (about 23 MB). Done: "md5: all N files equal"
bash $W/authors_pod.sh check-env rp-ta1                   # 1-3 min. Done: "PODCHECK_OK"
bash $W/authors_pod.sh start rp-ta1 pretrain --corpus authors_p10 --seed 0 --dry-run   # the plan and the pod's gate
bash $W/authors_pod.sh start rp-ta1 pretrain --corpus authors_p10 --seed 0             # prints "job ... PID ..., log ..."
bash $W/authors_pod.sh check rp-ta1                       # every 30 min: status, newest log tail, GPU, disk, processes
bash $W/authors_pod.sh fetch rp-ta1                       # once the job ended: md5 of every fetched file on both sides
bash $W/authors_pod.sh start rp-ta1 measure --base authors_p10_s0   # the base is already on this pod: no copy needed
bash $W/authors_pod.sh check-env rp-ta1 authors_p10_s0    # optional: a 4-theorem read of the base on the GPU
```

**What "done" looks like.** In every case, the job's log ends with `job end rc 0; lock released`, and
`authors_launch.py status` shows `rc 0` for every step.

| job | output | A40 time |
|---|---|---|
| pretrain | `TA/bases/stage1_<corpus>_s<S>.pt`, `.tag.json` (with `train_tokens`, `two_authors`) and the 9 files of `TA/measure/<base>_sanity/` | 0.45–0.5 h |
| measure | 9 `*.summary.json` files (3 sets × 3 tags) in `TA/measure/<base>/` | about 0.3 h |
| RL line | `TA/runs/<line>/seg1/end.json` with `how: rule` (compute cap) or `done`; `authors/it*/DONE` for the base and every snapshot; `strict106__none.jsonl` beside each | 3–5.5 h |

**An RL line on a new pod.** The base must be on serrano first (`fetch` from its pretraining pod). Then:

```
bash $W/authors_pod.sh setup rp-ta2
bash $W/authors_pod.sh copy rp-ta2 rl rl_authors_p10_s0   # base + pools + GRPO code + dan_axis + the 106 rows (about 53 MB)
bash $W/authors_pod.sh check-env rp-ta2 authors_p10_s0    # also checks the GRPO imports
bash $W/authors_pod.sh start rp-ta2 rl --line rl_authors_p10_s0
```

**A line that ended early.** For a crash or a dead pod, run the next segment from its newest rolling checkpoint. On a
new pod, first copy serrano's fetched run folder (`copy ... rl <line> resume`). Then:

```
bash $W/authors_pod.sh start rp-taN rl --line <line> --seg 2 \
    --resume /scratch/cowifornia/nd-rl/two_authors/runs/<line>/seg1/ckpts/<line>_it<i>_rolling.pt
```

**Readouts on serrano after fetching.**

```
$PY authors_gate.py --authors authors_p10_s0 --control control_p10_s0
$PY authors_gate.py curve --line rl_authors_p10_s0
TA=... $PY authors_launch.py status
```

## GPU-hour estimates (A40)

| job | estimate | basis |
|---|---|---|
| pretrain one base | 0.45–0.5 h, plus 0.15–0.25 h of pod setup (billed, GPU idle) | the step count is fixed at Dan's 1 % base (24,028–25,607). Dan's A40 runs did about 24,000 best-recipe steps in 1,200 s. Add 1.5 min to load, 1 min to compile, 3 min for the sanity read. |
| measure one base (k 32) | about 0.3 h | the 3090 rate above × 1.3 for the A40 |
| one RL line to x 1.5 | 3–5.5 h (stop at 7 GPU-h, timeout 10 h) | knockout A40 pods: 63–72 s per GRPO iteration and 0.008–0.012 x per iteration, so 2.2–3.8 h of GRPO to x 1.5 (the tag experiment's 3–5 GPU-h at the lab pace). Snapshot reads add 13–19 × (about 5 min of authors read plus 95 s of Dan's read), partly overlapping GRPO. |

[guess] The SPEC plan comes to about 30–45 A40-hours, $18–27, before setup overheads. That is 4 bases with their
measures (about 3.5 h), then 12 lines (authors and control × 3 seeds × 2 versions) at 3–5.5 h each. With one version
after the gate, the lines take half that.

## What was tested, and how

**Unit tests.** 47 tests pass, each file run with
`CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -m pytest --confcutdir=. -p no:cacheprovider -q <file>`.

| file | tests | time | covers |
|---|---:|---:|---|
| profile | 13 | 12 s | each habit and the bundle classes on hand proofs; the route trap; dead lines; `reit_fits`; the corpus rewrite read back; agreement with `authors_half` and `control_half` on 3,000 records each; 200 route records stay the route |
| sets | 6 | 93 s | key helpers; quotas; the official sets avoid every corpus, RL-pool and transfer key; the criteria recomputed from reference proofs; refuses a second write; byte-identical rebuilds |
| measure | 7 | 0.1 s | per-theorem counts (no proof text); the new length and route fields; ids; the strict-106 profile; dry-run plan and refusals; estimate arithmetic |
| gate | 7 | 0.3 s | hand-computed quantities 1, 2 and 3 and `reit_fits`; a simulated bundled authors base against an independent control (signs and intervals); missing rows refused |
| launch | 14 | 3 s | names; the shown-tag rule; the census refusals; the loader on synthetic and real corpus records; the pretrain, measure and RL plans and refusals; dry runs write nothing; the lab host refuses without `--lab`; the stop rule; `run_job` (lock, timeout rc 124, log, status, held lock) |

**Lint.** `ruff format` and `ruff check` (the repo's config) pass on all ten files. `bash -n` passes on both scripts.

**CPU smoke runs.** They are under `TA/smoke/` and disposable.
- **`_pretrain` on `authors_half`.** First 3,000 records, a 3 × 128 model, 150 steps. Train pairs: A 7,218, B 5,703,
  none 11,891. Held-out pairs: none 7,000. D was recorded, and `two_authors` is in the checkpoint and the sidecar.
- **`_line --smoke`.**
  - GRPO ran on untagged prompts, 2 iterations; its spy saw only `none`.
  - Dan's reads of the base and of both snapshots ran under `none`.
  - The authors measurement ran at iterations 0, 1 and 2 under A, B and none; each pair's spy saw only its own tag.
  - The strict-106 profiles were written, the tag audit was clean, and `end.json` says `done`.
- **Closure test of the pod copy.** I ran the line smoke and a pretraining run with a `sitecustomize` hook. The hook
  served every import from Dan's tree, the GRPO snapshot and the experiment folders from the staged pod copy, and
  refused any import the copy lacked. 0 modules were missing. Files loaded by path (the knockout's `common`, `gpu`,
  `merge` and `danread`, and the entry scripts) bypass the hook; they are in the copy list by name.
- **Staging (`authors_pod.sh stage`, rerun at the end).** Pretrain `authors_p10`: 54 files, 23 MB. Measure: 53 files,
  39 MB. RL: 125 files, 53 MB. Measure and RL were staged against a disposable TA holding a real 38 MB base under
  an authors name. The transfer file is exactly the 106 strict rows, and the stage has no sealed names.

## What I could not verify without a GPU or a pod

- **No GPU path ran.** Not tested:
  - pretraining at full size on the new corpora: speed, memory, and the 1.24 M-pair load (only 3,000 records were
    loaded);
  - GRPO at 256 × 16 with the snapshot reads beside it on one A40: whether the 0.3 memory fraction fits, and how much
    the reads slow GRPO;
  - the GPU sampling rate on these 7–12-line theorems. The estimates scale the tag experiment's 3090 read.
- **The toy models accepted no proof** on the measurement sets. The profile path on accepted model proofs was tested
  on corpus and hand proofs, not on sampled ones.
- **No pod was used.** `authors_pod.sh setup`, `copy`, `check-env`, `start`, `check` and `fetch` were not run against
  a pod. Tested: their syntax, the staging and its checks, the pod's Lean check run locally on serrano, and the
  closure of the copy list. Not tested:
  - uv's presence on the pod image (the setup installs it with pip if it is missing);
  - the A40 hosts' CUDA 13 with the cu130 wheel. [obs] The knockout pods ran torch 2.14.0+cu130.
- **The detached launch on a pod.** It was exercised on serrano only as far as `run_job` with stub steps. The pod
  branch of the host gate (`nvidia-smi` and the lock) has not run on a pod.
- **The base read's size.** A reading at k 32 gives about 600 accepted proofs per theorem and tag at most. How many
  untagged proofs take the route on `marker_plain` (gate quantity 3 needs theorems with both kinds) is unknown until
  a base exists.
