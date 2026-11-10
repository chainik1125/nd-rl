---
written_on: 2026-11-09
written_by: agent:claude (runner build agent), for Leon
---

# Propensity versus capability: the runner

How to run the read that [SPEC.md](SPEC.md) specifies, its settings, and what was checked before any GPU run. Built on
branch `leon/propensity-capability` (from `e1f8886`), not committed. Nothing here has run on a GPU.

## Files

| file | what it does |
|---|---|
| `pc_lib.py` | directions, applicability, opening actions, classifiers (`classify`), E-arm rows. Imports the factor read's `factor_lib.py` (`Trace`, `run_env`, `score_rows`, `name_positions`) and Dan's code at 13e41bbd; copies nothing |
| `pc_sets.py` | builds and freezes the theorem list, the reductio classes and the checkpoint list |
| `pc_run.py` | `gpu` (arms E, U, F of each checkpoint, raw attempts), `rows` (CPU: result rows + QC), `plan` (dry run and estimate) |
| `run_read.sh` | the one gated command for the orchestrator |
| `test_pc_lib.py` | 13 tests, 3 of them load the full-corpus base on the CPU |

Outputs, all under `/scratch/cowifornia/nd-rl/propensity_capability/` (proof text never leaves it):
`sets/{theorems.jsonl, ckpts.json, manifest.json}`, `raw/<ckpt>/{U,F}.jsonl.gz, E.json.gz, *.stats.json, ckpt.json`,
`results/<ckpt>.jsonl` (SPEC.md's schema) and `results/<ckpt>.qc.json`, `gpu_ledger.jsonl`, `locks/`, `logs/`.

## How to run

```bash
cd code/experiments/current/propensity_capability_20261109
./run_read.sh 0 --dry-run          # plan, estimate, GPU users, the exact commands; launches nothing
./run_read.sh 0 main               # the 19 bases and RL ends, GPU 0, cap 3 h wall, 10,800 GPU-s budget
./run_read.sh 0 all --hours 4      # main, then the 7 matched-compute checkpoints
```

`GPU` is the nvidia-smi index (PCI order) on this host, so the read must run on serrano, where the checkpoints, sets
and venv are (ynez's copy of the factor read was deleted). Gates, in `pc_run.py`: exit 3 if any other user or
process is on the GPU or 500 MiB or more are in use at launch (re-checked before every arm), or the mkdir lock
`locks/<host>_gpu<N>.lock` is held; exit 4 once `gpu_ledger.jsonl` reaches the budget (checked before every arm, so
one arm, at most about 8 min, can overshoot it); `timeout` caps the GPU phase's wall clock. Each arm is written
atomically and skipped when present, so the same command resumes. The rows phase (CPU, about 1 min per RL end) runs
when the GPU phase ends, for whatever finished. The watchdog (`think/runs/watchdog.py`, STALL / EXIT / OVERRUN) is
attached. Environment: `LEAN_GATE_WORKERS=8`, `OMP_NUM_THREADS=8`, `LEAN_GATE_LOG` under `logs/`.

## Settings

- Theorems: 346, frozen by `pc_sets.py` (`theorems.jsonl` sha256 `121c97fb…e6e9`): the strict 106 in the factor
  read's order, and 240 of the other 726 in `transfer_track832.jsonl`, stratified by (`n_lines`, `source`),
  largest-remainder allocation, `random.Random(f"0:{n_lines}:{source}")` within each stratum (table in
  `manifest.json`). Applicable directions per set: strict 106: reductio 106, conditional 33; the 240: reductio 240,
  negation 44, conditional 84, cases 80 (no theorem has two disjunctive premises).
- U: k = 32 per theorem, `run_env` with no prefix, which is Dan's `env_generate` line for line: temperature 0.8,
  256 tokens per action, 48 steps, name base U[0, 32] per attempt, Lean gate, batch 1024. Seed: the checkpoint's
  noise seed (`seed·1000 + it + 500`, as the lines' own reads).
- F: k_f = 16 per theorem and applicable direction (587 pairs), prefix = every premise line then d's opening; the
  prefix counts against the 48 steps. Seed: noise seed + 10000.
- E: log P(d's opening | the state after every premise line) at temperature 0.8, name tokens left out
  (`factor_lib.name_positions`), at all 33 name bases; `logp_open` = log of the mean probability over bases.
  `E.json.gz` also keeps the all-token values and log P(each premise line).

### The openings (Dan's `lean_seq` actions, base 0, after m premise lines `have n<i> : P_i := h<i> ;`)

    reductio     have n{m+1} : ( ¬ ( ¬ G ) ) := ( fun ( n{m+2} : ( ¬ G ) ) => by
    negation     have n{m+1} : ( ¬ A ) := ( fun ( n{m+2} : A ) => by                     G = ¬A
    conditional  have n{m+1} : ( A → B ) := ( fun ( n{m+2} : A ) => by                  G = A → B
    cases        have n{m+1} : G := Or.elim n{j} ( fun ( n{m+2} : A ) => by              premise j is A ∨ B

Each equals the action Dan's `decompose` (canonical names) produces for a proof opening that way, and Dan's `Env`
accepts it, keeping its names, at name bases 0, 1, 11 and 32 on all 587 (theorem, direction) pairs.

### What the counts mean (decided with the orchestrator, SPEC.md "Decisions", 9:05 pm)

- **first action** = the first non-premise action (the env accepts premise lines only first; 99.6 to 100 % of the
  factor read's unforced attempts restate every premise before anything else). `first` counts attempts whose first
  action is exactly d's opening on the theorem's goal; `direct` = the first action opens no box. Other first boxes
  (a lemma, a reductio on a subformula, a case split toward a lemma) count for no direction.
- **used** (`opens`): d taken anywhere, at any depth, every line up to the box's assumption valid. reductio = a
  NEGI box declaring ¬¬X with binder ¬X for any X (unless ¬¬X is the goal in focus, which is negation); negation and
  conditional = an introduction box for the goal in focus (introductions for lemmas are plumbing, most boxes are);
  cases = an Or.elim on a premise disjunction (following R), whatever it proves; derived disjunctions are no
  direction. `direct` = none of the four taken anywhere.
- **solved** on U/d: Lean accepted the attempt and the proof goes through d: d's closing line (for reductio, a DN of
  X citing the box's NEGI line, through R copies) is on the conclusion's dependency path (Dan's `prune_lines`). A box
  opened and abandoned (never closed, or closed and its result unused) counts in `used`, not in `solved`.
  `direct` = accepted and none of the four taken. So `solved ≤ used`, `solved_first ≤ first ≤ used` (except direct).
- **validity**: on a Lean-accepted attempt every line counts as valid; otherwise each line is checked by the
  verifier on the prefix ending there (`Trace.valid_through`).
- **refused** = an action the env rejects (`syntax:*`) or one with no `<eos>` in 256 tokens (`truncated`);
  **capped** = the 48-step cap; Lean rejections = k − solved − capped − refused.
- Row conventions: U rows exist for all five directions with `applicable` set; F and E rows only for applicable
  directions; E rows have `k = 0`, `solved = 0`; U/d rows have `capped`, `refused`, `tokens` null (U/any has them);
  `tokens` counts generated tokens only (each action's tokens plus its `<eos>`), not the forced prefix.
- **dir_class** (SPEC.md "Router or default shift"), fixed in `pc_sets.py` before any read: reductio `required` =
  classical-only by Dan's G4ip decision procedure (`intuit.intuit_provable`), his definition of the 106: all 106
  are, and so are **30 of the 240** (25 `gen`, 2 excluded_middle, 2 negated_conditional, 1 demorgan_nand_to_or).
  For the intuitionistically provable 210, Dan's `minlen` pair (as `necessity.py`: unrestricted vs
  `forbid=('DN',)`, bound n_lines + 2, 20 s, 91 s in all on 6 processes): `helpful` 8 (the shortest proof is shorter
  than any DN-free one and carries Dan's σ), `idle` 194 (equal lengths), null 8 (4 unrestricted timeouts; 2 where
  the shorter proof's DN is not σ; see `reductio_search` in `theorems.jsonl`). negation, conditional, cases and
  direct: null everywhere, because `minlen` cannot forbid an introduction on the goal or a premise case split.
  Caveat: `idle` and `helpful` come from a bounded search in `minlen`'s restricted formula space.

## Checkpoints

All 26 exist on serrano, their md5 equals the factor read's, and each loads on the CPU as a canonical `lean_staten`
model (`sets/ckpts.json`). x = Dan's RL compute ratio from the factor read's lists (0 for bases).

| prio | ckpt | group | stage | dose | seed | it | x | exists | md5 | path under /scratch/cowifornia/nd-rl/ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `full_s3` | main | base | full | 3 | 0 | 0.00 | yes | (new) | `knockout_danbase/bases/stage1_full_s3.pt` |
| 1 | `dose1_s0` | main | base | 1 | 0 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_dose1_s0.pt` |
| 2 | `dd1_s0` | main | rl_end | 1 | 0 | 60 | 0.41 | yes | yes | `knockout_danbase/backup/dd1_s0/seg1_dd1_s0_it60_rolling.pt` |
| 3 | `dose1_s1` | main | base | 1 | 1 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_dose1_s1.pt` |
| 4 | `dd1_s1` | main | rl_end | 1 | 1 | 60 | 0.45 | yes | yes | `knockout_danbase/backup/dd1_s1/seg1_dd1_s1_it60_rolling.pt` |
| 5 | `dose1_s2` | main | base | 1 | 2 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_dose1_s2.pt` |
| 6 | `dd1_s2` | main | rl_end | 1 | 2 | 90 | 0.69 | yes | yes | `knockout_danbase/backup/dd1_s2/seg2_dd1_s2_it90_rolling.pt` |
| 7 | `ko_s0` | main | base | 0 | 0 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_ko_s0.pt` |
| 8 | `dko_s0` | main | rl_end | 0 | 0 | 580 | 5.07 | yes | yes | `knockout_danbase/runs/dko_s0/seg6/ckpts/dko_s0_it580_snap.pt` |
| 9 | `ko_s1` | main | base | 0 | 1 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_ko_s1.pt` |
| 10 | `dko_s1` | main | rl_end | 0 | 1 | 676 | 5.93 | yes | yes | `knockout_danbase/runs/dko_s1/seg4/ckpts/dko_s1_it676_alarm.pt` |
| 11 | `ko_s2` | main | base | 0 | 2 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_ko_s2.pt` |
| 12 | `dko_s2` | main | rl_end | 0 | 2 | 400 | 5.94 | yes | yes | `knockout_danbase/backup/dko_s2/seg2_dko_s2_it400_rolling.pt` |
| 13 | `dose01_s0` | main | base | 0.1 | 0 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_dose01_s0.pt` |
| 14 | `dd01_s0` | main | rl_end | 0.1 | 0 | 695 | 4.42 | yes | yes | `knockout_danbase/backup/dd01_s0/seg2_dd01_s0_it695_rolling.pt` |
| 15 | `dose01_s1` | main | base | 0.1 | 1 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_dose01_s1.pt` |
| 16 | `dd01_s1` | main | rl_end | 0.1 | 1 | 400 | 5.43 | yes | yes | `knockout_danbase/backup/dd01_s1/seg5_dd01_s1_it400_alarm.pt` |
| 17 | `dose01_s2` | main | base | 0.1 | 2 | 0 | 0.00 | yes | yes | `knockout_danbase/bases/stage1_dose01_s2.pt` |
| 18 | `dd01_s2` | main | rl_end | 0.1 | 2 | 432 | 5.63 | yes | yes | `knockout_danbase/backup/dd01_s2/seg4_dd01_s2_it432_alarm.pt` |
| 19 | `m_dko_s0_it60` | matched | matched | 0 | 0 | 60 | 0.42 | yes | yes | `factor_read/matched/ckpts/dko_s0_it60_snap.pt` |
| 20 | `m_dko_s1_it70` | matched | matched | 0 | 1 | 70 | 0.48 | yes | yes | `factor_read/matched/ckpts/dko_s1_it70_snap.pt` |
| 21 | `m_dko_s2_it100` | matched | matched | 0 | 2 | 100 | 0.71 | yes | yes | `factor_read/matched/ckpts/dko_s2_it100_snap.pt` |
| 22 | `m_dd01_s0_it60` | matched | matched | 0.1 | 0 | 60 | 0.44 | yes | yes | `factor_read/matched/ckpts/dd01_s0_it60_snap.pt` |
| 23 | `m_dd01g0_s2_it90` | matched | matched | 0.1 | 2 | 90 | 0.66 | yes | yes | `factor_read/matched/ckpts/dd01g0_s2_it90_snap.pt` (the dd01g0 pilot line) |
| 24 | `e_dd01_s1_it35` | matched_early | matched | 0.1 | 1 | 35 | 0.27 | yes | yes | `factor_read/matched/ckpts/dd01_s1_it35_rolling.pt` (early: x below the 0.45 target) |
| 25 | `e_dd01_s2_it25` | matched_early | matched | 0.1 | 2 | 25 | 0.20 | yes | yes | `factor_read/matched/ckpts/dd01_s2_it25_rolling.pt` (early: x below the 0.69 target) |

## Time estimate

`pc_run.py plan` (the dry run prints it per checkpoint): attempts per checkpoint × that checkpoint's actions per
attempt in the factor read's unforced strict-106 read (stage mean for one it did not read) ÷ the factor read's
measured rate at batch 1024 with the Lean gate (bases 1,156 actions/s, RL ends 921, slowest end 626; ynez RTX 3090 at
250 W), plus 25 s per checkpoint for loading and E. Per checkpoint: 20,464 attempts (U 11,072, F 9,392) and 11,418 E
states. The CPU smoke (below) found attempts on the 240 no longer than on the 106, and F attempts about 10 % shorter
than U, so the per-checkpoint figure is if anything high. Cross-check: the factor read spent 361 GPU-s per
checkpoint on a comparable load (2.11 GPU-h for 21); this plan gives 373 s per main checkpoint.

| group | checkpoints | each | GPU-h (k 32, k_f 16) | +25 % (serrano's 200 W cap) |
|---|---|---|---|---|
| main: bases | 10 | ~3 min | | |
| main: 1 % ends | 3 | 7–9 min | | |
| main: 0 % and 0.1 % ends | 6 | 11–13 min | | |
| **main, total** | 19 | | **1.97** | 2.5 |
| matched | 5 | 6–8 min | 0.61 | 0.76 |
| matched_early | 2 | ~6 min | 0.22 | 0.27 |
| all | 26 | | 2.80 | 3.5 |

**Recommended plan within 3 GPU-hours: `./run_read.sh N main,matched --kf 12`**: the 19 and the 5 matched, U at
k = 32 (propensity keeps its precision; E covers small values), F at k_f = 12: 1.76 + 0.55 = **2.30 GPU-h** at the
factor read's rates, 2.9 h with the 200 W allowance. The budget guard stops it at 10,800 GPU-s, and the priority
order means a slow GPU cuts the last matched checkpoints first. The two early matched reads (0.2 h) only if the ledger
leaves room. Alternatives: defaults on `main` only (1.97 h, 2.5 h at 200 W) and matched later; or `--k 24 --kf 12`
on `main,matched` (1.98 h, 2.5 h at 200 W).

## What was verified (CPU only)

- **Tests** (`test_pc_lib.py`, 13 of 13 pass, also under pytest; `ruff check` and `ruff format --check` pass with
  the repo config):
  - applicability on 9 hand-made sequents over P Q R S, with nesting a string test gets wrong (¬ inside →, ∨ under ¬
    or ∧);
  - each opening equals Dan's `decompose` action and is legal, keeping its names, at name bases 0, 1, 11, 32, on the
    hand-made theorems and on all 587 (theorem, direction) pairs of the frozen 346; the frozen applicability is the
    parser's;
  - classifiers against the verifier on 19 constructed proofs: one per direction, a direct proof, reductio inside a
    conditional, reductio on a lemma, reductio on a negated goal, negation on ¬¬P (not reductio), opened and
    abandoned after NEGI, abandoned with the DN off the dependency path, a derived-disjunction case split, a premise
    case split toward a lemma, a lemma implication box, DN of a premise, and one broken line inside each of the four
    moves (verifier rejects; `opens` keeps d, `through` drops it). Each at name bases 0 and 9; `through` with the
    verifier equals `through` under Lean's verdict; Dan's `patterns.reductio` and derived-ORE agree on every valid
    proof. Also: attempts that open d and end inside the box or are refused; a wrong premise line (`invalid`);
  - a forced prefix plus any continuation (with a truncated last action) is classified as opening with d;
  - the scorer's rows: the state after every premise, the name positions;
  - the row builder on synthetic attempts meets SPEC.md's schema field by field;
  - **model, CPU** (`stage1_full_s3.pt`): U sampling equals Dan's `env_generate` byte for byte at (batch 8, seed 7)
    and (batch 16, seed 2500), with equal end counts, prompt tokens and waves (35 attempts); 45 sampled forced
    attempts over 7 theorems and all four directions see the env's own state after the prefix and open with their
    direction; **exact vs sampled first-move frequency** (E's
    name-free probability averaged over the 33 bases vs 1,500 sampled first proof actions from the state after the
    premises, bases drawn as the sampler draws them): cases 9.60 vs 10.20 % (z +0.79), conditional 50.10 vs 51.07 %
    (z +0.75), negation 53.53 vs 55.93 % (z +1.87); the name-free and all-token values differ in the 4th digit. No
    reductio opening had a probability between 3 and 97 % on the base, so reductio's E is checked only through the
    shared code path.
- **Classifier on real attempts** (the factor read's stored unforced draws, 6,784 each, dose1_s0, dd1_s0, dd1_s1,
  dd1_s2, dko_s2, dd01_s2): `through` reductio equals Dan's `patterns.reductio` on the pruned proof for every accepted
  proof except 2, whose DN cites an R copy of the NEGI line (counted here, as the factor read's σ chain counts it);
  the verifier accepts every Lean-accepted proof; replay agrees with the env's ending on every attempt; 99.6–100 %
  restate every premise first. Under 3 ms per attempt (about 1 min of CPU per RL end in `rows`). This check found one
  bug, now fixed: a second DN citing the same NEGI line was missed.
- **CPU smoke, end to end** (`full_s3`, six theorems covering all four forced directions and both sets, k 8, k_f 4,
  Lean gate on; output under `propensity_capability/smoke/`): 33 s; E 660 rows, U 48 and F 48 attempts; 60 result
  rows that pass the runner's schema check and the analysis side's `analyze.load`; QC: forced attempts whose first
  action is not their direction 0, replay mismatches 0, Lean-accepted proofs the verifier rejects 0, premises first
  100 %.
- **CPU length smoke** (`full_s3`, `dd1_s0`, `dko_s1`, `m_dko_s0_it60`; 8 + 8 theorems, k 4, k_f 2; under
  `smoke_len/`): actions per attempt, U strict / other: 7.0 / 9.2, 18.2 / 13.8, 34.7 / 32.1, 16.7 / 12.7; F: 7.0 /
  8.4, 14.8 / 13.6, 26.2 / 32.1, 14.4 / 14.4. QC zero on all four.
- `run_read.sh 0 --dry-run` prints the plan, the ledger, the GPU's users and the exact commands. At 9:10 pm PT all
  five serrano GPUs had other users' processes, so the gate would have refused (exit 3), as it should.

## Not verified, or assumed

- Nothing has run on a GPU. Batch 1024 is the factor read's forced-stage batch, which ran on these checkpoints on a
  24 GB RTX 3090 with allocator retries and no lost draw; U at 1024 on the longest RL ends is untested. If it runs
  out of memory, rerun with `--batch 512` (arms already written are kept; a batch change re-draws the noise).
- The time estimate assumes the factor read's throughput on ynez (RTX 3090 at 250 W); serrano's GPUs are capped at
  200 W.
- "Used" counts reductio on any formula, but negation and conditional only on the goal in focus; that asymmetry is a
  definition, documented in `pc_lib.moves`.
- E is conditional on the attempt having restated every premise first (measured at 99.6 to 100 % in the factor
  read's unforced draws; the QC file reports the share per checkpoint).
