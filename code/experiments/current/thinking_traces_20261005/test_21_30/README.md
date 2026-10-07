---
written_on: 2026-10-06
written_by: agent:claude
---

# test_21_30: a sealed test set of 21–30-line natural-deduction theorems

> **Update 2026-10-07.** Two changes, both made while building the [13–20 companion](../test_13_20/README.md):
> - **Shorter references.** `test_13_20/negf.py` shortened 63 references by one line. `minlen` proves
>   ( ~ F ) with a 3-line box; the 2-line `| F : AS ; ( ~ F ) : NEGI s s` is valid. 8 rows fell to
>   20 lines and were dropped, so the set is now **2,767**.
> - **Sizes and calibration.**
>   - Per length: 391 / 403 / 396 / 402 / 392 / 297 / 165 / 170 / 117 / 34.
>   - 2-line bins: 794 / 798 / 689 / 335 / 151.
>   - T0 s0 B0 was calibrated on 575 stratified items (k 8): **0 solved**.
>   - The tables below give the numbers before this update.

`test_21_30.jsonl` holds **2,775 theorems**. Each reference proof has 21–30 lines, is Lean-checked, and is close to
minimal length (see Minimality). It measures length generalisation plus genuine difficulty: training data stops
short of it (K12 2–12 lines, median 7; RL targets 7–14; new RL pool up to 20).

| file (under `/scratch/cowifornia/nd-rl/think/runs/pools/test_21_30/`) | contents |
|---|---|
| `test_21_30.jsonl` | the set, in the OOD-set row format `code.thinking.eval --sets` reads |
| `test_21_30.keys.jsonl` | sidecar of renaming-class keys (`name`, `key`, `pkey`; `code.thinking.data.class_keys`), for checking later training data |
| `test_21_30.meta.json` | recipe, seeds, histograms, diversity, dedupe, validation, calibration, power |
| `seal_report.json`, `profile.json`, `calib_report.json`, `power.json` | the reports summarised below |
| `calib/` | calibration eval outputs (`rows_*.jsonl`, `summary.json`) |
| `work/` | every intermediate file (pieces, composites, validation runs, logs) |

The scripts are in this folder. `regenerate.sh` lists the full pipeline.

## Row format

Fields shared with `ood_check/sets/*.jsonl`:

- `name` (`t2130_NNNN`)
- `prompt`
- `reference_lines`
- `family` (`cut_composite`)
- `knob` (= `reference_lines`)
- `reference_proof` (fork ND text: `nd_verify` and `state_env.decompose` accept it)
- `max_state`
- `max_action`
- `formula_depth`
- `in_train_class` (false)

Added fields:

- **Shape:** `n_prem`, `n_atoms`, `box_depth`, `n_ore`, `n_actions`, `max_name`, `prompt_tok`, `rules`.
- **Keys:** `key`, `pkey`.
- **`minimality`:**
  - `L_ub`: the reference length.
  - `L_lb`: certified lower bound.
  - `lb_search_bound`, `lb_timeout`.
  - `construction_lines`.
  - `vocab_search`: `minimal`, `shortened`, `timeout`, or null if not run.
  - `pieces_exact`.
- **`provenance`:** `parts` (role, piece class, piece L), `cuts`, `ops` (discharges), `target`.
- **`lean_ok`.**

## Why composition, not the generator alone

The K12 generator (`gen.py`) almost never produces a theorem whose *minimal* proof is long, whatever the length of
the generated proof. Dan's long-pool runs labelled about 840k generated theorems with `minlen`:

- only 91 had `L_true ≥ 17`;
- those 91 were built with 32–48 generated lines.

A generator-only 21–30 set would therefore be padded about 2×. Exact search is out of reach above about 17 lines
(bound 17: up to 30 min and 5–8 GB per theorem).

The set is therefore built from **pieces** joined by **cuts**:

- **Pieces** are generator theorems whose minimal proofs are known exactly.
- **Cuts** replace a premise by a lemma piece that proves it, the way long proofs are built from lemmas.

Exact search on composites short enough to search fully decided which cuts keep the result near-minimal.

## Recipe

1. **Pieces** (`gen_pieces.py`, `fastlen.py`, `prep_pieces.py`).
   - **Generator.** The fork's generator is unchanged (max_prem 3, max_depth 3), 50/50:
     - strict long mode (the ladder / long-pool generator);
     - take-home mode (the K12 generator).
   - **Literal mode.** In literal mode, the generator's forward-mode proofs also yield their depth-0 literal lines
     as conclusions, with the same pruning as `Gen.finish`.
   - **Exact labels.** Each theorem gets its exact minimal length L ≤ 13 from `fastlen.staged`: the fork's `minlen`
     rules and restricted formula space, with a cross-budget memo. It agrees with `minlen` on 300 / 300 random
     theorems and is about 2× faster.
   - **Seeds.** Seeds are 210000, 410000, 610000 and 810000; worker i uses seed + 1000·i.
   - **Unused premises.** Premises the minimal proof leaves unused are dropped. The reduced theorem's minimum is
     exactly L − u.
   - **Exclusion.** Pieces whose class appears in any training, eval or pool file are dropped.
   - **Result.** 147,764 pieces, of which 102,646 are consistent and ≤ 13 lines.
2. **Composites** (`build_composites.py`, `compose.py`, `ndproof.py`).
   - **Sampling.** Pick a target length in 21–30, weighted by remaining quota, and a main piece. Then apply 1–3
     cuts (up to 5 in the tail builds).
   - **Cut.** A cut replaces a premise p by a lemma piece proving p. The lemma's atoms are renamed; its free atoms
     go to the renaming that shares the most premises 70% of the time, otherwise to a random one. Then:
     - `simplify` drops duplicate and unused lines;
     - `sink_all` moves a lemma used only through a reiteration line into the box that uses it.
   - **Discharge.** When a cut would exceed 3 premises, another premise A is first discharged into the conclusion:
     Γ, A ⊢ C becomes Γ ⊢ A > C, adding one line.
   - **Allowed lemmas.** The minimal proof ends with ANDE or IMPE, or with ORE when the cut formula is a literal. It
     uses no ex falso (BOTE).
   - **Rejected composites:**
     - inconsistent premises;
     - a semantically redundant premise or discharged antecedent;
     - a conclusion that is a tautology;
     - a premise the reference does not use;
     - more than 3 premises;
     - formula depth > 5 (> 6 in the tail builds; K12 has 12% at depth 5 and 2.8% at depth 6);
     - prompt > 90 tokens;
     - box depth > 3;
     - a `state_env` state > 386 tokens (K12's maximum);
     - canonical name > 32 (with name base ≤ 32, at most 64 names).

     Every composite passes `nd_verify`.
3. **Refinement** (`refine.py`). For 15 s, a vocabulary-restricted exact search (`fastlen.vocab_shorten`) looks for a
   proof whose intermediate formulas are the reference's own. A shorter proof found replaces the reference.
4. **Checks.**
   - **Lean** checks every reference (`lean_set.py`: `state_env.decompose` → `lean_seq` text →
     `code.thinking.lean.lean_check`, 16 workers): 2,775 / 2,775 accepted.
   - **Lower bound** (`minimality.py`): a full-space search to 13 lines, in the fork's space plus the reference
     formulas.
5. **Assembly and sealing** (`assemble.py`, `seal.py`, `profile_set.py`, `calib_report.py`, `power.py`,
   `build_meta.py`).

### Run log (2026-10-06, all on this VPS; CPU only except calibration)

| step | what ran |
|---|---|
| pieces | 4 generator runs (15–25 min each); `prep_pieces` |
| composites | `comps.jsonl` (21–30); `comps_tail.jsonl` (24–30); `comps_tail2.jsonl` and `comps_tail3.jsonl` (28–30) |
| vacuous-antecedent filter | applied after the fact to the first three builds (791 removed); built into `build_composites.py` before `comps_tail3` |
| refinement | 15 s on 3,429 candidates |
| Lean + lower bound | on the 2,720 + 67 selected |
| write | at most 400 per length, 5 premise-permutation duplicates dropped |
| seal | done |
| calibration | GPU 0 |

`regenerate.sh` runs the same steps in sequence.

## Minimality: what "21–30 lines" means here

Exact minimal length cannot be computed at 21–30 lines. The claim rests on five things.

- **Pieces are exact.** Each piece's L is final: the search at every smaller bound finished without a proof. This
  holds in `minlen`'s restricted formula space, like every ladder label.
- **The recipe is validated by exact search** on composites short enough to search exhaustively, in the fork's
  space plus all reference formulas. Excess is the reference length minus the minimum found.

  | validation | n | exactly minimal | within 1 line | mean excess (lines) |
  |---|---|---|---|---|
  | final recipe, 12–16 lines (`val6`) | 462 | 93.7% | 96.5% | 0.12 |
  | final recipe, 17 lines (`val7`, bound 16) | 59 | 89.8% | 96.6% | 0.15 |
  | the same after refinement, 12–16 | 462 | 99.1% | 99.6% | 0.013 |
  | the same after refinement, 17 | 59 | 98.3% | 100% | 0.017 |
  | literal-cut-only variant, 12–16 (`val3`) | 598 | 96.7% | 99.3% | 0.045 |

  These rules came from earlier validation rounds (`work/val*_comps_min.jsonl`), and the recipe excludes each case:

  - **Implication cuts were 18% exact.** Deriving A > B in a box and then applying it is longer than inlining it.
  - **Lemmas ending in an introduction rule were 8% exact.**
  - **ORE-ending lemmas for compound formulas were 33–53% exact.** The case split moves to the use site.
  - **Ex falso lemmas** left a 1-line shortcut in about half the composites.
  - **Inconsistent premises** let anything follow in a few lines.
  - **Vacuous antecedents.** A discharge whose antecedent is not needed lets everything below it be skipped. The
    lower-bound search found one 27-line candidate with a 13-line proof this way.
- **Refinement mostly times out at 21–30 lines.** On the set, the 15 s vocabulary search certified 149 theorems,
  shortened 1, and timed out on 2,069. It was not run on 556 (late builds). Where it did not complete, the
  pre-refinement validation rate applies: about 6–10% of theorems may have a proof 1–3 lines shorter, mean excess
  about 0.15 lines.
- **Certified lower bound.** 2,772 theorems have no proof of ≤ 13 lines in the search space (`L_lb = 14`). For 3,
  the 13-line stage timed out (`L_lb = 13`). A deeper search on a random sample is reported under Results.
- **The models' own proofs** (calibration) can be compared with the references. See Results.

## Results

### Size and length

There are 2,775 theorems. Every reference proof is Lean-accepted (2,775 / 2,775) and passes `nd_verify`.

| reference lines | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| theorems | 392 | 398 | 400 | 400 | 399 | 291 | 173 | 167 | 120 | 35 |

In 2-line bins, 21–22 / 23–24 / 25–26 / 27–28 / 29–30 hold 790 / 800 / 690 / 340 / 155.

### Minimality

- **Lower bound.** 2,772 theorems have no proof of ≤ 13 lines; 3 hit a timeout at the 13-line stage.
- **Random sample of 120** (all lengths): 120 / 120 have no proof of ≤ 15 lines, so L ≥ 16. The median search took
  134 s.
- **Random sample of 36** at 21–23 lines: 36 / 36 have no proof of ≤ 16 lines, so L ≥ 17.
- **Recipe validation** (exact search, 12–17 lines; table above): 90–94% exactly minimal before refinement.
- **Refinement on the set.** The 15 s vocabulary search certified 149 theorems, shortened 1, timed out on 2,069
  and was not run on 556.
- **Calibration proofs.** The 6 theorems the models solved had shortest accepted proofs 2× equal to the reference
  and 4× longer; none shorter.

### Diversity against K12

The K12 column is a random sample of 20,000 from `train_k12.jsonl`.

| | test_21_30 | K12 |
|---|---|---|
| distinct renaming classes | 2,775 / 2,775 (100%); premise-order-invariant 100% | — |
| premises 0 / 1 / 2 / 3 | 0 / 0.0 / 2.8 / 97.2% | 5.9 / 25.5 / 48.3 / 20.2% |
| atoms 2 / 3 / 4 | 0.0 / 10.1 / 89.9% | 25.2 / 47.8 / 23.8% (1: 3.2%) |
| theorem formula depth 3 / 4 / 5 / 6 | 11.5 / 47.4 / 34.4 / 6.6% | 45.5 / 29.1 / 12.3 / 2.6% (≤ 2: 10.1%, ≥ 7: 0.4%) |
| box depth 1 / 2 / 3 | 21.9 / 60.2 / 17.9% | 43.1 / 31.9 / 0% (0: 25%) |
| ORE per proof 0 / 1 / 2 / 3 / ≥ 4 | 0.3 / 4.4 / 25.6 / 42.1 / 27.6% | 55.7 / 39.7 / 4.6 / 0 / 0% |
| proofs using rule: IMPE / ORE / IMPI / ANDE1 / BOTE / NEGE / DN | 100 / 99.7 / 94.5 / 58.5 / 57.8 / 61.9 / 36.5% | 46.2 / 44.3 / 42.7 / 4.4 / 8.0 / 22.1 / 13.8% |
| prompt tokens (median / max) | 76 / 90 | 39 / 156 |
| `state_env` actions (median / max) | 25 / 31 | 8 / 13 |
| max state tokens (median / max) | 214 / 357 | 98 / 386 |
| largest canonical name (median / max) | 13 / 22 | 6 / 11 |

Composition:

- **Pieces.** 3,954 distinct pieces; 2,612 are used once, the most-used 14 times. There are 1,922 distinct main
  theorems.
- **Cuts per theorem.** 1 / 2 / 3 cuts: 871 / 1,806 / 98.
- **Discharges per theorem.** 0 / 1 / 2 discharges: 898 / 1,842 / 35.

Full histograms and the rule mix are in `profile.json`.

### Sealing

The set was checked by renaming class (`data.class_keys`: canonical key and premise-order-invariant key) against 106
files, 1,547,645 records:

- K12 train and held-out;
- the cap-6 Stage-1 set;
- every jsonl under the fork's `data/` (ladder and take-home `rl_targets`, `transfer`, `transfer_long*`, textbook72,
  gt, p2, r1–r3, lj, bs);
- the 5 OOD sets;
- the 4 thinking-trace files;
- the RL-pool files `raw/cands.jsonl`, `cands_clean.jsonl`, `long2_cands.jsonl`, `rl_hard.jsonl` and
  `rl_hard_filtered.jsonl`.

There are **0 hits** and 0 within-set duplicates, and also 0 hits against `runs/restart2_hard12/hard12_test_1k.jsonl`.
Pieces whose class appears in those files were excluded too. To check a future pool, use one of:

    cd /scratch/cowifornia/nd-rl/think/nd-rl && ND_OFFLINE=1 code/.venv/bin/python -m \
      code.experiments.current.thinking_traces_20261005.test_21_30.seal check \
      --index /scratch/cowifornia/nd-rl/think/runs/pools/test_21_30/work/excl_index_final.pkl \
      --set /scratch/cowifornia/nd-rl/think/runs/pools/test_21_30/test_21_30.jsonl \
      --report /tmp/seal_new.json --rl_pool NEW_POOL.jsonl
    # or, from training code: build data.EvalIndex([".../test_21_30.jsonl"]) and drop rows whose hit() is non-empty;
    # the sidecar test_21_30.keys.jsonl lists the same keys

### Calibration (GPU 0, `code.thinking.eval`, plain arm, k 8, `--max_steps 96`)

| model | n | pass@1 | pass@8 | solved | wall |
|---|---|---|---|---|---|
| T1v1 s0, B 256 | 2,775 | 0.027% | 0.22% | 6 | 3,975 s (GPU shared with another agent's training job) |
| T0 s0, B 0 | **not run** | | | | GPU 0 never fell below 500 MiB in two 60-min windows |

T1v1 by 2-line bin (pass@8): 21–22: 0.38%; 23–24: 0.38%; 25–26 and above: 0.

- **Where attempts fail.** The median failure comes after 8–9 accepted steps, out of about 25.
- **Structurally complete but wrong.** 14–25% of attempts (more at shorter lengths) finish structurally, but Lean
  rejects the terms in all but 6.

**This is the 0% cliff: pass@8 is at the floor.** The set cannot rank these Stage-1 checkpoints.

To run T0 when GPU 0 is free:

    bash calib_one.sh /scratch/cowifornia/nd-rl/think/runs/pools/test_21_30/test_21_30.jsonl \
      /scratch/cowifornia/nd-rl/think/runs/pools/test_21_30/calib 8 T0_s0 \
      /scratch/cowifornia/nd-rl/think/runs/stage1/T0_s0/model.pt 0

Estimate: about 1,000–1,500 s, since T0 does no thinking (about 290 action tokens per attempt, against 1,003 total
for T1v1 B 256).

### Power (`power.py`, `power.json`)

- **On this set (T1v1 rows).** The single-model SE of pass@1 is 0.011 pp, and two runs of the same model differ
  with SE 0.016 pp at k 8. Both numbers are degenerate: only 6 theorems were ever solved.
- **Projection for a set with mid-range pass rates.** This uses the variance structure of existing paired rows
  (textbook72, T0 B0 vs T1v1 B256, k 256, pass@1 ≈ 19%): theorem spread of the true difference 246 pp², attempt
  noise 3.9 pp² per theorem at k 256.

  | N | k | SE, fixed set | significant at 95% (fixed) | 80%-power MDE (fixed) | SE, new theorems | significant (new) | MDE (new) |
  |---|---|---|---|---|---|---|---|
  | 2,775 | 8 | 0.21 pp | 0.42 pp | 0.59 pp | 0.37 pp | 0.72 pp | 1.02 pp |
  | 2,775 | 16 | 0.15 pp | 0.29 pp | 0.42 pp | 0.33 pp | 0.65 pp | 0.93 pp |
  | 1,000 | 8 | 0.35 pp | 0.69 pp | 0.99 pp | 0.61 pp | 1.19 pp | 1.71 pp |
  | 600 | 8 | 0.46 pp | 0.89 pp | 1.28 pp | 0.79 pp | 1.54 pp | 2.20 pp |

- **Recommendation.** Use k 8 on about 2,800 theorems whose pass@1 sits mid-range: a 1-point difference between two
  checkpoints is then significant. Use k 16 if the claim should generalise beyond the sealed items.
- **Cost per checkpoint** at k 8 on 2,775 theorems:
  - T1v1 B 256: 3,975 s wall, with GPU 0 shared; probably about 45 min alone.
  - T0: about 20–25 min (estimate).

  k 16 doubles both.

### Proposed fix for the floor (not run; needs Leon's approval)

1. **Keep the far tier.** test_21_30 stays sealed as the far tier, for RL checkpoints and for the length-extrapolation
   floor check.
2. **Build a graded companion** with the same recipe and the existing pieces (CPU only, about 1 h):
   - `build_composites.py --lo 13 --hi 20`, about 300 per length;
   - at 13–16 lines, certify every reference exactly with `minimality.py --bound 15`;
   - at 17–20, certify a sample at bound 16.

   With failures at a median of 8–9 accepted steps, the 5–50% pass@1 band for Stage-1 models probably lies around
   13–18 lines.
3. **Calibrate it** on a 600-item subsample, T0 and T1v1 at k 8 (about 15 min of GPU each). Then use the bins with
   pass@1 in 5–50% as the 1-point comparison set, and report test_21_30 separately as the far tier.

## Caveats

- **Restricted search space.** Exactness and lower bounds hold in `minlen`'s restricted formula space, plus the
  reference's formulas, as for every fork label. Lean may accept shorter terms (e.g. `n.elim`).
- **The theorems are composites, not raw generator output.** Each contains 2–4 generator theorems.
  - The pieces are natural.
  - The joins are cuts on a premise, plus at most two discharges (Γ, A ⊢ C → Γ ⊢ A > C).
  - Piece reuse is in `test_21_30.meta.json`: 3,954 distinct pieces, 2,612 used once, at most 14 uses.
  - Compared with K12, the set has more premises (97% have 3), 4 atoms (90%), deeper boxes (18% at depth 3, which
    K12 never has) and 2–4 case splits per proof (91%). Its prompts (median 76 tokens) and states (median 214) sit in
    K12's upper tail.
- **The length tail is thin.** Long pieces are rare and the 3-premise limit caps how far cuts can grow a theorem.
  The counts per length are 392 / 398 / 400 / 400 / 399 / 291 / 173 / 167 / 120 / 35. The 2-line bin 29–30 holds
  155, below the 200 aimed for.
- **Not byte-reproducible.** Generation and composition stop on wall-clock budgets, so `regenerate.sh` reproduces
  the recipe and its distribution, not this exact set.
