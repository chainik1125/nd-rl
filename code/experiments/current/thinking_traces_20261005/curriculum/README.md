---
written_on: 2026-10-09
written_by: agent:claude
---

# Length-curriculum RL pools: 10–12 lines (stage 1), then 13–15 lines (stage 2)

**Result in brief.** Two RL pools of 5,000 problems, each with a 1,000-problem in-distribution test split. Between
the two RL stages the shortest-proof length rises by 2.3 lines; the generator family stays the same.

| RL stage | band | file | rows | shortest-proof lines (%) | mean |
|---|---|---|---|---|---|
| 1 | L2: 10–12 | `L2/pool.jsonl` | 5,000 | 10: 28, 11: 42, 12: 30 | 11.03 |
| 1 | L2: 10–12 | `L2/test_1k.jsonl` | 1,000 | 10: 29, 11: 43, 12: 28 | 10.99 |
| 2 | L3: 13–15 | `L3/pool.jsonl` | 5,000 | 13: 78, 14: 16, 15: 6 | 13.28 |
| 2 | L3: 13–15 | `L3/test_1k.jsonl` | 1,000 | 13: 77, 14: 16, 15: 6 | 13.29 |

Paths are under `/scratch/cowifornia/nd-rl/think/runs/pools/curriculum/`. The final checks (`report.py`; output in
`report.md` and `report.json`) found:

- Lean accepts every row's constructed proof and reference (shortest) proof: 12,000 of 12,000 each.
- No row shares a renaming class with any file of the exclusion index or with textbook72.
- No row matches, as written or in curried normal form, a problem of Charles's 2026-10-02 release, textbook72 or a
  sealed test set.
- The four files are pairwise disjoint by renaming class and by curried normal form.

For comparison, the RL stages run so far used `rl_hard12` (mean 9.7 lines) and then two samples of hard712 (mean
10.4 each). All of them stay within 7–12 lines, so the proof-length distribution did not rise after the first stage.

**The curriculum (Leon, 2026-10-09).**

- RL stage 1 trains on L2 and stage 2 on L3.
- A 7–9-line band (L1) was started and then dropped (§ L1).
- Training on 13–15 lines is confirmed. The **out-of-distribution tests** are therefore the 16–20-line part of
  `test_13_20` (t1320), `test_21_30` and textbook72.
- The 13–15-line part of t1320 is in distribution for stage 2.

## Sources

Every row comes from one generator family: hard712's proof-first generator (`../hard712/pfgen.py`), certified by
hard712's gates (`../hard712/pfgates.py`). Those gates require:

- a compact statement: ≤ 2 premises, ≤ 7 connectives, 2–3 atoms, formula depth ≤ 3, and no artifacts;
- a certified shortest proof;
- at least one target structure that the shortest proof needs.

| band | source | rows certified | in the reservoir |
|---|---|---|---|
| L2 | hard712 rows unused by any RL run or sealed split, at 10–12 lines (`reuse.py`) | 14,256 (hard712's) | 12,416 |
| L3 | hard712's 4,847 classes rejected only for a 13-line shortest proof, rebuilt from their seeds (all 4,847 matched their stored class) and certified at 13–15 (`regen.py --status long`) | 4,261 | 3,804 |
| L3 | fresh seeds 0–25.75M (`run.py generate`, band 13–15) | 3,508 | 2,181 |
| L3 | the fresh run's construction-gap rejects (constructed proof 2–3 lines over the shortest), certified with gap limit 3 (`regen.py --status gap --source L3/gen`) | 700 | 572 |

The reservoir counts are what is left after the curried-form duplicates and the exclusions below are removed.

**What differs from hard712's run.**

- **The length window.** It is 10–12 or 13–15 lines instead of 7–12. L3 allows constructed proofs of up to 18 lines.
- **The gap limit, for 830 of L3's 6,557 reservoir rows (739 of the 6,000 final rows).** Up to 3 lines (hard712: 1)
  between the constructed proof and the shortest one. This is invisible in a row: `min_lines`, the reference proof, the families and the necessity
  checks all come from the shortest proof. Only `gen_lines` / `minlen_bound` show it.
- **Recipe weights for fresh L3 seeds.** pfgen's defaults for the first ~1M seeds. After that they are adaptive and
  proportional to each recipe's recent yield of rows that are neither classical-only nor ex falso
  (`PFG_CLASSICAL_VALUE=0 PFG_EXFALSO_VALUE=0 PFG_SCARCE_BONUS=0`; hard712 valued classical rows at 0.1 and gave
  scarce attributes a bonus). Generation also skipped classes the `long` source certifies anyway (`--skip`).
- **The mix gate's caps** (§ Mix): L3 needs a 47% classical-only cap, and L2 is matched to L3.

## Exclusions

A candidate is dropped when its renaming class (atom renaming, premise permutation: `class_keys`) matches any of the
following.

- **The renaming-class index** (`excl.py`): 497 files, 977,825 classes.
  - hard712's 364 files: K12 train and held-out, the cap-6 Stage-1 set, every jsonl under the fork's data/ (textbook72
    dev and train included), the OOD sets, rl_hard12 and the other RL pools, hard12_test_1k, test_13_20, test_21_30,
    and the trace files.
  - hard712's own outputs: pool, test_1k, dev_500, pool_s5k_a/b, and every gated row.
  - Every prompt-bearing jsonl under `runs/pools/`, including test_13_20's and test_21_30's candidate composites and
    their pieces, and the canary set.
  - The eval subsets under `runs/eval/matched/sets/`.
  - Rows reused from hard712 ignore hits on hard712's own pool and work files, which are where they come from.
- **Held-out sets, in curried normal form** (`heldout.py`; coordinator, 2026-10-09).
  - Charles's textbook release of 2026-10-02: `runs/pools/charles_1002/{candidate_v0, candidate_v1, batch3,
    hand_proved_14, derived}.jsonl`, 412 rows, commit `5d0084a`.
  - textbook72.
  - Curried normal form moves every leading antecedent of the goal into the premises and drops repeated premises, so
    `|- A > (B > C)` and `A, B |- C` are one problem (Charles's audit rule of 2026-09-30). Both sides are compared in
    that form and as written.
- **The sealed tests, in curried normal form**: test_13_20, test_21_30, hard12_test_1k, and hard712 test_1k and
  dev_500. The index above already covers them as written.
- **Another row of the same band, or of the other band, in curried normal form.** So no two rows of the curriculum
  are one problem up to currying, and a band's pool and test cannot share a problem.
  - L3 was assembled first. Cross-band twins therefore left L2, which had the larger supply.
- **Substitution instances of textbook72 theorems**, in either direction (hard712's shape gate, applied to every row).

**Removed by the curried and held-out checks** (distinct classes; a row can match several files):

| band | Charles 2026-10-02 | textbook72 | sealed tests | curried duplicates within the band | curried twins of an L3 row |
|---|---|---|---|---|---|
| L2 | 11 (candidate_v0 4, candidate_v1 3, batch3 4, derived 4) | 1 (train) | 256 (hard712 test_1k 166, dev_500 94) | 1,171 | 403 |
| L3 | 7 (batch3 3, candidate_v1 2, candidate_v0 2, derived 2) | 2 (dev) | 39 (hard712 test_1k 30, dev_500 9) | 1,846 rows (exact or curried duplicates across its three sources) | — |

The index also removed candidates before certification:

- L2: 5 hard712 rows matched composite pieces of the sealed sets.
- L3 fresh seeds: 2,997 candidates.

The final files have 0 remaining overlaps, including with `charles_1002/charles_clean.jsonl`.

## Mix and splits

The mix gate is hard712's deterministic one (`run.py select`).

- **Caps** on classical-only, reductio, nested negation, (A>B)>C goals, ex falso, biconditionals, premise-free
  statements, case splits and irrelevant atom occurrences.
- **Floors** on the scarce attributes, met by taking those rows first.
- The gate fills 6,000 rows per band. When it cannot, the caps relax in hard712's fixed order.

**L3.** With hard712's caps the gate fills only 4,048 rows: about half the 13–15-line rows the generator finds are
classical-only. The smallest classical-only cap that fills 6,000 is **47%** (`find_cap.py`, binary search). Ex falso
stays at hard712's 10%, and the other caps relaxed in hard712's order.

**L2** uses L3's realized shares as its caps: classical-only 47%, ex falso 10%, reductio 31.55%, nested negation
43.9%, nested negation without reuse 29.03%. Its other caps are hard712's. So stage 1 is not much easier than stage 2
on these attributes.

- The alternative, hard712's own caps (classical-only ≤ 20%), would leave a 20% → 47% jump in classical-only problems
  between the stages.
- It takes one command:
  `run.py --out L2 select --n 6000` followed by
  `run.py --out L2 split --n_test 1000 --n_dev 0 --prefix cL2 --source curriculum_L2`.

Shares in the pools (`report.md`):

| pool | classical-only | ex falso | reductio | nested negation | case split | no premises | (A>B)>C goal | box depth | premises | connectives |
|---|---|---|---|---|---|---|---|---|---|---|
| L2 | 39.6% | 10.0% | 27.6% | 34.4% | 30.2% | 7.7% | 11.9% | 2.23 | 1.33 | 6.50 |
| L3 | 47.1% | 10.2% | 31.5% | 43.8% | 22.9% | 0.6% | 22.3% | 2.66 | 1.63 | 6.75 |

L2's greedy selection fills 6,000 below its 47% classical-only cap. The remaining differences go with proof length,
as far as this generator's 13–15-line supply shows:

- more nested negation;
- more (A>B)>C goals;
- almost no premise-free statements;
- deeper boxes.

The state-env reference takes 12.0 steps on average for L2 and 14.3 for L3.

**Splits.** `run.py split` shuffles the band's 6,000 selected rows with `random.Random(20261008)`. The first 1,000
are `test_1k`, the rest `pool`. No model was run on any of these rows before or after the draw, so the test splits
are unfiltered random samples of the band's selected set.

## L1 (7–9 lines): dropped

The supply under hard712's gates is small:

- 3,750 unused hard712 rows;
- 2,607 of hard712's construction-gap rejects, re-certified with gap limit 6;
- 793 fresh rows, at 118, 66 and 39 rows per million seeds in successive rounds (the class space is nearly exhausted).

Leon dropped the band. Its intermediates stay under `L1/`; no pool or test file was written.

## Files

Under `/scratch/cowifornia/nd-rl/think/runs/pools/curriculum/`:

| path | contents |
|---|---|
| `L2/pool.jsonl`, `L2/test_1k.jsonl`, `L3/pool.jsonl`, `L3/test_1k.jsonl` | the pools and test splits: hard712's row format; names `cL2_*`, `cL2t_*`, `cL3_*`, `cL3t_*`; `source` = `curriculum_L2` / `curriculum_L3` |
| `L*/keys.jsonl`, `L*/*.keys.jsonl` | renaming-class sidecars (`name`, `key`, `pkey`, `split`) |
| `L*/work/` | `gated_lean.jsonl` (the band's reservoir), `assemble_report.json`, `selected.jsonl`, `spare.jsonl`, `select_report.json` |
| `L2/reused.jsonl`; `L3/regen/`, `L3/regen_gap/`, `L3/gen/` | the sources (certified rows, rejects, progress, Lean reports; `L3/gen/skip_keys.txt`) |
| `work/excl_index.pkl` | the renaming-class index and its build log |
| `report.json`, `report.md` | the final checks |

The rows are read by `ei.py --targets`, `grpo.py --targets` and `code.thinking.eval --sets`, like hard712's.

## Timing

CPU only, with at most 12 worker processes at `nice 10`, from 22:45Z on 2026-10-09 to 03:13Z on 2026-10-10.

- Exclusion index: 1 min.
- L2: about 2 min.
- L3 fresh generation: 3.6 h of wall time across runs.
- L3 `long` re-certification: 2.4 h (4 workers, then 8).
- L3 gap re-certification: 13 min (12 workers).
- Lean, selection, splits and checks: 4 min.

## Code

Worktree `/scratch/cowifornia/nd-rl/think/nd-rl-curr`, branch `leon/thinking-curriculum` (from `leon/thinking`
`744f282`, uncommitted).

- **This folder**: `excl.py`, `reuse.py`, `regen.py`, `assemble.py`, `heldout.py`, `find_cap.py` and `report.py`, plus
  `build.sh`, which lists every step in the order it ran.
- **`../hard712/pfgates.py`**: the length band, gap limit and search limits are read from the environment
  (`PFG_MIN_L`, `PFG_MAX_L`, `PFG_MAX_GAP`, `PFG_SEARCH_SECS`, `PFG_NEC_SECS`). The defaults are unchanged.
- **`../hard712/run.py`**:
  - worker memory (`PFG_MEM_GB`);
  - the adaptive-weight values (`PFG_CLASSICAL_VALUE`, `PFG_EXFALSO_VALUE`, `PFG_SCARCE_BONUS`);
  - `generate --skip`;
  - the split's row-name prefix and `source` (`--prefix`, `--source`).

  The defaults are unchanged.

Checks: ruff check and format are clean on the new and changed files. pyright reports 0 errors in this folder; the 5
errors in `run.py` and `pfgates.py` are also in the unmodified files.

## Big-statement twins (`size_twins.py`, 2026-10-16)

A statement-size knob for the mini RL runs (knob proposal A1, "bigger statements, same proofs"). `size_twins.py`
substitutes each row's atoms with small formulas over P–S, so that the statement grows to 45–90 whitespace tokens
(drawn from test_13_20's sizes) while the reference and constructed proofs keep their rules, boxes, citations and
line count.

- Each substitution is *independent*: the images reach every truth-value combination, so the twin keeps the
  original's truth-table facts (consistency, premise non-redundancy, no tautological goal).
- The builders' shortest-proof search (`fastlen.fast_minlen`, bound L − 1) re-certifies `min_lines` = L on the twin.
- The builders' necessity search checks the needed structures.
- A classical-only original keeps a classical-only twin.

Built for the mini subsets (`runs/pools/mini/`) and both `test_1k` splits. The data, `report.json` and a README are
under `/scratch/cowifornia/nd-rl/think/runs/pools/size_twins/`; `build_size_twins.sh` lists the steps, and
`test_size_twins.py` has the tests.

## Found on the way (outside this build)

- **hard712's own splits are not disjoint up to currying.**
  - hard712's discharge step turns premises into antecedents, so its pool holds pairs that are one problem in curried
    form. There are 1,035 such classes among the 14,256 unused rows alone.
  - 133 of hard712's 1,000 `test_1k` problems have a curried twin in `pool_s5k_a` / `pool_s5k_b`, the RL samples that
    EI and GRPO trained on in rounds 2 and 3.
  - Reads of hard712 dev and test for those runs are therefore partly in distribution in that sense.
- **Trained hard712 samples also match held-out problems in curried form.** `pool_s5k_a` has 13 matches with
  Charles's release and textbook72, 3 of them textbook72 dev problems.
- **hard712's index missed some sealed-set pieces.** 31 hard712 rows at 7–9 lines and 5 at 10–12 match composite
  pieces of the sealed test_13_20 / test_21_30 sets (`work/pieces*.jsonl`), which hard712's index did not hold.
