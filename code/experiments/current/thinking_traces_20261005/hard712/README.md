---
written_on: 2026-10-08
written_by: agent:claude
---

# hard712: a proof-first generator of compact, genuinely long 7–12-line problems

**Result in brief.**

- **Size:** 120M seeds gave 29,542 gated, unique problems in 171 min of generation (24 CPU workers). Lean accepted
  every constructed and every reference proof. The mix selection kept 23,092, split into `pool.jsonl` 21,592,
  `test_1k.jsonl` 1,000 and `dev_500.jsonl` 500.
  - That is **below the 25,000 target**: the generator's class space saturates (§ Yields).
  - The other 6,450 gated rows, mostly classical-only, are in `work/spare.jsonl`.
- **Compact and genuinely long:**
  - every problem has ≤ 7 connectives and a shortest proof of 7–12 lines, at most 1 line shorter than the constructed
    proof; 90% need 9–12 lines (rl_hard12 8%, textbook dev58 7–12 70%);
  - nested negation boxes: 40% vs 1% (textbook 25%); box depth ≥ 2: 98% vs 17% (75%);
  - no `~~`, no inconsistent or redundant premises.
- **Missed targets:** premise-free statements 9% (target 25–35%), biconditionals 0.4% (10%), needed case splits 25%
  (35–40%), the inner negation reused in 18% of T3 rows (about half), classical-only 25% (15–20%, cap relaxed). Irrelevant
  statement material remains common (§ Open issues).

## Why

The RL pool `rl_hard12.jsonl` (5,000 problems from Dan's generator) is nearly exhausted: RL'd checkpoints solve
about 93% pass@1 on it, but only 33–40% of textbook problems of the same nominal length. The
[gap analysis](../gap_analysis/README.md) explains the difference:

- the pool's length is nominal (57% of its "7–12-line" problems have a proof of ≤ 6 lines);
- failed textbook problems are compact statements with an irreducibly long shortest proof;
- they need nested negation boxes, reductio with real work inside, depth-2 elimination work and case splits that are
  actually needed;
- the pool carries artifacts that textbook problems lack.

This generator implements the analysis' targets T1–T8. Leon approved the direction on 2026-10-08.

## Design

Code: this folder, branch `leon/thinking` (uncommitted, on top of `864ee17`). The exact code of the production
run is copied to `work/src_snapshot/` with `SHA256SUMS`.

| file | does |
|---|---|
| [`ptree.py`](ptree.py) | ND proof trees; linearisation into the fork's line format with sharing and hoisting; nd_verify re-check |
| [`pfgen.py`](pfgen.py) | the proof-first generator: recipes, the backward normal-proof builder, substitution, discharge |
| [`pfgates.py`](pfgates.py) | shape gates (T2, T7, relevance, textbook instances), the shortest-proof gate (T1), structure and necessity (T3–T6) |
| [`excl.py`](excl.py) | the exclusion index (364 files) |
| [`ext.py`](ext.py) | loads the reused `fastlen`, `gap_analysis/common` and `gap_analysis/necessity` by path (their bare names clash with the fork's) |
| [`run.py`](run.py) | the pipeline CLI: `generate`, `lean`, `select`, `split`, `verify`, `meta` |
| [`test_hard712.py`](test_hard712.py) | unit tests (synthetic theorems only) |

**Proof first.** Each candidate is built as a natural-deduction proof tree; the theorem is read off the finished
proof. Nothing is seeded from a data file, and a seed fixes the candidate (`pfgen.sample(seed)` is deterministic).

1. **Skeleton.** A recipe (34 in `pfgen.RECIPES`) fixes the box nesting at the top of the proof and the kind of
   work inside:
   - nested negation: `IN` (goal A > ~B), `INO` (the inner contradiction is a case split), `INR` / `G_inr` (the inner
     ~B is reused by an elimination), `NN` (N(N())), `CN` (reductio around a negation box);
   - reductio with work: `CW` (an elimination chain), `CO` (a case split inside);
   - case splits inside a box: `IO`, `IOX` (branches routed to Or.inl / Or.inr), `IIO`, `NO`, `G_split_ante`
     ((X v Y) > Z, split on the assumption), `G_imp_split`, `G_neg_split`;
   - depth-2 work: `II`; (A>B)>C goals: `HO`, `G_ho`;
   - goal-shape recipes `G_*` (only the goal's shape is fixed; the builder chooses the boxes);
   - `B_*`: the same recipes with a biconditional premise (X > Y) & (Y > X), kept only when both directions are
     used.
2. **Filling.** Box hypotheses are small random formulas (0–2 connectives) over 2–3 atoms. The work inside is built
   backward by a randomised normal-proof builder (`pfgen.Builder.prove`):
   - introduction rules for compound goals;
   - elimination paths (ANDE / IMPE chains) out of available formulas;
   - contradictions X / ~X where X is built from the hypotheses;
   - case splits whose two branches both use their case;
   - reductio.

   A fact that is not available may come from a new premise (at most 2; statement ≤ 7 connectives). Every new premise
   is shaped so that using it consumes an open hypothesis, so no premise restates the goal. Move weights are jittered
   per construction (×0.3–3) for variety.
3. **Substitution.** With probability 0.7 (and 0.35 for a second time) one atom is replaced in every line by a literal
   or a one-connective formula. Uniform substitution keeps an ND proof valid. It is applied only when the statement
   stays within the shape limits.
4. **Discharge.** With probability 0.8 premises become antecedents of the goal (the deduction theorem on the tree:
   P |- G becomes |- P > G, two premises curried or conjoined), within the shape limits. Separately, 60% of
   non-biconditional constructions allow no premises at all. These give the premise-free share (T2) and move
   top-level case splits on a premise into splits on an assumption inside a box.
5. **Linearisation.** `ptree.linearize` writes the tree with sharing (a formula on a citable line is never derived
   again) and hoisting (a sub-derivation that needs no hypothesis of a box is written before the box), then nd_verify
   re-checks it.
6. **Canonical form.** `key` / `pkey` are `code.thinking.data.class_keys` (atom renaming, premise permutation).

## Gates

Every kept row passed all of these, cheapest first:

| gate | rule |
|---|---|
| shape (T2) | ≤ 2 premises, ≤ 7 connectives, 2–3 atoms, formula depth ≤ 3 |
| shape (T7) | premises jointly satisfiable; no premise redundant (the goal does not follow classically without it); no ~~ and no F in the statement; the goal is not a premise; no subformula X o X |
| relevance | ≤ 2 irrelevant atom occurrences: replacing that one occurrence by a fresh atom leaves the theorem valid (an unused conjunct of an assumption, the unused half of a biconditional) |
| textbook instances | not a substitution instance of a textbook72 theorem, and no textbook72 theorem an instance of it, premises in any order (beyond the renaming classes below) |
| constructed proof | 7–13 lines, nd_verify-valid |
| dedupe | the class (`pkey`) not seen before in this run, rejected classes included |
| exclusion | the class (atom renaming or premise permutation) is in none of the 364 indexed files (below) |
| T1 shortest proof | `fastlen.fast_minlen` (the fork's `minlen.Search` with a memo; search space = subformulas of the theorem, their negations and double negations, plus every formula of the constructed proof), iterative deepening up to the constructed length: shortest L is 7–12 and the constructed proof has ≤ L + 1 lines. The proof found is the row's `reference_proof` |
| target structure | the shortest proof has at least one of the four structures below, and it is **needed**: `gap_analysis/necessity.restricted_min` with that structure forbidden finds no proof within L + 2 lines |
| Lean | the constructed proof is accepted by Lean 4 (`code.thinking.lean.lean_check`, through the state-env decomposition the eval uses) |
| mix | the deterministic selection below |

The four structures (from the shortest proof):

- **nested_neg (T3):** a negation or reductio box inside another box, whose body has ≥ 2 elimination lines and uses
  the outer box's hypothesis. Needed: no nested negation box within L + 2.
- **reductio (T4):** a reductio box (assume ~G, F, NEGI, DN) whose body holds a box or ≥ 3 eliminations. Needed: no
  DN within L + 2.
- **case_split (T5):** a case split inside an implication, negation or reductio box, both branches ≥ 2 lines. Needed:
  no ORE within L + 2.
- **depth2 (T6):** box depth ≥ 2 with ≥ 2 elimination lines at depth ≥ 2. Needed: box depth ≤ 1 impossible within
  L + 2.

Rows record every structure that is present and needed (`families`) and a primary `family` (the recipe's intended
family when it qualifies, otherwise nested_neg > case_split > reductio > depth2).

**Mix selection (`run.py select`).** Each brief percentage is a cap on the share of the selected set. Floors are met
by taking rows with scarce attributes first; a floor the reservoir cannot meet is reported as a shortfall.

- caps: classical-only ≤ 20%, reductio ≤ 22%, nested_neg ≤ 32% and nested_neg without inner reuse ≤ 17%, (A>B)>C
  goals ≤ 12%, ex falso ≤ 10%, shortest proof ≤ 8 lines ≤ 40%, biconditionals ≤ 12%, premise-free ≤ 35%, case splits
  ≤ 42%, ≥ 1 irrelevant occurrence ≤ 40% and ≥ 2 ≤ 12%;
- scarce, taken first: case splits (floor 35%), premise-free (25%), biconditionals (10%), reused inner negation (12%).

The pass is greedy over rows ordered by a hash of the class key, scarce rows first. When the reservoir cannot fill the
requested size (`--n 26500`) at these caps, the first blocking cap in `RELAX_ORDER` is relaxed by 5 pp, and the pass
repeats. The order is the irrelevance caps, nested_neg without reuse, nested_neg, (A>B)>C goals, ≤ 8 lines,
reductio, then the scarce attributes' upper caps, and classical-only last, in 1 pp steps and never above 25%. Ex falso
is never relaxed. When nothing can be relaxed, the size shrinks to what fits, so every final cap holds. No model is
involved anywhere before the splits are drawn.

## Outputs

Everything is under `/scratch/cowifornia/nd-rl/think/runs/pools/hard712/`.

| file | contents |
|---|---|
| `pool.jsonl` | the RL training pool |
| `test_1k.jsonl` | **sealed** test split: a uniform random 1,000 of the selected problems, drawn before any model saw them |
| `dev_500.jsonl` | dev split: a uniform random 500, drawn the same way |
| `keys.jsonl`, `*.keys.jsonl` | renaming-class sidecars (`name`, `key`, `pkey`, `split`) for later contamination checks |
| `meta.json` | yields per stage and recipe, timing, family mix, histograms, the verification report |
| `work/` | intermediates: `gated.jsonl` (every gated row), `rejects.jsonl` (every other class seen, with its reason), `gated_lean.jsonl`, `selected.jsonl`, `spare.jsonl` (gated rows the mix selection left out; disjoint from all three splits), `progress.json` (counters, streams, per-round weights), `weights_stream2.json`, `finish.sh`, logs, reports (`select_report.json`, `lean_report.json`, `verify.json`, `verify_table.md`), `excl_index.pkl`, `src_snapshot/` (the exact code, with `SHA256SUMS`) |

## Sealing

`test_1k` and `dev_500` were drawn by `run.py split` (`random.Random(20261008)` shuffle of the selected rows).
No model was run on any hard712 problem before the draw; no model has been run on them since. All three splits are
disjoint by renaming class (asserted in `split`).

Every row's class (atom renaming, and premise permutation via `pkey`) was checked against the exclusion index
`work/excl_index.pkl` (`excl.py`). It holds 564,905 classes from 364 files:

- the 120 files test_13_20 was sealed against: K12 train and held-out and the cap-6 Stage-1 set; every jsonl under the
  fork's `data/` (the ladder and take-home pools and transfer sets, p2, r1–r3, bs, gt, lj, and **textbook72 dev and
  train**); the OOD sets; the K12-prompt trace files; the RL pools and candidate files under `runs/pools/`
  (**rl_hard12**, rl_hard, rl_hard_filtered, cands, long/short); **hard12_test**, **hard12_test_1k** and the hard12 work
  files; **test_21_30**;
- added here: **test_13_20** and its calibration set, all of `runs/pools/test_21_30/`, the hard12r candidate and
  pilot sets, the ood_eval_20261005 sets, every trace file under `traces/`, and the trace-generation chunks under
  `gen_runs/`.

The test `test_production_index_covers_sealed_sets` checks that textbook72, hard12_test_1k, hard12_test,
test_13_20, test_21_30 and rl_hard12 are in the index and that their rows hit it. Beyond renaming classes, no
candidate is a substitution instance of a textbook72 theorem, and no textbook72 theorem is an instance of a candidate.
textbook72 statements are read only by these exclusion checks and by aggregate reporting; no formula is generated
from them.

## Regeneration

CPU only. The run used 24 workers at `nice 10`. The machine crashed during the first production run (10:06Z); the run
was resumed from its row files at 21:29Z. The class-level dedupe makes resumption and stream switches produce the
same set.

```bash
cd /scratch/cowifornia/nd-rl/think/nd-rl
export ND_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 LEAN_GATE_WORKERS=16
X=code.experiments.current.thinking_traces_20261005.hard712
W=/scratch/cowifornia/nd-rl/think/runs/pools/hard712/work      # this run (index, recorded weights)
O=/scratch/cowifornia/nd-rl/think/runs/pools/hard712_regen     # a fresh output directory
mkdir -p $O/work && cp $W/progress.json $O/work/progress_ref.json
nice -n 10 code/.venv/bin/python -m $X.excl --out $W/excl_index.pkl   # only if the index is missing
G="nice -n 10 code/.venv/bin/python -m $X.run --out $O generate --workers 24 --round 2000000 --chunk 20000"
G="$G --hours 99 --target 999999 --index $W/excl_index.pkl"
$G --stop_seed 30000000                                                  # stream 1: pfgen.WEIGHTS
$G --stop_seed 42000000 --weights "$(cat $W/weights_stream2.json)"         # stream 2
$G --stop_seed 120000000 --adaptive --weights "$(cat $W/weights_stream2.json)" \
   --replay $O/work/progress_ref.json                                    # stream 3, recorded weights
R="nice -n 10 code/.venv/bin/python -m $X.run --out $O"
$R lean && $R select --n 26500 --workers 16 && $R split && $R verify --workers 16 && $R meta
```

The production run executed the same steps (`work/finish.sh` for the last line). `--replay` reuses stream 3's
recorded per-round weights. Without it the adaptive weights are recomputed from the rows, which reproduces them unless
a search timeout (20 s, 73 classes) fell differently.

Tests:

```bash
code/.venv/bin/python -m pytest -q -p no:cacheprovider \
    code/experiments/current/thinking_traces_20261005/hard712/test_hard712.py
```

## Yields

Funnel of the production run (`work/progress.json`, `work/rejects.jsonl`; seeds 0–120M):

| stage | count | share |
|---|---|---|
| seeds | 120,000,000 | |
| construction failed (a recipe's move found no proof within budget, or degenerate choices) | 107,466,301 | 89.6% of seeds |
| shape gate rejects (candidates; largest: constructed length 2.38M, redundant premise 1.94M, > 2 irrelevant occurrences 1.73M, goal is a premise 1.35M, depth 0.89M, atoms 0.82M, textbook instance 157k / textbook general 23k) | 9,465,210 | |
| passed the shape gate | 3,068,489 | 2.6% of seeds |
| new renaming classes | 71,397 | |
| excluded by the index (0 hits on textbook72, test_13_20, test_21_30 or hard12_test_1k; 10 on the K12 held-out set, 6 on OOD sets, 1 on rl_hard12, the rest K12-train / trace / candidate files) | 564 | 0.8% of classes |
| certified | 70,833 | |
| rejected: constructed proof ≥ 2 lines longer than the shortest | 17,376 | 24.5% of certified |
| rejected: no target structure in the shortest proof | 12,915 | 18.2% |
| rejected: shortest > 12 / < 7 lines | 4,847 / 3,358 | 6.8% / 4.7% |
| rejected: structure present but not needed within L + 2 | 2,722 | 3.8% |
| search timeout (20 s) | 73 | 0.1% |
| **gated** | **29,542** | **41.7% of certified** |
| Lean (constructed / reference proofs) | 29,542 / 29,542 | 100% / 100% |
| **selected** (mix gate) | **23,092** | 78% of gated |

**Timing.** Generation took 171 min; Lean, selection, splits, verification and meta took 4 min.

**Three weight streams.** Unique classes grow only about as seeds^0.6, and recipes saturate at different rates. Late in
stream 1, 27% of seeds went to biconditional recipes that no longer produced anything.

- **Stream 1** (seeds 0–30M): `pfgen.WEIGHTS`. Per 2M-seed round, accepted rows fell from 2,266 to 359.
- **Stream 2** (30–42M): `work/weights_stream2.json`, weighted towards the recipes that were still producing. 874
  falling to 586 per round.
- **Stream 3** (42–120M): adaptive. Before each round every recipe's weight was set proportional to its recent
  accepted value per seed (classical-only rows count 0.1, scarce attributes add), with a 0.05 floor. The weights are
  recorded per round in `progress.json`. 952 falling to about 210 per round.

The machine crashed during the first production run, at 10:06Z after 4 min. Generation resumed from the row files at
21:29Z.

**Per recipe** (certified classes → gated → selected; `meta.json` → `yields.by_recipe`):

| recipe | certified | gated | selected | | recipe | certified | gated | selected |
|---|---|---|---|---|---|---|---|---|
| G_imp | 17,166 | 6,030 (35%) | 4,284 | | IN | 1,133 | 672 (59%) | 670 |
| G_or | 15,192 | 5,263 (35%) | 3,158 | | G_imp_neg | 1,047 | 598 (57%) | 595 |
| G_imp_split | 5,402 | 2,886 (53%) | 2,577 | | G_neg_split | 934 | 503 (54%) | 503 |
| G_ho | 6,146 | 2,502 (41%) | 1,961 | | CO | 2,842 | 814 (29%) | 496 |
| IO | 3,442 | 1,874 (54%) | 1,712 | | NO | 1,242 | 476 (38%) | 473 |
| G_imp_imp | 2,631 | 1,776 (68%) | 1,610 | | G_neg | 1,198 | 410 (34%) | 410 |
| G_split_ante | 3,452 | 1,420 (41%) | 1,289 | | CN | 813 | 290 (36%) | 265 |
| CW | 3,864 | 1,469 (38%) | 790 | | HO | 728 | 368 (51%) | 257 |
| II | 1,102 | 845 (77%) | 773 | | INO / G_inr / INR | 640 / 189 / 148 | 179 / 140 / 87 | 175 / 140 / 87 |
| IOX | 876 | 748 (85%) | 680 | | IIO / G_bicond / NN | 76 / 349 / 151 | 62 / 50 / 46 | 57 / 50 / 46 |
| B_* (all) | 70 | 34 | 34 | | | | | |

**Per family** in the pool: the primary family is depth2 39.9%, nested_neg 28.0%, case_split 22.4% and reductio 9.7%.
Any family present and needed: nested_neg 29.5%, reductio 17.9%, case_split 24.7%, depth2 85.7%.

## Mix achieved

% of the 23,092 selected rows (pool, test and dev; the three splits agree within 4 pp, `meta.json` → `mix`). "Gated"
is the 29,542-row reservoir before the mix gate. The final caps after relaxation are in `work/select_report.json`.

| target | brief | selected | gated |
|---|---|---|---|
| T1 shortest proof 7–12, ≤ constructed + 1 | all | 100% (gate) | 100% |
| T1 shortest 9–12 lines | ≥ 60% | 90% | 92% |
| T2 ≤ 7 connectives, ≤ 2 premises, 2–3 atoms, depth ≤ 3 | all | 100% (gate) | 100% |
| T2 no premises | 25–35% | **8.8%** | 6.9% |
| T3 needed nested negation box with ≥ 2 inner eliminations using the outer hypothesis | 25–30% | 29.4% | 32.3% |
| T3 … with the inner ¬ result reused | about half of T3 | **18%** of T3 (5.3% of all) | 13% |
| T4 needed reductio with work inside | ~20% | 18.0% | 26.1% |
| T4 classical-only (G4ip) | 15–20% | **25.0%** (cap relaxed from 20%) | 37.9% |
| T5 needed case split in a box, both branches ≥ 2 lines | 35–40% | **24.8%** | 19.4% |
| T5 … split on an assumption | some | 36% of T5 | |
| T6 box depth ≥ 2 | ~70% | 97.6% | 95.9% |
| T6 elimination lines at depth ≥ 2 (mean) | ≥ 2 | 2.48 | |
| T7 consistent premises / no redundant premise / no ~~ | all | 100% (gate) | 100% |
| T7 ex falso in the shortest proof | ≤ 10% | 10.0% | 11.3% |
| T8 biconditional | ~10% | **0.4%** | 0.3% |
| T8 goal (A>B)>C | ~10% | **24.5%** (cap relaxed to 27%) | 23.0% |
| shortest proof ≤ 8 lines (cap 40%) | | 9.8% | 8.5% |
| ≥ 1 / ≥ 2 irrelevant atom occurrences (dev58 7–12: 20% / 10%) | | **81% / 46%** (caps relaxed fully) | 82% / 48% |

**How the mix gate relaxed.** It could not fill 26,500 rows at the targets. It relaxed caps in its fixed order: the
irrelevance caps to 85% / 47%, nested_neg without reuse to 27%, (A>B)>C goals to 27%, and classical-only in 1 pp
steps to its 25% ceiling. Ex falso was never relaxed. It then shrank to the 23,092 rows that fit.

## Feature comparison

The gap analysis' feature table, recomputed by `run.py verify` (`gap_analysis/common.thm_features` and
`proof_features` on the reference proof):

- a random 2,000 of the new pool;
- the rl_hard12 pool (shortest-proof features from `runs/gap_analysis/features.jsonl`);
- textbook dev58 at reference length 7–12, **aggregates only**.

The last two rows are the relevance measure introduced here. Copy: `work/verify_table.md`.

| feature | hard712 pool sample (n=2,000) | rl_hard12 pool (n=5,000) | dev58 7–12 (n=20) |
|---|---|---|---|
| no premises | 9% | 2% | 30% |
| goal is an atom | 5% | 6% | 20% |
| goal is a negation | 6% | 10% | 20% |
| goal is a disjunction | 12% | 9% | 10% |
| goal is an implication | 76% | 55% | 45% |
| goal is (A>B)>C | 24% | 4% | 15% |
| biconditional (A>B)&(B>A) inside | 0% | 0% | 10% |
| negated compound ~(A o B) inside | 23% | 49% | 30% |
| double negation ~~A inside | 0% | 43% | 0% |
| classical-only (G4ip) | 24% | 4% | 15% |
| premises jointly inconsistent | 0% | 33% | 0% |
| shortest proof >= 9 lines | 90% | 8% | 70% |
| <= 7 connectives and shortest >= 9 lines | 90% | 1% | 70% |
| reductio, or a negation box inside another box | 45% | 2% | 35% |
| negation / reductio box nested inside another box | 40% | 1% | 25% |
| ... inside a negation / reductio box | 6% | 0% | 15% |
| ... inside an implication box | 29% | 1% | 10% |
| plain negation intro (NEGI, no DN) | 31% | 4% | 45% |
| reductio | 25% | 1% | 20% |
| case split (ORE) | 59% | 3% | 45% |
| case split inside a negation / reductio box | 11% | 0% | 20% |
| both case branches >= 2 lines | 27% | 1% | 35% |
| case split on an assumption | 19% | 0% | 15% |
| box depth >= 2 | 98% | 17% | 75% |
| no box at all | 0% | 44% | 0% |
| implication box | 79% | 50% | 45% |
| ex falso | 9% | 42% | 5% |
| a negation-intro result reused by IMPE / NEGE | 8% | 0% | 10% |
| a box result reused by an elimination | 40% | 6% | 20% |
| first step forward (not an assumption) | 0% | 46% | 5% |
| >= 1 irrelevant atom occurrence | 81% | 99% | 20% |
| >= 2 irrelevant atom occurrences | 46% | 97% | 10% |
| **mean** connectives | 6.45 | 11.39 | 4.90 |
| **mean** premises | 1.23 | 2.24 | 1.10 |
| **mean** distinct atoms | 2.81 | 3.12 | 2.45 |
| **mean** formula depth | 2.90 | 3.96 | 2.45 |
| **mean** nested-implication depth | 1.60 | 1.76 | 1.05 |
| **mean** redundant premises | 0.00 | 1.19 | 0.00 |
| **mean** shortest proof (lines) | 10.39 | 6.36 | 9.40 |
| **mean** box depth | 2.23 | 0.76 | 1.85 |
| **mean** elimination lines at depth >= 2 | 2.48 | 0.26 | 1.95 |
| **mean** longest box (lines) | 7.89 | 2.13 | 6.55 |

**Lean.** Two checks:

- **Gate:** all 29,542 constructed proofs and all 29,542 reference proofs were accepted.
- **Verification:** a fresh check of the reference proofs of a random 1,000 pool rows accepted 1,000/1,000 (100%).

**Diversity** (all 23,092 selected rows; `work/verify.json`):

- 23,092 renaming classes (all distinct, also up to premise order);
- 65 distinct box skeletons;
- 9,175 distinct statement shapes with atoms abstracted;
- 6,208 distinct reference-proof rule sequences;
- atoms: 3 in 83% of rows, 2 in 17%.

**Most common skeletons:** I(O(|)) 26%, I(I(O(|))) 11%, I(I()) 9%, I(C()) 9%, I(N()) 9%, C(N()) 5%, N(O(|)) 5%,
C(O(|)) 3%, I(N(O(|))) 3%. Every other skeleton is under 3%. The gap analysis' failed-textbook skeletons are present
in these numbers of rows: I(N()) 2,030, N(N()) 133, C(N()) 1,252, C(O(|)) 707 and N(O(|)) 1,156. The exception is a
bare top-level O(|), which never counts as a needed target structure here.

**Row format.** Pool, test and dev rows share one format:

- the `rl_hard12.jsonl` pool fields: `name`, `thm`, `key`, `prompt`, `n_lines`, `L_true`, `minlen_bound`, `source`,
  `gen_lines`, `n_prem`, `rules`, `gen_proof`, `prompt_tokens`, `formula_depth`, `max_state` and the hard12 `f_*`
  features;
- the eval fields: `reference_proof`, `reference_lines`, `min_lines`, `family`, `knob`, `in_train_class`;
- hard712 fields: `families`, `necessity`, `structure`, `skeleton`, `attrs`, `n_irrelevant`, `recipe`, `seed`,
  `subst`, `discharged`, `lean_ok`, `lean_ref_ok`, and the full gap-analysis features (`gap_thm`, `gap_proof`).

`code.thinking.eval --sets` (through the fork's `guided_eval.load_sets`, checked), `grpo.py --targets` and `ei.py
--targets` read `name` and `prompt` (and `min_lines` for evals). The largest state is 121 tokens (hard12 gate: 386).

## Open issues

- **Size below target.** The class space of this generator under these gates saturates: unique classes grow about as
  seeds^0.6. After 120M seeds, a 2M-seed round adds about 200 gated rows.
  - The mix gate then keeps 78%, giving 23,092 instead of ≥ 25,000. The 6,450 spare rows, mostly classical-only, can
    top up the pool at the cost of the T4 mix.
  - More diversity needs new recipe families, not more seeds. Candidates: independent sub-arguments joined by ANDI,
    lemmas used twice, goals with compound antecedents built from two hypotheses.
- **Premise-free problems (9%) and biconditionals (0.4%).**
  - Discharging premises costs a connective and a depth level. Under ≤ 7 connectives and depth ≤ 3 it often fails.
  - A biconditional premise costs ≥ 3 connectives, and both directions must be used. Once one-direction biconditionals
    were (correctly) rejected as irrelevant material, the B_* recipes yielded almost nothing.
  - Either needs dedicated recipes; T8 biconditionals may need the connective cap raised to 8–9 for that share.
- **Case splits (25%) and inner-negation reuse (18% of T3)** come from narrow recipes (IO, NO, IOX, G_neg_split; INR,
  NN, CN, G_inr) that saturated first.
- **Irrelevant atom occurrences.**
  - This is a new artifact measure, added when spot checks showed unused conjuncts and half-used biconditionals. An
    occurrence is irrelevant when replacing it by a fresh atom leaves the theorem valid.
  - The gate allows ≤ 2. Still, 81% of rows have at least one, vs 20% of dev58 7–12 (38% of all dev58) and 99% of
    rl_hard12. Many come from disjunction introduction, which textbook problems also use.
  - A stricter gate cuts yield about 2.5×.
- **Skews.** Implication goals are 76% (textbook 45%), atom and negation goals are rare, (A>B)>C goals are 24%, box depth
  ≥ 2 is 98% (target ~70%) and 3 atoms 83%. Depth-1 problems with a needed target structure are rare in this design.
- **Difficulty is structural, not measured.** No model was run (as briefed). Whether these problems are hard for the
  RL'd checkpoints is untested. A T0 / RL'd pass-rate measurement on `dev_500` is the obvious next step. `test_1k` stays
  sealed for the final comparison.
- **The shortest proof is within the search space:** subformulas, their negations and double negations, plus the
  constructed proof's formulas. A shorter proof through other formulas is not ruled out (the same caveat as the gap
  analysis). Necessity is "none within L + 2 lines" in the same space.
- **The splits are uniform samples of the mix-gated set**, not of the 29,542-row reservoir. The mix gate is deterministic
  and model-free; a test drawn from the reservoir would not match the pool's distribution.
- **dev58 was a design set** (the gap analysis chose these targets from it). Use a fresh textbook-like set for claims
  about transfer.
- **Abandoned runs.** `work/run1_abandoned/` and `work/run2_abandoned/` hold two abandoned pilot runs from earlier
  generator versions; their rows are in no output file.
- **`make -C code check` fails** only on another session's uncommitted `collapse_audit/` files (ruff: 44 errors, 8
  format). This folder passes ruff check and format, and pyright reports 0 errors.

**Checks run.**

- `test_hard712.py`: 25 passed. It covers:
  - generator determinism and nd_verify validity;
  - linearisation, including hoisting;
  - discharge and substitution;
  - every shape gate on hand-built cases;
  - relevance;
  - substitution-instance matching;
  - certify on hand-built T1 / T3 / T5 cases;
  - key equality with `code.thinking.data.class_keys`;
  - index hits on a renamed and permuted theorem, and coverage of the sealed sets in the production index;
  - the crash-safe reader.

  Every fixture was checked against textbook72's renaming classes.
- Lean: 100% (above).
- The fork's eval loader reads `test_1k.jsonl`.
- **Backup** (NFS, since `/scratch` is machine-local): `/home/cowifornia/backups/nd-rl-misc/hard712_20261008/` holds the
  splits, keys, meta and the work reports and source snapshot.
