# Agent B: autopsy of the looped model's failed attempts, and what extra loops fix

Model: v2 `looped-ei` seed 0, `E/resume/looped-ei_s0/final.pt` (`ND_ARCH=looped`). All sampling on this Mac (CPU),
T=0.8, Lean-judged through the harness's `sample()`. Everything in `scratchpad/probe/B/`:
`targets.jsonl` (65 targets), `samples_L{1,2,4,8}_k16_s0.jsonl` (+ `.classified.jsonl`), `samples_PRE_L2_k8_s0.jsonl`
(pretrain-only checkpoint), `goldlogp.jsonl`, `examples.txt`, scripts `run_samples.py analyze.py factexists.py
goldlogp.py devfeats.py trend.py`.

## TL;DR

1. **Sample level.** Failed attempts are not near misses. 93 % of the Lean-rejected samples *state the goal* on
   their last top-level line; the rejection is an ill-typed step in the middle (on average 72 % of the way in). For
   74 % of those steps the fact Lean wanted was **never derived anywhere in the sample** (a bluff); only 12 % are
   name-pointer slips to a fact that did exist. The single biggest class is syntactic-looking but semantic: the model
   opens a box (→-intro, ¬-intro or an `Or.elim` branch), cannot derive what the box must end with, and closes it by
   citing *the very `have` it is defining* (circular; 23 % of all samples, 27 % of samples on never-solved targets).
2. **Target level.** On dev, holding length fixed (band 10-14), the gold proof's **nesting depth** is the predictor:
   depth 1: 75/75 solved, depth 2: 22/42, depth 3: 0/29. Proofs needing a ¬-intro box: 1/20; a DN step: 0/9;
   negated goals: 0/15; proofs with no modus ponens at all (pure ∧/∨/¬ shuffling): 0/15. On old dev the unsolved
   textbook schemas are the classical ones: excluded middle 0/19, ∧-over-∨ distribution (conv) 0/26, Peirce 5/20,
   negated conditional 6/18, De Morgan ∨→¬∧ 8/18; export/import/disjunctive syllogism/contraposition are ≥ 94 %.
3. **Loops.** 1 → 2 loops is the big step (106 → 136 accepted samples of 1,040, 12 → 14 targets); 4 and 8 add
   little (146, 141; 15 targets). Loops reduce *local* errors (ill-typed modus ponens 133 → 61, hallucinated names
   68 → 26, wrong-disjunct case splits 52 → 35) and leave the *planning* errors untouched (circular box 242 → 243,
   scope leak 62 → 103, one-armed case split 66 → 102, restating rambles 109 → 155: the model gets further before it
   fails, then fails the same way). 49/65 targets are solved at no loop count. The RL-time "retries crack 15-18 % at
   2/4/8 loops vs 3-4 % untied" effect is mostly (a) 24 extra samples and (b) 1 loop being a crippled setting of a
   model trained at 1-8 loops (first-pass accuracy 0.37 vs untied 0.49), not depth: 2 vs 8 loops crack 14.9 % vs
   17.5 % of all-fail targets, and the extra cracks are short ladder theorems (0-6 and 7-9 bands), not long ones.

## Setup and what was run

Targets (65, `targets.jsonl`, seed 7, stratified by pool, length and generator family):

| pool | n | what |
| --- | --- | --- |
| old_unsolved_short | 14 | old dev (Dan's ladder/textbook), 6-9 body lines, 0/16 at eval |
| old_unsolved_mid | 6 | old dev, 10-12 body lines, unsolved |
| dev_unsolved_short / _mid | 10 / 6 | dev, body ≤ 14 / 15-20, 0/64 at eval |
| rl_unsolved_short_{ladder,d,s} | 6 / 3 / 3 | RL pool, ≤ 12 lines, never found in RL (ladder: Dan's statements, no gold) |
| rl_unsolved_mid_{c,s,d,b} | 2 / 2 / 2 / 1 | RL pool, 13-20 lines, never found |
| old / dev / rl solved controls | 3 / 4 / 3 | solved at eval, low n_ok where possible |

Runs: k=16 per target at n_loops 1, 2, 4, 8 (1,040 samples each; 1.5 min at 1 loop, 9 min at 8 loops on 3 threads),
k=4 pilots, and k=8 at 2 loops from `pretrain.pt` (520 samples) as the before-RL comparison. Lean error messages
come from re-running Lean on every rejected sample's literal text (`analyze.py:lean_errors`) and mapping the error
column back to the `have` and the rule used there. Parse failures were re-decoded with a copy of the strict decoder
that records how far it got, the box path, and whether the cited name is the enclosing `have`, a name bound in a
closed box, or nothing (`analyze.py:partial_inverse`).

Note on "unsolved in RL": an RL target is seen only ~2 times × 8 samples in 3 h, so "never found" is weak evidence;
the `rl_unsolved_short_s` targets were solved 43/48 at k=16 here. Dev/old-dev "unsolved" (k=64 / k=16 at eval) held
up: 0/16 at every loop count for all 36 of them.

## Taxonomy of failed attempts (sample level)

Counts are samples out of 1,040 at 1 loop / 8 loops (k=16 × 65 targets). "Never-solved" is the subset of 784
samples on the 49 targets no loop count solved. The pretrain column is the pre-RL checkpoint at 2 loops, k=8 (of 520).

| # | class | how it is detected | L1 | L8 | never-solved L1 / L8 | pretrain |
| --- | --- | --- | --- | --- | --- | --- |
| A | **Circular box**: a box ends with `exact nK` where `nK` is the `have` being defined (`have n80 : A → C := ( fun n81 : A => by ... ; exact n80 )`), or `have n53 : X := Or.elim n53 ...` | decoder `unbound`, name is the open `have` | 242 | 243 | 215 / 214 | 68 |
| B | **Ill-typed modus ponens / ¬-elim**: `nF nA` where `nA` is not F's antecedent | Lean app-mismatch at an IMPE term | 133 | 61 | 89 / 50 | ~150 |
| C | **Ramble**: hits the token cap without `<eos>`; 72 % of the lines in these samples restate an earlier line | no eos | 109 | 155 | 98 / 125 | 6 |
| D | **Ill-typed intro/elim**: ⟨a,b⟩ of wrong components, `Or.inr` of the wrong formula, `.1/.2` claiming the wrong conjunct, `R`/`False.elim` claiming the wrong formula | Lean type/app mismatch | 93 | 92 | 67 / 85 | ~150 |
| E | **One-armed case split**: `Or.elim nJ ( fun ... exact ) ;` with no second branch, then carries on | decoder `expected (` at `;` | 66 | 102 | 62 / 101 | 3 |
| F | **Scope leak**: cites a fact derived in a closed box (90 % a sibling `Or.elim` branch, 10 % a box already closed at the enclosing level) | decoder `unbound`, name bound on a closed box path | 62 | 103 | 54 / 83 | 19 |
| G | **Wrong case split**: branch hypothesis is not the disjunct (`Or.elim` on a conjunction whose second part is a disjunction, or on a disjunct the model wishes it had) | Lean app-mismatch at `Or.elim` | 52 | 35 | 44 / 28 | — |
| H | **Hallucinated / forward name** (an `n` never bound, or bound later) | decoder `unbound` | 68 | 26 | 63 / 23 | 21 |
| I | **Ill-typed DN** (`Classical.byContradiction` claiming a formula whose ¬¬ it does not have) | Lean | 23 | 30 | 20 / 28 | — |
| J | other grammar (missing `;`, `)`, bad formula, ORE box ending in False, ...) | decoder | 66 | 46 | 54 / 41 | ~70 |
| — | accepted | Lean ok | 106 | 141 | 0 / 0 | 0 |

Three cross-cutting observations:

- **The goal is always "reached".** Among Lean-rejected samples (fully grammatical), 298/321 end with the goal on the
  last top-level line; among parse-failed samples only 32 % ever state the goal. Classes B/D/G/I are *bluffs that
  look complete*; A/E/F are *collapses that never get there*.
- **Bluffs are not pointer errors.** For the Lean-rejected steps, the fact Lean expected was never derived anywhere
  in the sample in 183/246 cases (74 %), appeared earlier (a mis-citation) in 30 (12 %), later in 33 (13 %). Among
  the IMPE mismatches, "unrelated" formulas dominate (85/133); polarity flips (`¬R` fed where `R` was expected) are
  8/133. The model is not confusing `P` with `¬P`; it applies an implication to whatever is handy.
- **RL changed the failure mode.** Before RL (pretrain.pt, ≤ 6-line training data) the model fails by local type
  errors (B+D ≈ 60 %, plus 59 "projection of a non-∧") and hardly ever rambles (6/520) or leaves a case split
  one-armed (3/520). After 3 h of EI, local errors drop and failures move to restating loops (C), circular boxes
  (A), one-armed splits (E) and scope leaks (F): the RL'd model has learned the *shape* of a long proof (many boxes,
  many lines) without the content. In failed samples 45 % of all `have` lines duplicate an earlier line's formula
  (restating via `R` or re-deriving); in accepted dev proofs padding is small (written 16.5 vs pruned 15.0 lines,
  gold 12.8), so restating is specifically what the model does when stuck.

### Concrete examples (full texts in `examples.txt`)

**A. Circular box.** `la_transfer_1101` (old dev, 6 lines; premises `(P∧P)∨(S∧S)`, `¬R`, `¬(S∧S)`; goal
`¬¬(R∧S) → ((R→Q) → Q)`). Sample: opens the two →-boxes correctly, then
`have n135 : Q := Or.elim n126 ( fun n136 : (P∧P) => by have n137 : False := n130 n135 ; ...` — inside the first
branch it cites `n135`, the `Q` it is in the middle of proving, as the argument to `n130 : ¬¬(R∧S)`. It never
derives `R∧S` (real proof: from `¬R` get `¬(R∧S)`, contradiction with `¬¬(R∧S)`, ex falso). Same target, class B:
`have n133 : Q := n130 n125` with `n130 : R→Q` and `n125 : ¬R` — modus ponens on the negation of the antecedent.

**A, needed fact derived one box too deep.** `la_transfer_1119` (solved control, 7 lines, goal `R → (¬¬P → S)`):
`have n95 : ¬¬P → S := ( fun n96 => by have n97 : R → S := ( fun n98 => by have n99 : S := ... ; exact n99 ) ;
exact n95 )` — it proves `R → S` *inside* the `¬¬P → S` box instead of `S`, so `S` only exists one box deeper, and
it closes the outer box with its own name. 14/57 circular closures at 8 loops had the needed formula derived, always
at a deeper nesting level.

**E + G. One-armed, wrong case split.** `la_transfer_86` (old dev, `dist_and_over_or`, premise
`¬P ∧ (¬¬Q ∨ ¬(R∧P))`, goal `(¬P ∧ ¬¬Q) ∨ (¬P ∧ ¬(R∧P))`): `have n22 : ¬P ∧ ¬(R∧P) := Or.elim n20 ( fun n23 : ¬P =>
by ... exact n25 ) ; have n31 : goal := Or.inr n22 ; exact n31`. It splits on `n20`, the *conjunction*, not its
right conjunct (it "sees through" the ∧ without a `.2`), writes one branch and moves on. The schema is 3/18 solved.

**F. Scope leak.** `la_transfer_1842` (old dev, 6 lines; premises `P`, `(¬P ∧ (P∧R)) ∨ ¬P`; goal `¬¬R → Q`):
a correct `Or.elim` whose second branch derives `n10 : False := n9 n4`; then
`have n12 : ¬¬R → Q := ( fun n13 => by have n14 : Q := False.elim n10 ; exact n14 )` — `False` was proved inside
the earlier branch (under hypothesis `¬P`) and is reused outside it.

**C. Ramble.** Same target: `have n27 : (¬P ∧ (P∧R)) := ⟨n22, n26⟩ ; have n28 : ... := ⟨n22, n26⟩ ; have n29 :
... := ⟨n22, n26⟩ ; have n30 : Q := Or.elim n23 ( ... have n33 : Q := Or.elim n23 ( ...` — the same ∧-intro three
times, then the same case split nested inside itself, to the 384-token cap. The first `⟨n22, n26⟩` also claims
`¬P ∧ (P∧R)` from `n22 : P`, a bluff Lean would reject anyway.

**I. Ill-typed DN.** Same target: `have n20 : (P ∧ R) := Classical.byContradiction ( fun hh => n19 hh )` with
`n19 : ¬¬R`: it wants `P ∧ R`, has `¬¬R`, and writes the DN step as if `¬¬R` were `¬¬(P∧R)`.

## Taxonomy of unsolved targets (what kinds of proofs)

Dev (843 theorems, k=64, 8 loops; solve rate by gold-proof feature *within* the 10-14-line band, so length is held
roughly fixed; `devfeats.py`):

| feature of the gold proof | solved / n (band 10-14) | band 15-19 |
| --- | --- | --- |
| nesting depth 1 / 2 / 3 / 4 | 75/75 / 22/42 / 0/29 / — | 20/24 / 14/41 / 4/50 / 0/30 |
| number of case splits (ORE) 1 / 2 / 3 | 87/94 / 9/39 / 0/12 | 28/69 / 11/29 / 0/7 |
| uses a ¬-intro box (NEGI) | 1/20 vs 100/130 without | 1/44 vs 38/106 |
| uses DN (byContradiction) | 0/9 | — |
| uses ex falso (BOTE) | 17/35 vs 84/115 | 5/20 vs 34/130 |
| any of NEGI/DN/BOTE ("classical flavour") | 18/55 vs 83/95 | 6/64 vs 33/86 |
| goal is ∧ / atom / → / ¬ | 33/34 / 52/83 / 16/18 / 0/15 | 6/7 / 21/36 / 12/96 / 0/11 |
| no modus ponens at all (n_IMPE = 0) | 0/15 | 0/11 |
| closing rule ANDI / IMPE / IMPI / ORE | 33/34 / 30/34 / 16/17 / 22/65 | 6/7 / 12/14 / 12/96 / 9/30 |
| generator family d / s / s2 (10-14); c (15-19) | 1/36 / 35/43 / 64/68 | c 2/88, d 0/13, s 15/20, s2 22/28 |

Old dev (1,108 theorems, k=16; textbook half by schema): `excluded_middle` 0/19, `dist_and_over_or_conv` 0/26,
`dist_and_over_or` 3/18, `dist_or_over_and_conv` 4/16, `peirce_sequent` 5/20, `negated_conditional` 6/18,
`dist_or_over_and` 7/20, `demorgan_or_to_nand` 8/18, `demorgan_nand_to_or` 9/19, `peirce` 12/18; everything else
(contraposition both ways, De Morgan ∧↔¬∨, export, import, disjunctive syllogism, constructive dilemma) ≥ 94 %, and
the generated half 696/742. Unsolved old dev by length: 6-line 9/183, 7-line 15/260, 8-line 27/320, 9-line 69/194,
11-line 48/56.

Four target classes, with a falsifiable hypothesis each:

| class | what the proof needs | evidence | old-dev schemas | fails because → would be fixed by |
| --- | --- | --- | --- | --- |
| T1 Classical / indirect | a ¬-intro box, DN, or ex falso from a contradiction between premises; goal often a negation | negated-goal 10-14s 0/15; NEGI 10-14s 1/20; DN 0/9; gold BOTE lines cost 10.9 nats each on unsolved vs 0.3 on solved | excluded middle, Peirce, negated conditional, De Morgan ∨→¬∧ | ≤ 6-line pretraining proofs contain NEGI/DN only as one-step patterns; the model never learned "assume the opposite, hunt a ⊥". Fix: pretraining + RL pools with the NEGI/DN share matched to the targets. Test: EI on 2,000 `has_NEGI` 7-12-line targets; negated-goal dev rate vs matched non-NEGI controls. |
| T2 Nested case analysis | depth ≥ 3, or ≥ 2 case splits whose branches themselves split or introduce | depth-3 10-14s 0/29; 3-split 10-14s 0/12; sample classes A/E/F are exactly "lost track of which box owes what" | distribution (∧ over ∨, both ways) | the surface form has no explicit obligation. Fix: emit the box's required conclusion when it opens (`fun ( n : A ) [ C ] =>`), both branch hypotheses before the first `Or.elim` branch. Test: retrained tokenizer, depth-3 dev rate (predict > 20 % if bookkeeping, ~0 if planning). |
| T3 Structural shuffling | no modus ponens: only ∧/∨/¬ intro/elim and re-association | n_IMPE = 0: 0/15 (10-14), 0/11 (15-19); family d 1/36; gold ANDI lines cost 6.9 nats vs 1.5 | distribution, De Morgan | needs a backward plan (which components to pair); nothing in a ≤ 6-line proof requires one. Fix: backward-generated (goal-first) training proofs; test by family d / c dev rates. |
| T4 Long ∨-chains (families b/c) | 20+ lines, 2-3 splits over fresh atoms, 3-10 premises | b/b2/b3 20-24: 26/120; c 15-24: 2/118 | — | mostly T2 at scale; one 20-line b target did flip with loops (rl_2071: 0/16 at 1-2 loops, 9-10/16 at 4-8), so some are compute-limited. Test: 2 loops × k=24 vs 8 loops × k=6 on band 20-24. |

Gold-proof likelihood probe (`goldlogp.jsonl`, 207 dev theorems): per-line nats of the gold proof, solved vs
unsolved: ORE 6.1 vs 7.0, ANDI 1.5 vs 6.9, IMPE 1.1 vs 4.1, BOTE 0.3 vs 10.9, DN — vs 4.2, NEGI — vs 6.2, PR 0.95
both. The first structural disagreement with gold sits at the same place (median 44 % of the proof) at 1 and 8
loops; 8 loops lowers total gold −log p by 4.1 nats on average (median 1.9; better on 65 % of theorems): loops make
the model slightly more confident in the right proof but do not move where it diverges.

## What extra loops fix, and what they never do

k=16 on the 65 targets (`trend.py`):

| loops | accepted samples /1040 | targets solved /65 | of which from "unsolved" pools |
| --- | --- | --- | --- |
| 1 | 106 | 12 | 5 |
| 2 | 136 | 14 | 6 |
| 4 | 146 | 15 | 7 |
| 8 | 141 | 15 | 7 |

Class counts on the 49 never-solved targets, L1 → L2 → L4 → L8:

- shrink: ill-typed modus ponens 89 → 60 → 48 → 50; hallucinated names 63 → 30 → 18 → 23; wrong-disjunct splits
  44 → 28 → 20 → 28; other grammar 54 → 49 → 44 → 41.
- grow: ramble 98 → 119 → 125 → 125; one-armed split 62 → 78 → 94 → 101; scope leak 54 → 69 → 82 → 83; ill-typed
  intro/elim 67 → 88 → 102 → 85.
- flat: circular box 215 → 226 → 215 → 214; DN misuse 20 → 34 → 31 → 28.

Interpretation: more loops buy *local* correctness (the right antecedent, a real name, the actual disjunct), so the
sample survives longer, then dies of the planning failures (does not know what the open box owes, forgets the second
case, reuses a case-local fact). The targets that moved with loops (rl_2071, 20-line family b, 0 → 10/16; rl_5149,
16-line s, 1 → 6/16; three solved controls +5 to +7 of 16) are long, flat ∨-chains: many local steps, shallow
obligations. Nothing in T1-T3 moved.

RL-time retries (`summary.json` `rl.stats.retries`, all four runs). Each round: 1,500 targets × 8 samples at 1 loop;
the all-fail targets are split at random three ways and retried × 24 at 2, 4 or 8 loops (untied: same schedule,
the knob is a no-op, so untied's retries measure "24 more samples"):

| arm | first-pass acc @1 loop | cracked at 2 / 4 / 8 loops (share of all-fail targets) | retry sample acc 2 / 4 / 8 |
| --- | --- | --- | --- |
| looped s0 | 0.371 | 935/6273 = 14.9 % / 15.8 % / 17.5 % | 0.046 / 0.063 / 0.072 |
| looped s1 | 0.372 | 15.3 % / 18.8 % / 17.8 % | 0.058 / 0.084 / 0.079 |
| untied s0 | 0.485 | 3.4 % / 3.3 % / 3.8 % | 0.0035 / 0.0033 / 0.0037 |
| untied s1 | 0.502 | 4.1 % / 4.1 % / 4.5 % | 0.0041 / 0.0038 / 0.0044 |

The looped model's retries succeed 4-5× more often than untied's, but (i) its 1-loop first pass is much weaker
(0.37 vs 0.49; pretrained at 1-8 loops with default 2, so its all-fail set is full of targets a normal setting
solves), and (ii) 2 vs 8 loops differ by 2.6 points. What the retries crack (targets whose first accepted proof came
in a retry round, looped s0): 1,949 targets, 60 % ladder (Dan's few-premise statements), 20 % s/s2; bands 0-6: 786,
7-9: 793, 10-14: 261, 15-19: 66, 20+: 43 — the first-pass mix shifted *shorter*. The marginal cracks at 8 vs 2 loops
are in the 0-6 band (310 vs 214). Depth reaches no new kind of target during RL either.

## Caveats

- 65 targets, one seed, one model; sample counts are solid, per-class target counts are small. Target-level rates
  come from the full dev/old-dev evals (843 / 1,108 theorems), not from my sample.
- Classes are by *first* error (Lean's first message, or where the strict decoder stops); rejected samples average
  1.9 Lean errors. The "never derived" check ignores scope, so 74 % bluffing is a lower bound.
- Family labels s/s2 are from file names (approach_c "tiny_s"/"hunt_small_s"); I did not re-read the generators.
  `ladder` targets have no gold proof, so they are absent from the gold-feature tables.
- Not done: the untied model's failure mix (agent A), a 1-loop-trained looped model, holdout.

## What I would test next on a GPU

1. **Obligation-annotated format** (targets A/E/F, 45 % of failed samples): emit each box's required conclusion when
   it opens, and both `Or.elim` branch hypotheses up front. Retrain pretrain + 3 h EI with run 198's recipe; measure
   depth-3 dev rate (now 0/29) and the A/E/F share.
2. **Classical-flavour curriculum** (T1): add 10-20 % NEGI/DN/BOTE-closing 4-6-line proofs to pretraining (length
   cap kept) and 7-12-line ones to the RL pool. Readout: negated-goal dev rate, excluded-middle / Peirce old-dev
   schemas (0/19, 5/20).
3. **Loops vs samples at equal compute**: 2 loops × k=24 vs 8 loops × k=6 on dev bands 15-24. Prediction from the
   class trends: 2 × k=24 wins, since loops fix only the local classes and k buys more plans.
4. **Restating** (class C): track `R` lines and duplicate formulas in accepted proofs over EI rounds; if they grow,
   train on the pruned proof instead of the written one (one-line change in `rl_ei`).
