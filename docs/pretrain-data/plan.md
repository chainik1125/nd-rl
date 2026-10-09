# Pretrain-data plan: a pretraining set that lets RL reach further on textbook problems

For Claude Code agents working for Charles. Standing rules (git, Python, experiment folders, when
to stop and ask) are in the root `CLAUDE.md`. Week of Oct 3 to Oct 10, 2026. A separate workstream (Charles,
2026-10-03): the textbook sets stay the yardstick and are not changed here. Items marked
**(decide)** are Charles's call; until he answers, use the stated default and record it in
`docs/pretrain-data/questions.md`.

## 1. Scope

> Build and test pretraining (Stage-1) sets for the small proof models so that the same RL
> recipe, applied afterwards, solves more real textbook problems than it does from the current
> generator pools. Everything is measured on the textbook sets built last week, with a strict
> seed / test split, and on the sealed 72 in aggregate.

**Why now.** Last week's evaluation (`experiments/textbook-problem/eval_dan_best_state`, doc
"RL vs pretrain on textbook problems") showed that every gap the RL model has on textbook problems
lines up with a gap in its pretraining pool:

| textbook failure (RL best-state r8, 2 seeds) | pool fact (K12, 155,000 proofs) |
| --- | --- |
| all 7 four-premise problems at 0 of 64, including a 5-line modus ponens chain | premises capped at 3 in every pool (generator `max_prem 3`) |
| box-free ∧E/∧I/→E shuffles with 3+ letters never solved (Roy E6.7j, LoVe 072, lean4game 003) | ∧E in 4 % of proofs; conjunction goals 58 % (the generator builds conjunctions, rarely unpacks them) |
| solve rate halves from nesting depth 2 to 3 (60/82 → 13/37) | nesting never exceeds depth 2 |
| 79–100 % solved through 10 reference lines, 1 of 31 beyond 20 | lengths flat 2–12, nothing longer |
| accepted proofs padded by 5 lines with manufactured ∨I→∨E splits (49 % of proofs) | 11 % of pool proofs carry such a split, 27–29 % at lengths 10–12; RL amplified it |
| hypothetical-syllogism shape (P→Q ⊢ (R→P)→(R→Q)) never solved | conditional goals 8 % of the pool; nested →I with no premise driving the first step is rare |

The pretrains (cap-6 pool) show the same pattern one level down: nothing above 9 lines, 4 of 77
∨E problems. The hypothesis of this workstream is that these are data gaps, not capacity gaps,
and that a pool which closes them lets the unchanged RL recipe go further.

**In scope.** Building candidate pools (synthetic is fine here: this is training data, not the
evaluation set), auditing them against every evaluation set, Stage-1 training with Robbie's recipe
and Dan's T1 ladder held fixed, and scoring on the textbook sets. **Out of scope.** Changing the
network, the RL recipe, the tokenizer, the verifier or Lean; touching Dan's or Robbie's branches,
pods or checkpoints; using the sealed 72 or any test-split problem as a seed, template or filter.

## 2. What exists today (surveyed 2026-10-03)

**Generator** (`gen.py` in the fork; pinned copy `code/textbook_problem/fork_tools/`): forward
mode (0–3 random premises, random rules, lazy premises) and goal mode (`reach()`: decompose the
goal, never backtracks); dependency-pruned, verified, deduplicated by renaming class. Knobs:
`--min/--max` pruned length, `--per_len` cap (flat histogram), `--max_prem` (default 3),
`--long` (strict mode for 7–16 lines), and Dan's ds-generator knobs `--ore_steps --ore_boxes
--goal_only --strict --budgets --gdepth --ladder`. Atoms P Q R S only.

**Pools in use.** Pretrains: the fork's cap-6 Stage-1 set (154,990, flat 2–6). RL model: Dan's
K12 (155,000, flat 2–12: the cap-6 control set for 2–6, long mode for 7–12). RL targets: 3,000
generator theorems with generator lengths 7–16, 69 % of which have a ≤6-line minimal proof. Full
statistics are in the doc's "What the models were trained on" section.

**Textbook schemata** (`textbook_pool.py`, 26 laws: De Morgan ×4, distribution ×4, contraposition
×2, Peirce ×2, disjunctive and hypothetical syllogism, export/import, DN intro/elim, triple
negation, explosion, excluded middle, consequentia mirabilis, negated conditional ×2, constructive
dilemma, modus tollens) instantiated with random sub-formulas; used by Dan as RL targets and
transfer pools (`data/ladder/POOLS.md`: 3,797 instances with minlen labels), never as pretraining
data.

**Prior experiments on the data itself.** See section 2.1 (filled from the survey of Dan's and
Robbie's run records).

### 2.1 Survey of prior data experiments (Dan's branch, read 2026-10-03)

All of Dan's data arms were run at **cap 6, 3.3M parameters, whole-proof format**; none at cap 12,
in the state format, or with the 9.56M recipe. That is the hole this workstream fills. The record,
with the file in `fork_dan/`:

| run | what varied | finding | file |
| --- | --- | --- | --- |
| ds-generator | proof-shape distribution of the cap-6 set (C0 control, G1 more ∨E, G2 strict long generator at ≤6) | G1 was a null manipulation (∨E share fell 1.5→1.0 % because such proofs exceed 6 lines). G2 changed the distribution profoundly and transfer readiness not at all; 10 of 19 textbook schemata at 0 in every arm. Reviewer: "the generator's rule shape is not the textbook wall." | `review_ds-generator.md` |
| ds-composition | length histogram and rule quotas (A1 natural histogram, A2 quotas ∨E ≥10 % ∧E ≥8 % ⊥E ≥5 %, A3 cap 8, A4 length-heavy) | only the histogram change is supported, and only in-distribution; A2 moved one schema on one seed; **A3 (cap 8) was the only arm that wrote accepted 8+-line reductio proofs**. | `[origin/dan_ds-composition]` |
| noise-floor | four independent pools from the same generator, up to 13 seeds | frozen-ladder solves spread 62–265 (4.3×) across null cells; **≈99 % of the variance is the training run, ≈0 the data draw**; of 20 standing data findings only the two cap-8 ones survive. MDD at n = 2 is ±235 % on frozen ladder; textbook72 per-seed sd 2.44. | `NOISE_FLOOR.md`, `numbers.md` N1–N11 |
| pattern-coverage dial | frequency f of a pattern (depth-3, derived-∨E, reductio) in Stage-1 data | depth-3 RL solves 27–34 % whether f is 0, 10⁻³ or 10⁻¹; derived-∨E ≈0 at f = 0 and 22 % at f = 10⁻⁴, flat above. **A pattern needs to exist in the data, not to be frequent.** | `campaign.md`, `followup.md` |
| ignition | injecting pattern proofs at RL round 4 | one sibling proof ignites dead arms (4/4); 4 generator proofs with the pattern 3/4; the other pattern 0/4; even verifier-rejected strings with the structure ignite. Structure transfers, not correctness. | `ignition.md`, `run3.md` |
| cap-horizon, state-cap12, best-state | training cap 6 vs 8–14; state format | max accepted length ≈ 0.91 × cap + 1.4; solving by required length plateaus at cap 8 in whole-proof format; state format × cap 12 compounds (rr600 Q 233–320 vs 72–79 whole-proof); **cap 12 adds +14 to +18 textbook72 solves after T1** (best recipe 37.3 → 51.7). | `run_best_state.md`, `numbers.md` |
| stage1-dynamics | steps and data size | **"steps, not data"**: 572,759 fresh proofs perform the same as 155k seen 19.8 times; 24k steps instead of 6k moves depth-3 +25.6 pp. | `[origin/dan_stage1-dynamics]` |
| lean-format, ds-rendering | rendering | `lean_seq` beats token rendering; no rendering beats it on the ladder; Stage-1 rankings invert after RL. | `run_lean_format.md` |
| round3-run4b | 25M / 85M parameters on cap-6 data | "bought nothing". | `[branch]` |
| phase 3 precursors | generator proofs matching 66 schema sub-step templates, added ×2 to the EI mix | export 12→19 of 24; De Morgan, distribution, dilemma stay 0–2 because a ≤6-line generator only makes degenerate (X∨X) derived ∨E. | `phase3.md` |

Never tried: schema-instance proofs in Stage-1; any data arm at cap 12, in the state format or on
the 9.56M recipe; textbook-derived training data of any kind; minimal (search-found) proofs as
training data; premise counts above 3; nesting depth 3 in the data.

**Robbie's side and the project docs** (second survey, same day). Robbie's pretraining loop held
the data fixed by design ("ablating pretraining only"; only presentation order was in scope), and
every order experiment lost: short-first curriculum −39, tail curriculum −54, an extra copy of
6-line proofs −22 on his dev metric. So curriculum by *order* over the same data is a known null;
this plan changes *content*. The 2026-09-20 group meeting named the dataset "lever one" and asked
whether to use minimal-length proofs or a curated subset (P3 and P1 are those two options). The
fork's write-up lists generator limits that no knob fixes: nesting depth ≥ 4, →E chains of 4 or
more and nested ∨E are unreachable, and 97 % of derived ∨E is the degenerate (X ∨ X). Against the
sealed 72 in aggregate (local `textbook72/overview.md`): ∨E in 1.5 % of cap-6 proofs vs 29.6 % of
textbook proofs, ∧E 2.1 % vs 37 %, box depth ≥ 2 in 14 % vs 62 %. Dan's proposal 10 rejected
schema instances in pretraining as "decorative (it would show elicitation of what was put in)";
that objection is why P4 is last and why every arm here is judged on real exercises from books the
pool never saw. The literature note (`docs/literature/astra_suggestions/`) suggests exactly P1's
shape: a backward curriculum "starting near the ends of training-only proofs".

**Three consequences for this plan.**
1. **Seeds.** Two seeds resolve nothing about data; the baseline cell's two seeds differ by 1–3
   problems on our textbook sets, which is far tighter than Dan's ladder read-outs, but the plan
   still requires 3 seeds for any arm that is reported as a gain (section 4).
2. **Small slices are enough if the shape is new.** The dial says a pattern at f = 10⁻⁴ ignites
   as well as at 10⁻¹, and ignition says structure is what transfers. So P1 and P4 are tested as
   small slices (1 % and 10 %), not as wholesale replacements, and the point of each slice is the
   shapes K12 lacks, not volume.
3. **Steps, not data.** Fresh data does not beat repeated data at this scale, so no arm argues
   from size; every arm keeps 155,000 records and the same wall-clock.

**The take-home's data rules do not bind this workstream (Charles, 2026-10-03).** Dan's fork
forbids search-based provers as a source of training data and caps supervised data at 6 lines
(`fork_dan/README.md` ≈245–252, `AGENT_POLICY.md:13-14`); those rules exist so that the take-home
measures RL beyond a fixed generator, which is not this workstream's question. K12 itself is
already outside the cap rule. P1, P3 and P4 use `minlen` proofs (the textbook reference proofs are
search-found). Each proposal still states which of the fork's rules its arm is outside, so that a
result can be compared with Dan's cells on equal terms, and P2 remains the arm inside both rules.

## 3. Candidate pools (ideas, ranked by expected value per GPU-hour)

Each pool is 155,000 records, flat over its length range unless stated, so Stage-1 wall-clock and
record count match the K12 baseline and only the content changes.

**P1. Textbook sub-problem curriculum ("easier versions of textbook problems").** Every line of a
verified Fitch proof is a theorem: premises plus the open assumptions at that line entail the
line's formula, and the lines it cites are its proof. Prototype `code/pretrain_data/subproblems.py`
(2026-10-03) over the 262 scraped reference proofs: **1,359 distinct curry-aware classes, 5.2 per
seed**, verified by the pinned verifier; lengths 2–20 (54 / 145 / 213 / 206 / 181 at 2–6, 119 /
94 / 60 at 7–9, 55 / 39 / 36 at 10–12, 94 at 13+); premises 1–7 with **323 at four or more**;
25 are twins of a sealed-72 problem and 26 of another evaluation problem, both to be removed by
the audit. This is the user's "easier versions" idea made concrete, and it directly supplies the
missing shapes (4+ premises, ∧-unpacking, real conditional goals). It is small, so it is a slice
mixed into K12 (section 4), with atom permutations (24 per class, allowed on seeds) to raise its
weight without duplicating records. Seeds come only from the **seed split** (section 5).
*Built 2026-10-06* (`--prune_prem`, seed split v0 + Lean): 383 audited records, `data/pretrain_data/subproblems.jsonl.gz`;
with premises pruned to the ones each sub-proof cites the classes collapse (406 before the audit), so the slices are
filled with atom permutations. See `pools.md`.

**P2. Generator with the gaps closed (no textbook data at all).** The same generator with
`max_prem 5`, a quota of ∧-unpacking proofs (goal mode from conjunction premises: ∧E chains to an
atom or a rebuilt conjunction), a quota of nesting depth 3, a quota of conditional and negation
goals matching the textbook mix (31 % and 12 %), and lengths 2–14. Cheap, purely synthetic, and
the cleanest test of "data gap, not capacity gap". Needs new generator options written in
`code/pretrain_data/` (the pinned copy is not edited).

**P3. Minimal-proof K12.** The same 155,000 theorems as K12, re-proved by `minlen` (bound 14,
10–120 s each; Dan's ladder labels already cover 63,342 long-mode theorems at bound 8) so that
every training proof is minimal: no manufactured splits, no dead lines, no reiterations. Tests
whether the proof *style* the model imitates, not the theorem distribution, is what RL amplifies
into padding, and whether unpadded pretraining lets the 48-step budget reach longer proofs. If P3
helps, it is also the cheapest fix for the padding found last week.
*Built 2026-10-06* (`minimal_k12.py`, bound = one line fewer, 3 s per row, 85 s in all): 85,244 of 155,000 proofs got
shorter, 374,537 lines removed; **89 % of K12's long-mode rows were padded** (lazy-premise case splits). Mean length 7.0
→ 4.6, almost nothing above 9 lines. `data/pretrain_data/k12_minimal.jsonl.gz`; see `pools.md`.

**P6. Join problems (from `hard_problems.md`).** The hard textbook proofs are made of pieces K12 knows, joined in
ways K12 almost never shows (∧E feeding →E ≈ 90× rarer in K12 than in the hard proofs; ¬I feeding ¬E 128×; nested ∧E
35×; ↔-introduction 18×; nested ∨E 12×). `joins.py`: 71 templates in nine families filled with small formulas, proved by
`minlen`, kept only when the minimal proof contains the join, audited. *Built 2026-10-06*: 12,295 records,
`data/pretrain_data/joins.jsonl.gz`. The ¬E←¬I and ∨I←¬I joins are still missing (bypassed or over the bound);
reductio templates at bound 16 are the next builder.

**P4. Schemata pool.** The 26 textbook laws and their compositions (two laws chained through a
shared formula, e.g. contraposition then modus tollens) instantiated with random sub-formulas and
proved by `minlen`, mixed into K12 as a slice. Dan's instances exist; compositions are new.
Overlaps P1 in spirit but needs no textbook seeds, so it can be used without a seed/test split.

**P5. Longer and deeper only (K16 + depth 3).** K12 extended to length 16 and nesting depth 3 by
the existing long mode, filtered of manufactured splits with the padding classifier. Tests the
length knee directly. Likely overlaps P2 and P3; run only if those two leave the length ceiling
untouched.

Not proposed: hand- or LLM-written training proofs (every record must carry generator, search or
extractor provenance and pass the pinned verifier); renamings of the 72; any filter that reads a
test set.

## 4. Experiments

Everything held fixed: Robbie's best recipe as ported by Dan (6 × 384, 9.56M, Peri-LN, Muon),
`lean_staten` proof-state format, Stage-1 wall-clock 1,200 s on one A40, held-out as in
best-state, then Dan's T1 ladder exactly as in best-state (`state_ladder_ei.py`, 8 rounds, k 32,
T 0.8, `rl_targets.jsonl`, replay from the arm's own Stage-1 data, `max_steps` 48). The only
variable is the Stage-1 file. Baseline = Dan's best-cap12 cell (seeds 0 and 1 already scored:
v0 0.34 / 73–76, v1 0.36 / 49–50, batch 3 0.31 / 10–12, 72 aggregate 47–49).

| exp | Stage-1 set | question | seeds |
| --- | --- | --- | --- |
| E0 | K12 (existing checkpoints) | baseline, re-scored on the test split only; a 3rd seed if any arm reaches phase B | 0, 1 (exist), +2 |
| E1 | K12 with a 1 % and a 10 % slice replaced by P1 (sub-problems × permutations) | do easier textbook problems transfer to unseen textbooks? | phase A: 1 each; phase B: 3 |
| E2 | P2 (generator with gaps closed) | are the gaps data gaps? | phase A: 1; phase B: 3 |
| E3 | P3 (minimal-proof K12) | does proof style drive padding and reach? | phase A: 1; phase B: 3 |
| E4 | K12 with a 1 % slice replaced by P4 | do laws as data beat laws as targets? | phase A: 1; phase B: 3 |
| E5 | K12 with a 1 % and a 10 % slice replaced by P6 (join problems) | do the joins the hard proofs need transfer? | phase A: 1 each; phase B: 3 |
| E6 | P3 with a 10 % P6 slice (`k12_minimal_joins10`) | minimal style plus the joins, with some length kept | phase A: 1; phase B: 3 |

| E7 | size ladder: 15k, 50k, K12, 500k records of K12's content (`size_ladder.py`, ideas H8; the 1.5M pool was dropped 2026-10-07) | does unique data beat epochs? Robbie's 9.56M pretrain sees K12 20 times in 1,200 s; the 500k pool 6 times | phase A: 1 each; phase B: 3 for the best size |
| E8 | fraction dial: K12 with a nested 10⁻⁴ / 10⁻³ / 10⁻² / 10⁻¹ slice of ↔-elimination problems, a pattern K12 never shows (`k12_dial_*`, ideas I1–I4) | how much exposure does RL need to ignite a pattern: Dan's 10⁻⁴ or Zhang's 1 %? read at base pass@k and after the ladder, look for a jump (Gu) not a slope | phase A: 1 each; phase B: 3 at the two fractions around the jump |
| E9 | depth tail: K12 with a 10 % / 30 % slice of depth-3+ records (`k12_tail10/30`, ideas I19, I24) | does an easy-plus-deep mix reach the depth-3 textbook problems that K12 (depth ≤ 2) cannot? | phase A: 1 each; phase B: 3 |
| E10 | distractor premises: 10 % / 30 % of K12 rows with 1–2 unused premises, shortcut-checked (`k12_distract10/30`, `k12_minimal_distract10`, ideas I10, I17) | do unused premises teach premise selection (the 4-premise cell is 0 of 7 solved)? | phase A: 1 each; phase B: 3 |
| E11 | formula substitution: 30 % / 100 % of K12 rows with atoms replaced by depth-1 formulas at rate 0.7 (`k12_subst30/100`, idea I11) | does width generalisation (bigger formulas, same skeletons) transfer to the textbook shapes? note the 100 % pool is 1.5× the tokens | phase A: 1 each; phase B: 3 |
| E12 | long easy auxiliary stream: 10 % slice of 13–30-line ∧I/∧E/→E chains with 3–6 premises (`k12_aux10`, idea I18) | does length bookkeeping learned on easy long proofs transfer to 13+-line textbook proofs? | phase A: 1; phase B: 3 |
| E13 | counterfactual premise pairs: 1 % slice of siblings with one critical premise swapped so a different, longer proof is needed (`k12_pairs1`, idea I16) | does the pair signal (same goal, different proof) reduce shortcut proofs? smallest pool, lowest prior | phase A: 1; phase B: only if A shows a sign |
| E14 | spurious-reward control: K12 with 10 % / 100 % of its proofs replaced by a one-error version the pinned verifier rejects (`k12_wrong10/100`; Charles 2026-10-07, cf. Shao et al. 2025) | does Stage-1 teach correctness or only the proof format? if the all-wrong pretrain plus the ladder reaches K12's cells, the pretrain's role is surface structure and RL supplies the correctness; a Stage-1 checkpoint from this pool must never be mistaken for a prover | phase A: 1 each (the 100 % arm first); phase B: 3 if the 100 % arm is within 10 points of K12 |
| E16 | closure: K12 with a 10 % slice of sub-theorems cut from every interior line of the minimal K12 proofs and the depth-3 side pool (`k12_closure10`, ideas A2 / H1) | does AlphaGeometry-style traceback (every derived line is a theorem, enclosing assumptions become premises) fill the mid-length, more-premise cells without any textbook seed? | phase A: 1; phase B: 3 |
| E17 | composition: K12 with a 10 % slice of two minimal proofs spliced lemma-then-use, kept only when the splice is itself minimal (`k12_compose10`, ideas A4 / H15) | do →E chains and hypothetical-syllogism shapes built from generator pieces transfer? | phase A: 1; phase B: 3 |
| E18 | recovery: K12 with 20 % of rows replaced by their minimal proof plus one dead-but-valid line after the premises (`k12_recover20`, ideas B3 / H3) | does "a wasted step, then a clean finish" in the data reduce RL's padding more than all-minimal data (E3) does? read with the padding metric | phase A: 1; phase B: 3 |
| E19 | second proofs: K12 with a 10 % slice of alternative proofs of K12 theorems found by `minlen` under a restriction the original violates, originals kept (`k12_multi10`, ideas B2 / H11) | does proof diversity per theorem in Stage-1 widen the ladder's pass@k and the number of distinct proofs on unsolved problems (I8)? | phase A: 1; phase B: 3 |
| E20 | duals: K12 with a 1 % slice of Γ, ¬A ⊢ ⊥ and Γ ⊢ ¬¬A rewrites of K12 theorems (`k12_dual1`, ideas A5) | do ⊥ goals and ¬¬ goals as a pattern-dial slice move the refutation / DN cells? | phase A: 1; phase B: only if the cell moves |
| E21 | shape-matched pool: the generator population (`k12_500k` ∪ depth-3 side pool) importance-resampled to the seed split's (goal connective, premises, length, depth) histogram (`k12_shape`, ideas C3 / H10) | is matching the textbook *shape* distribution, with no textbook problem, enough? the cells the generator cannot produce (4+ premises, 13+ lines) are reported as unreachable target mass | phase A: 1; phase B: 3 |
| E22 | rule-balanced pool: the same population resampled by 1 / share of each record's rarest rule (`k12_balance`, ideas C5 / H13) | does removing K12's rule skew (44 % ∨E) remove shortcut learning and move the ∧-shuffle and →E-chain cells? | phase A: 1; phase B: 3 |
| E23 | atom-count pools: every training record uses exactly 1, 2, 3 or 4 sentence letters (`atoms1..4`, `atom_pools.py`; Charles 2026-10-08) | does the number of distinct atoms in Stage-1 data set what RL reaches on the textbook sets, where 19–25 % of problems use all four and the tokenizer has no fifth? atoms4 against K12 (mixed) separates letter count from content; atoms1 is 34k records (the generator has no more distinct 1-atom theorems) and 1-atom theorems are degenerate (P ∨ P, Q ∧ Q), so it is a floor, not a candidate | phase A: 1 each (atoms2–4; atoms1 only if budget is spare); phase B: 3 for any arm that moves |

Pools for E1, E3, E5, E6, E7 and the 1 % spice-rack mix are built and compared at the data level in `pools.md`
(2026-10-06); the files are under `data/pretrain_data/pools/` (gitignored, regenerated by `mix.py --seed 0`).
Pools for E8–E13 are built by `scan_pools.py` (2026-10-06, `pools.md` "Scan pools" section; files under
`data/pretrain_data/pools/scan/`). Pools for E16–E22 are built by `new_pools.py` (2026-10-07, `pools.md` "New pools";
files under `data/pretrain_data/pools/new/`). Every pool keeps K12's first 1,024 rows at its head, which the pinned trainer
holds out as its val slice, so val_loss is comparable across pools. The scan's "edge slice last" ordering (I1)
cannot be built as data: the pinned trainer reshuffles every epoch.

**Phase A** (one seed per arm, frozen + T1) is a screen, not a result: it picks the two arms whose
own type cell moved most. **Phase B** runs those two and the baseline to 3 seeds. Only phase B
numbers are compared, per Dan's noise floor. An arm whose phase-A seed is worse than the baseline's
worse seed on the test split is dropped unless its type cell moved.

Read-outs per arm, frozen (after Stage-1) and after T1, all with `state_eval.py` at k 32, T 1.0
and the pinned verifier re-judging, exactly last week's protocol:
1. **Test split** of the textbook sets (section 5): avg solve rate, any-of-32, and the
   new-capability count (problems the baseline never solved that the arm solves in ≥ 16 of 32).
2. The **72** in aggregate (counts only).
3. **By type:** 4-premise problems, box-free ∧-shuffles, depth-3, 14+ lines, hypothetical-syllogism
   shape, ∨E, DN. These are the cells the pools were built to move; a win that does not show up in
   its own cell is suspect.
4. **No regression:** fork held-out greedy (≥ baseline − 2 pp) and Dan's transfer dev metric.
5. **Padding:** mean written − reference and manufactured-split share, with
   `code/textbook_problem/padding_analysis.py`.

**Pre-registered expectations (numbers to be fixed in each proposal.md before launch).** E1:
+5 to +15 problems on the test split after T1 and at least 3 of the 7 four-premise problems solved;
a gain confined to problems from the seed books would mean memorised shapes, not transfer. E2: the
4-premise and ∧-shuffle cells move, length does not. E3: padding falls below 2 lines; reach at
14–20 lines rises from 12 of 36 to ≥ 18 of 36 if the step budget was binding, unchanged if not.
E4: schemata families (De Morgan, distribution, contraposition) move, others do not.

## 5. Seed / test split and leak rules

- **Seeds (may be mined for sub-problems, permuted, filtered):** candidate_v0 (forall x Syracuse,
  Calgary, MSU; the Carnap Book: 87 released rows) and the Lean-sourced 47. Different books from
  the test split, so a gain is cross-book transfer.
- **Test (never read by any pool builder):** candidate_v1 (Roy, Carr: 72), batch 3 (Pelletier,
  R³, Oxford: 14), the hand-proved 14, the derived rows whose base is a test problem, and the
  sealed 72 (aggregate only). **(decide)** whether Roy's set is too different from forall x for E1
  to be fair; default: keep it, and also report the seed-split sets as an in-distribution check.
- Every pool is audited by curry-aware renaming class (`release_1002.BR.keys`) against every test
  set, the 72, validation-36, the fork's held-out, transfer and RL targets **before** training;
  sub-problems that are twins of a test problem or a sealed problem are removed and counted. The
  audit file lives in the experiment folder with the pool's hash.
- The RL targets stay Dan's 3,000 generator theorems: the ladder never sees a textbook problem.

## 6. Gates

- **G0 (before any GPU):** pool built, verified line by line, audited (section 5), statistics
  table (length, premises, rule use, goal connective, depth, manufactured-split share) committed
  next to the proposal, and the pool's sha256 in git with the file on the HF bucket.
- **G1 (after Stage-1, 1 seed):** held-out greedy within 2 pp of K12's; otherwise the pool is
  retuned once, then dropped.
- **G2 (after T1, 3 seeds):** a test-split gain in mean any-of-32 of at least 8 problems (of
  ~100) and at least twice the baseline's seed spread, or a type-cell gain the pool was built
  for that every seed shows. Otherwise record the null and move on. Two-seed numbers are never
  written up as a comparison.
- Any result that beats the baseline by more than the pre-registered band triggers a leak
  re-audit before it is written up (as last week).

## 7. Tasks for the week

1. **Fri Oct 3:** this plan; survey (section 2.1); prototype P1 (done: 1,359 classes); open
   questions to Charles. Commit.
2. **Sat–Sun:** pool builders in `code/pretrain_data/`: `subproblems.py` (P1, with the seed split,
   permutations, audit and stats), `gen_gaps.py` (P2), `minimal_k12.py` (P3, CPU, 8 procs),
   `schemata_pool.py` (P4). G0 for each.
3. **Mon:** Stage-1 pilots, 1 seed each (E1–E4), on one pod; G1.
4. **Tue–Wed:** T1 ladders for the arms that pass G1, 2 seeds; frozen and T1 read-outs.
5. **Thu:** E5 confirmation seed; write-up (results.md per experiment, doc for the group meeting).

## 8. Budget **(decide)**

Per arm and seed: Stage-1 1,200 s plus data load (≈ 0.5 A40-hour); T1 ladder 8 rounds (Dan's
records: a `lean_seq` ladder round took 657 s solo on a 3090 for the 3.3M model, 27–36 min with
four arms sharing a GPU; the 9.56M model in the state environment with Lean in the loop is
slower, so assume ≈ 3 A40-hours per ladder until the phase-A pilot measures it); scoring on the
textbook sets (CPU, ≈ 1 hour per seed as last week). Phase A: 5 Stage-1 runs + 5 ladders; phase B:
6 more of each (2 arms × 2 extra seeds, baseline + 1). Total ≈ 11 ladders ≈ 35–40 A40-hours, about
$20 at Dan's billed A40 rate ($0.49/h, `fork_dan/numbers.md`), plus pod overhead. Spend is logged
in `docs/pretrain-data/spend.md`. Default: no pod is started until Charles confirms the budget
and that we run Dan's ladder ourselves rather than asking Dan to.

## 9. Metrics and reporting

As in `CLAUDE.md`: every number with its source file, n and seeds; intervals over problems; one
seed is signs of life, two is a pattern, three for any comparison claim; new-capability counts,
not only averages; every selection effect named (the seed/test split above all). Results go to
`experiments/pretrain-data/<exp>/results.md` and to a doc for the Saturday meeting.
