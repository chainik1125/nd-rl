---
written_on: 2026-11-10
written_by: agent:claude
---

# How many checker-neutral habits can an author have in K12?

Leon asked: "are there really that many different ways of doing a proof?" This note measures it on the
K12 pretraining corpus, for a planned corpus written by two synthetic authors with different habits.
The work is CPU only: no model was trained or sampled.

Labels used below: **[obs]** observed in this run's outputs, **[interp]** my reading of them, **[guess]**
not tested.

## Summary

**Answer to Leon's question.** [obs] There are few genuinely different routes. In a random sample of 5,000
theorems, 8.6 % have a second proof within 2 lines of the shortest that opens a different set of boxes.
Half of those (222 of 429) have inconsistent premises, where ex falso competes with an introduction rule.
44.7 % of corpus proofs are already as short as the shortest proof the search finds.

[obs] What there is a lot of is freedom in how one route is written, and Lean accepts every form. Steps can
sit inside or outside a box and in different orders (39.8 % of theorems, 0 lines). Outer lines can be
restated inside boxes (66.5 %, about +2 lines). Some decorations apply to every proof but cost lines
(idle reductio: 100 %, +4 lines). [obs] Short proofs have almost no room. At 2 to 4 lines (27 % of the
corpus) a zero-cost habit applies to under 1 % of proofs. [interp] So an author's style would be visible
only on the longer proofs.

**Which habits to give two authors** [interp]:

1. **Step order** (premise-driven against goal-driven). This is the best habit to reward. It has room in
   39.8 % of theorems (75 to 93 % at 10 to 12 lines), costs 0 lines, and differs in the state at 65 % of
   steps. The corpus is already split about evenly (40 % / 44 % / 16 % mixed), so each author's corpus
   only needs normalising.
2. **Reiteration** (restate outer lines inside the box against cite them directly). This is the habit to
   watch for drag. It is the textbook Fitch habit and has room in 66.5 %. It shows inside the box. It costs
   about +1.9 lines for the reiterating author. It is coupled to habit 1: hoisting lines out of boxes
   creates more outer citations, so measure it as the reiterated share of sites, not as a count of R lines.
3. Optionally, **conjunct order** (left conjunct first against right first). It costs 0 lines, is visible
   and is independent of the other two, but has room in only 9.1 % of theorems. The alternative is **ex
   falso style** (BOTE against a vacuous reductio): 13.0 % room, already split 63/37 in the corpus, ±3 lines.

I would leave the idle reductio out of the first pair. It costs 4 lines. Its "inside" form raises box depth
past K12's depth-2 cap on the 32 % of proofs that are already at depth 2. It would also add a template
reductio to every proof, which is the quantity the reductio knockout manipulates.

## The table

[obs] Checks: all 155,000 K12 proofs, `train_k12.jsonl.gz`; Dan's code `dan_code_13e41bbd`; Lean 4.34.0
core. Room is a % of the 155,000 theorems. Each rewritten variant was checked on a seeded sample of
proofs with room:

- Lean, through `nd2lean.translate` and `nd2lean.lean_check` (Dan's check);
- `nd_verify`;
- a replay through `state_env.decompose(canon=True)`, the step format the models see.

The **visible** column compares the two forms' state sequences, with hypothesis names stripped. It counts
a pair if some state after the first differing step never occurs in the other form. It also gives the
share of the rewrite's steps whose state the other form never shows.

| # | habit | room, % of 155,000 | current split at the room points | visible in later states | line cost | Lean-checked |
|---|---|---:|---|---|---|---|
| 1 | step order | 39.8 % | corpus proof = premise-driven form 40.3 %, = goal-driven 43.8 %, neither 15.9 % (of room proofs) | yes: 300/300 pairs differ; 65 % of steps | 0 | 300 + 300, all accepted |
| 2 | conjunct order | 9.1 % | left first 94.5 % of 14,567 sites | yes, while both conjuncts are in scope: 300/300; 39 % of steps | 0 | 300, all |
| 3 | reiteration | 66.5 % | restated at 10.0 % of 215,649 sites | yes, inside that box: 300/300; 39 % of steps | +1.9 to reiterate; −1 per optional R removed | 300 + 300, all |
| 4 | idle reductio | 100 % | 16,717 proofs hold a reductio, 59 working (Dan's numbers reproduced) | inside: yes, 76 % of steps; outside: only the last 4 steps | +4 | 300 + 300, all |
| 5a | disjunct, both already citable | 2.3 % | ORI1 at 50.6 % of 4,186 sites | no when only the term changes: 149/300 differ, all through pruned lines | 0 or less (−1.0 mean, dead lines pruned) | 300, all |
| 5b | disjunct, other side derivable | 7.4 % at ≤ 0 extra lines, 16.1 % at ≤ +2 (n = 5,000) | not counted | yes: 300/300 | +0.9 mean for the rewrite | 300, all |
| 6 | ∧ detour (X, X & X, X) | 94.0 % | 0 existing | yes: 300/300; 69 % of steps | +2 | 300, all |
| 7 | different route (box multiset) | ≥ 8.6 % of n = 5,000 (≥ 4.1 % with consistent premises) | not applicable | yes (different hypotheses) | ≤ +2 by definition | 112 search alternatives, all |
| 8 | ex falso style (added) | 13.0 % | BOTE at 63.1 % of 22,523 sites | yes: 195/195 and 109/109; 30 to 36 % of steps | +3.4 to the reductio form, −2.2 back | 195 + 109, all |
| 9 | redundant case split (added) | 37.6 % | 81.7 % of 75,317 ORE lines are redundant | partly: 300/300 differ; 17 % of steps | −4.5 to remove | 300, all |

The full machine-written table, with definitions and per-variant checks, is
`/scratch/cowifornia/nd-rl/author_tags/habits/table.md`.

[obs] Room by proof length (2 to 12 lines, % of that length's 14,091 proofs):

| habit | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| order | 0 | 0 | 0 | 8 | 22 | 41 | 50 | 62 | 75 | 87 | 93 |
| conjunct | 0 | 0 | 0 | 3 | 10 | 6 | 8 | 14 | 16 | 23 | 19 |
| reit | 0 | 0 | 30 | 69 | 74 | 86 | 91 | 95 | 96 | 95 | 95 |
| disjunct 5a | 0 | 0 | 1 | 1 | 1 | 2 | 2 | 4 | 5 | 5 | 5 |
| ex falso | 0 | 0 | 2 | 3 | 19 | 9 | 10 | 18 | 20 | 29 | 32 |
| redundant split | 0 | 0 | 0 | 0 | 2 | 27 | 57 | 71 | 82 | 84 | 91 |

[obs] Other counts:

- 30.1 % of proofs have room for none of order, conjunct, reiteration or disjunct (5a).
- 38.2 % have room for both order and reiteration.
- 23.5 % of theorems have inconsistent premises: no assignment of P, Q, R, S satisfies them, so ex falso
  proves any conclusion.
- Every theorem in K12 is distinct (155,000 `key`s and 155,000 `thm`s), so the corpus offers no
  same-theorem, different-proof pairs.

## How the proofs are handled

[obs] `habits.py` parses each proof into a tree shaped like Lean's: every box is attached to the line that
discharges it (IMPI or NEGI: one box; ORE: two), which is where `nd2lean` puts it.

- A **unit** is one non-AS line together with the boxes it discharges.
- A **scope** is the top level or one box body.

Every rewrite renders back to spec.md text. All 155,000 corpus proofs re-render byte for byte. Dependencies
come from citations: a unit depends on any unit of its scope that it, or anything inside its boxes, cites.

## The habits, one by one

**1. Step order.**
- Definition. The *premise-driven* form moves out of each box every unit that does not depend, directly or
  through other lines, on the box's assumption. It never moves the box's last line, and it repeats until
  nothing moves. Each scope is then ordered with flat units (no box) as early as the dependencies allow.
  The *goal-driven* form moves every unit whose citers all lie inside one box of one later unit into that
  box. Each scope is then ordered with boxes first and flat units as late as possible.
- Room: the two forms differ.
- [obs] Freedom within scopes alone (box placement fixed) gives more than one order in 11.8 % of proofs. The
  median count is 1: 136,705 proofs have 1 order, 12,353 have 2, 4,546 have 3 to 5, 1,361 have 6 to 23,
  and 35 have 24 to 119. A line can sit inside or outside a box in 37.2 %.
- [obs] Most of the room is therefore placement, not permutation. Among incomparable pairs of one flat and
  one box unit in a scope, the flat unit comes first in 60.2 % of 6,932.
- [guess] The generator's two modes probably produce the two styles. Forward mode applies rules to the lines
  it already has and opens boxes at random; goal mode (`reach`) opens the goal's box first and derives inside
  it. That would explain why the corpus is already near 50/50. The records do not say which mode made them,
  so I did not check this.

**2. Conjunct order.**
- Site: an ANDI whose two cited lines are different units of its own scope and do not depend on each other.
  An ANDE1/ANDE2 pair on one conjunction in one scope also counts (only 52 of the 14,567 sites are such
  pairs).
- [obs] Left first at 94.5 % of sites. This matches the generator: `reach` builds the left conjunct first.
- Rewrite: swap one site. The scope is reordered with the right conjunct's unit placed before the left's
  and otherwise the original order kept.

**3. Reiteration.**
- [obs] How R is used now. Of 40,167 R lines, 14,568 (36 %) are *required*. Each is the last line of a box
  and cites a line that is neither that box's assumption nor the line just before it. The box must end in
  that formula, and in the step format `exact` closes a frame only on its last statement. Lean's kernel
  would accept citing the outer line directly, but the ND format and `state_env` would not. The other
  25,599 (64 %) are *optional*, and 3,630 of these are top-level restatements.
- Site: (box, outer line) pairs where the box cites the outer line at its own level, either directly or
  through an optional R. [obs] 10.0 % of 215,649 sites are restated.
- Variants: `reit` restates each such outer line once, right after the box's assumption, and redirects that
  level's citations to the copy (+1.9 lines on average). `rfree` removes every optional R.

**4. Idle reductio.**
- Variants: `outside` adds `~G : AS ; F : NEGE ; ~~G : NEGI ; G : DN` after the conclusion. Dan's
  classifier calls this kind "pointless". `inside` puts the whole derivation inside the ~G box. Both apply
  to every proof and both cost +4 lines.
- [obs] Dan's numbers reproduce exactly with his classifier (`proof_strategies/strategies.py` from the fork
  at `e6bfe0e`): 16,717 proofs (10.8 %) contain a proof by contradiction (#10), 59 of them working and
  16,658 template-only. `patterns.reductio` gives 16,703. Proofs holding at least one instance of each
  kind: vacuous 8,209, scaffolded 7,693, pointless 1,082, negative goal 48, free 6.
- [obs] The `outside` form is visible only in the last four steps. The `inside` form changes the goal to
  False and adds ~G to every later state.
- [obs] The ¬¬ detour of candidate 6 is the same construction as `outside`, so it is not counted separately.

**5. Disjunct choice.**
- (a) Site: an ORI on A v B (A ≠ B) where the other disjunct is already citable. [obs] 2.3 % of proofs;
  ORI1 at 50.6 % of sites. [interp] From reading `gen.py`: `reach` picks ORI1 whenever both sides are
  available, and forward mode, which adds a random second disjunct, supplies the rest at 50/50.
- [obs] Swapping changes only a term, which the state does not show. 151 of 300 swaps leave every state
  identical; the other 149 differ only because the line the old side used becomes dead and is pruned.
- (b) On a 5,000 sample, minlen's dynamic programme searched for the other disjunct from the formulas
  citable at the ORI line, within (the lines only the current side uses) + 2.
- [obs] 43.1 % of sampled proofs have an ORI. The other side is derivable at a cost of no more than the
  current side in 7.4 % of proofs, and within +2 lines in 16.1 %. 300 spliced variants (derivation
  inserted, side switched, dead lines pruned) pass Lean, nd_verify and the state replay.

**6. ∧ detour.**
- Rewrite: insert `X & X : ANDI p p ; X : ANDE1` after the premises and redirect citations of premise p.
- [obs] The rewrite works in the 94.0 % of proofs that cite a premise, costs +2 lines, and does not occur in
  the corpus. [interp] It is valid but artificial: no human author writes it.

**8. Ex falso style (added).**
- Site: a BOTE line, or a vacuous reductio: ~G assumed, F derived without it, then ~~G and G. The NEGI line
  must be cited only by the DN.
- [obs] 13.0 % of proofs have a site; 63.1 % of sites use BOTE.
- [interp] Both forms are already present in the corpus at a balanced rate, and both are textbook forms
  (explosion, or indirect proof).

**9. Redundant case split (added).**
- Site: an ORE with a branch whose last line does not depend on its case. The rewrite replaces the split by
  that branch's lines and prunes.
- [obs] 37.64 % of proofs, the same as Dan's proof-strategies figure. 81.7 % of all ORE lines are redundant.
  Removing them saves 4.5 lines on average.
- [interp] This is a "tidy author against split-happy author" habit, but it changes length a lot.

**7. Different routes.**
- Method: Dan's `minlen.py` dynamic programme, run on a random sample of 5,000 theorems (bound 14 lines,
  20 s each, no timeouts). First find the shortest proof, with box multiset {IMPI, NEGI, ORE, RAA}, where
  RAA is a NEGI on ~G cited by a DN that yields G. Then search again with each box rule of that proof
  banned in turn. Banning RAA also bans DN, or a ~~G lemma would rebuild it.
- [obs] The shortest proof opens a box in 45.6 % of the sample. A proof within +2 lines with a different
  multiset was found by the bans for 111 theorems (20 at the same length). In 340 more, the corpus proof
  itself is such a witness. Together that is 429 (8.6 %).
- [obs] The most common pairs: an IMPI against an inner ex falso, or against a second IMPI; a proof by
  contradiction against a case split; a box-free ex falso against an introduction rule.
- [obs] 222 of the 429 have inconsistent premises (the sample's rate is 22.7 %).
- [obs] The corpus proof exceeds the shortest found by 2.4 lines on average and is the shortest in 44.7 %.
- [interp] The figure is a lower bound. minlen's dynamic programme keeps one best proof per subgoal, so it
  cannot enumerate. Alternatives that *add* a box are only seen when the corpus proof happens to be one.
- [obs] Ties are real. minlen iterates over Python sets, so which of several equally short proofs it returns
  depends on the process's hash seed. In an unpinned rerun the bans found 110 alternatives instead of 111 and
  the corpus witnesses rose from 340 to 341, with the total unchanged. `habits.py` now pins
  `PYTHONHASHSEED=0`; two pinned runs give identical numbers.

## (d) What the state carries

[obs] The state at each step is the list of hypothesis formulas in scope, in the order they were derived,
plus the goal (`state_env.state_tokens`). It does not contain terms or history. So:

- order, placement, reiteration, detours and the inside reductio leave traces that persist;
- conjunct order shows only while both conjuncts are in scope;
- the outside reductio shows only at the end;
- a zero-cost ORI swap, or any choice between equal formulas, leaves no trace.

[interp] An author's habit can carry from step to step through the state, without a tag, for habits 1, 3, 4
(inside), 6 and 8. The trace is weaker for 2 and 9 and absent for 5a.

## What I could not measure

- I could not enumerate all proofs within +2 lines. The route figure is a lower bound (see 7), and it is
  limited to minlen's formula space: subformulas of the theorem, their negations and double negations.
- I counted orders within scopes and, separately, whether any placement freedom exists. I did not count
  the full number of distinct proofs reachable by combining both.
- The visibility figure is a proxy: a state the other form never shows. It does not test whether a model
  can infer the author from one state. That needs a probe or a classifier on a trained model.
- The rewrites were not filtered to K12's caps. The inside reductio and the reiteration and detour variants
  can push proofs past 12 lines or to depth 3.
- Lean checked a seeded sample of each rewrite, of 109 to 300 proofs (rows above), not every proof in the
  corpus.

## Reproduce

```
cd /scratch/cowifornia/nd-rl/worktrees/author-tags/code/experiments/current/author_tags_20261110
LEAN_GATE_WORKERS=8 /scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python habits.py all
/scratch/cowifornia/nd-rl/think/nd-rl/code/.venv/bin/python -m pytest -q test_habits.py
```

- The full run takes about 4 minutes on 8 cores of serrano: scan 6 s, Lean 13 s, search 74 s, disj 31 s.
  The script re-executes itself with `PYTHONHASHSEED=0`, so repeated runs give the same table.
- Outputs, in `/scratch/cowifornia/nd-rl/author_tags/habits/`:
  - `scan.json` and `scan.jsonl.gz`: per proof;
  - `lean.json` and `lean.jsonl.gz`: every rewritten proof with its Lean, nd_verify and state verdicts;
  - `search.*`, `disj.*` and `dan.json`;
  - `table.md` and `run_all.log`.
- The code is on branch `leon/author-tags`. It uses Dan's code at `knockout_danbase/dan_code_13e41bbd` and
  his proof-strategies fork at `/scratch/cowifornia/nd-rl/ext/nd-takehome-ps` (`e6bfe0e`).
- The 15 tests in `test_habits.py` cover the byte-identical round trip, each rewrite's validity on
  hand-made proofs, the order normal forms, the R classes, visibility, and a Lean negative control.

## Next steps

Decide the author pair (my proposal is above). Then build the two-author corpus by normalising each
theorem's proof to its author's form. The rewrites in `habits.py` are the building blocks. Keep the
reiteration measure per site so that habit 1 does not move it mechanically.
