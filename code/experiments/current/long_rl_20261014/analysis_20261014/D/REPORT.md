# Agent D: minimal pairs and constructed probes — what the long-RL models cannot prove

Models: the four v2 EI runs (untied s0/s1, looped s0/s1), `found_targets.jsonl` + `eval_dev.jsonl` for the set
analysis; `untied-ei_s1/final.pt` and `looped-ei_s0/final.pt` (8 loops) for sampling probes (k=8, T=0.8, Lean-judged).
Work dir: `scratchpad/probe/D/` (`analyze.py`, `step*.py`, `probe_swap.py`, `probe_synth.py`, `*.out`, `*.json`).

## TL;DR

The dominant failure on short targets is **syntactic, not logical, and one token wide**: the model cannot write a
nested box (Or.elim / ¬-intro / →-intro) inside the **first** branch of a case split. After the inner box closes it has
seen "two fun-blocks" and emits `; exact …` (end of Or.elim) instead of `( fun …` (second branch); the correct `(` has
log-prob ≈ −12 (p ≈ 6e-6) in 143/146 gold proofs of that shape. Pretraining contains **0** proofs with a box in the first
branch of an ORE (61 in the second branch, out of 2,277 ORE lines in 155k proofs). Swapping the two disjuncts of the
case-split premise — the only change — flips solve rate from **0/80 → 77/80** samples (and 77/80 → 2/80 the other way).

Two further, logical, classes: **(B) a negation/implication lemma that is not the current goal** (NEGI/IMPI box
whose conclusion is intermediate; 0/80 on a synthetic family vs 68/80 for the same theorem with the box proving the
goal directly) and **(C) classical reductio** (assume ¬goal … DN; excluded middle 0/80, Peirce 0/80; old-dev
`excluded_middle` 0/19, `peirce_sequent` 15/20 never solved, RL pool: 262/271 DN targets never solved).

## Taxonomy (RL pool, 13,946 targets with a gold proof; "never" = solved by none of the 4 arms)

Classes assigned in order; a target gets the first that matches its gold proof.

| class | definition (on the gold proof) | n | never | solved by all 4 | never, by family |
| --- | --- | ---: | ---: | ---: | --- |
| A box-in-first-branch | some ORE whose **first** branch contains ORE/NEGI/IMPI | 1,043 | 931 (89%) | 1 | d 481, s2 232, s 166, c 51 |
| B intermediate ¬-lemma | NEGI inside a box whose formula ≠ theorem goal (and not A) | 535 | 526 (98%) | 1 | c 499, d 27 |
| C classical | gold uses DN (and not A/B) | 111 | 102 (92%) | 0 | d 102 (all DN targets: 262/271 never, 0 by all) |
| D other, body ≤ 19 | none of the above | 4,344 | 1,132 (26%) | 1,685 | c 893, s2 100, d 81, s 58 |
| E long, body ≥ 20 | length ≥ 20 (b/b2/b3/c/e families) | 7,913 | 7,647 (97%) | 0 | e 2,889, b* 4,049, c 666 |

Restricted to body 7–14, where length alone is not the story: A 498/607 never (82%), B 28/36 (78%), C 79/88 (90%),
**D 90/2,606 (3%)** with 57% solved by all four. So at ≤ 14 lines the three structural classes explain 605 of the
695 never-solved gold targets (87%); the remaining 90 are mostly d-family 12–14-liners and c-family 14-liners.
Same picture on `dev.jsonl` (k=64): A 57/61 never, B 30/34, D 58/275, E 429/471.

Union/intersection: 10,550 targets solved by no arm, 5,622 by all four, 686 by exactly one. Seed-specific ("barely
reachable") targets are concentrated in band 15–24 (390 of 686) and are overwhelmingly class D/E: the *structural*
classes are nearly all-or-nothing across arms (A: 1 of 1,043 solved by all; B: 1 of 535), i.e. they are not a sampling
budget problem. Old dev (1,108, k=16): the 61 never-solved are textbook schemas needing reductio: `excluded_middle`
19/19 never, `peirce_sequent` 15/20, `negated_conditional` 11/18, `demorgan_or_to_nand` 10/18, `demorgan_nand_to_or`
8/19; constructive schemas (`constructive_dilemma`, `contraposition`, `export`, `import`, `disjunctive_syllogism`) 0 never.
The ladder targets in the RL pool have no gold proof and were not classified (212 of 4,493 never solved).

Features that do **not** matter at matched length and family (step3.out): distractors (gold cites every premise, and
adding 3–4 distractor premises to solvable synthetic theorems changed nothing: 63 vs 68, 6 vs 5 of 80), lemma reuse,
premise count, whether the goal is negated, whether the first branch is longer (`b1 longer` 0.44 vs `b1 shorter`
0.41 solved-by-all at body 13 among class-D s-family targets).

## Class A: nested box in the first case-split branch (the mechanism)

### Minimal pairs from the pool (same family, length, premises; key edit distance 2)

Solved by all four vs never solved, s family, body 13, 6 premises. Only the order of the disjuncts differs:

```
SOLVED rl_10517: (a0>(a0>a1)), (a2>(a2>a1)), (a3>(a0 v a4)), (a4>a5), (a5>a1), (a2 v a3) |- a1
UNSOLV rl_11321: (a0>(a0>a1)), (a2>(a2>a1)), (a3>(a0 v a4)), (a4>a5), (a5>a1), (a3 v a2) |- a1
```
In rl_10517 the a3 branch (which needs the inner case split on a0 v a4) is the **second** Or.elim branch; in rl_11321
it is the first. Same pair shape: rl_9706/rl_11595 (body 11), rl_5056/rl_10478, rl_4165/rl_11025, rl_10455/rl_9714,
rl_5462/rl_10439, rl_5323/rl_11407 (step4.out lists 14). d family, body 9: rl_13846 (all) vs rl_5198 (none), where
the ¬-intro box sits in the second vs first branch.

Pool statistics (step5.out), s+s2 family, solved-by-all / never-solved by where the nested box is:

| body | nested box in 1st branch | nested box in 2nd branch | no nested box |
| --- | --- | --- | --- |
| 12 | n=24: 0.00 / **0.92** | n=17: 0.41 / 0.00 | n=218: 0.43 / 0.00 |
| 13 | n=55: 0.00 / **0.93** | n=55: 0.38 / 0.00 | n=217: 0.43 / 0.01 |
| 14 | n=61: 0.00 / **0.95** | n=68: 0.34 / 0.00 | n=161: 0.41 / 0.05 |
| 16 | n=45: 0.00 / 0.98 | n=27: 0.00 / 0.07 | n=74: 0.09 / 0.19 |

d family body 10: first 0.01 / 0.57, second 0.01 / 0.23 (the d "second" residue is class B/C, below).

### Where the gold proof's probability collapses (untied s1, teacher-forced; step6/7.out)

s/s2 "first" gold proofs, body 11–14 (n=146): mean total log-prob −39.9 vs −20.7 for "second"; the single worst token
is the `(` opening the second branch in **143/146**, mean log-prob −12.9, model argmax `;`. Removing that one token
leaves −27; the next-worst tokens are name tokens (noise). d family: 99/113 "first" proofs have the same worst token.

What the model actually writes (rl_7147, k=8, all `LEANPARSE expected (`):
```
… Or.elim n92 ( fun ( n94 : U ) => by … have n97 : R := Or.elim n96 ( fun … ) ( fun … exact n104 ) ; exact n97 ) ; exact n93 <eos>
```
It finishes the inner Or.elim, closes the first outer branch, and then **ends the outer Or.elim** — it has counted two
`( fun … )` blocks since the last `Or.elim` and treats the case split as complete. A flat "blocks closed" counter is
correct for every proof in pretraining (0 boxes in a first branch) and for the second-branch case, so EI never had a
positive sample to teach the stack.

### Constructed probe 1: swap the disjuncts (probe_swap.py, probe_swap_results.json)

For 20 s/s2 targets (body 11–13) and 12 d targets (body 9–10) I reversed `( A v B )` in the case-split premise
(in the prompt and in the gold proof, which is re-ordered and renumbered; all 32 swapped golds Lean-accepted) and sampled
k=8 on the original and the swapped theorem.

| arm | direction | fam | original: samples ok / targets solved | swapped |
| --- | --- | --- | --- | --- |
| untied s1 | first→second (never solved in RL) | s/s2 n=10 | 0/80, 0/10 | **77/80, 10/10** |
| untied s1 | second→first (solved by all) | s/s2 n=10 | 77/80, 10/10 | **2/80, 1/10** |
| untied s1 | first→second | d n=6 | 0/48, 0/6 | 13/48, 3/6 (the 3 unsolved are also class B) |
| untied s1 | second→first | d n=6 | 34/48, 6/6 | 7/48, 1/6 |
| looped s0 (8 loops) | first→second | s/s2 n=10 | 0/80, 0/10 | **73/80, 10/10** |
| looped s0 | second→first | s/s2 n=10 | 73/80, 10/10 | **0/80, 0/10** |
| looped s0 | first→second | d n=6 | 0/48, 0/6 | 0/48, 0/6 |
| looped s0 | second→first | d n=6 | 32/48, 6/6 | 33/48, 5/6 |

Observed: in the s family the disjunct order alone decides solvability for both architectures. Interpretation: the
looped model can place a ¬-intro box first (d, 33/48) but not an Or.elim; the untied model can do neither. The d
first→second residue (0/48 looped, 13/48 untied) is confounded with class B (4 of the 6 need an intermediate ¬-lemma).

Hypothesis A: fails because the model tracks box nesting with a flat count of closed `( fun … )` blocks learned from a
pretraining set that has no box in a first branch; p(correct token) ≈ 6e-6 so EI with 8–32 samples never finds it.
Would be fixed by (i) adding pretraining proofs with the nested box in the first branch (tens of examples should do;
the second-branch version was learned from 61), or (ii) letting the sampler reorder branches (an ND proof may write the
boxes in either order; the Lean renderer forces disjunct order — a `Or.symm`/`Or.comm` rewrite before `Or.elim` would
remove the asymmetry), or (iii) at test time, re-prompting with the disjuncts swapped (a free 0 → 77/80 on this class).

## Class B: a ¬-intro / →-intro box whose conclusion is not the goal

Pool: 535 targets (c 499, d 27, mostly body 14–19; d examples at body 9–10), 526 never solved. Minimal pair from the
pool (d, body 9, class A removed by swapping):
```
rl_13846 (all 4): ((~a0)>(~a1)), (a2>(~a0)), (a2 v a3), (~(a3 & a1)) |- (~a1)   NEGI proves the goal ~a1
rl_5198  (none) : ((~a0)>(~a1)), (a2>(~a1)), (a2 v a3), (~(a3 & a0)) |- (~a1)   NEGI proves ~a0, then IMPE
```
c-family examples: rl_2568 (body 14; needs `~S` as a lemma inside the `Q>` box), rl_3953.

Constructed probe 2 (probe_synth.py, 10 random relabelings + premise shuffles each, golds Lean-checked, untied s1 k=8):

| template (9 body lines, 4 premises, NEGI box in the 2nd branch) | samples ok | instances solved |
| --- | --- | --- |
| `negi_direct`: (A v B), (A > D), (D > ~C), ~(B & C) ⊢ ~C — box proves the goal | 68/80 | 10/10 |
| `negi_inter`: (A v B), (A > ~G), ~(B & C), (~C > ~G) ⊢ ~G — box proves ~C, then IMPE | **0/80** | 0/10 |
| `negi_direct` + 4 distractor premises | 63/80 | 10/10 |
| `impi_second`: →-intro box for the lemma (C > D) in the 2nd branch (10 lines) | 5/80 | 3/10 |
| `impi_first`: same, box in the 1st branch | 0/80 | 0/10 |

What it writes on `negi_inter`: it never opens a `fun (n : C) => …` box for ¬C; it re-runs Or.elim on the same
disjunction inside the branch and forges `⟨ n115 , n115 ⟩ : U ∧ Q` to reach False (8/8 LEANREJ).

Hypothesis B: the model only opens a box whose conclusion is the *current* goal (the enclosing box's target); it has
no "prove a lemma, then use it" move for negations/implications because pretraining proofs (≤ 6 lines) only open a box
as the last step toward the goal. Would be fixed by pretraining/RL targets whose minimal proof contains a box at
depth ≥ 1 whose conclusion is consumed by a later IMPE/NEGE (a `have` lemma), or by a hint channel for the lemma.
Check: count in pretraining the NEGI/IMPI lines at depth ≥ 1 whose conclusion is not the enclosing goal (not done).

## Class C: classical reductio (assume ¬goal … NEGI … DN)

Old dev: the never-solved textbook schemas above (61 theorems). RL pool: 271 targets whose gold uses DN (all d family),
262 never, 9 by one arm, 0 by all. Examples: rl_566 (body 10: assume ~R inside a branch, derive F, NEGI ~~R, DN R),
rl_867. Pretraining *does* contain the reductio shape: 9,663 of 155k proofs end in DN with the goal's negation assumed
(6%) — but at ≤ 6 lines, i.e. assume ¬G, one or two IMPE, NEGE, NEGI, DN.

Probe 2, same protocol: `excluded_middle` ⊢ (A v ~A) (9 lines, nested NEGI) **0/80**; with 3 distractors 0/80;
`peirce` ((A>B)>A) ⊢ A (9 lines) **0/80**; control `nested_negi_constructive` ~(A v B), chain ⊢ ~C (8 lines, same
box depth, no reductio) 66/80, 10/10. Failure text: for excluded middle it writes `Or.inl n74` with an unbound name
(fakes a premise, 8/8 `LEANPARSE unbound`); for Peirce `have n102 : U := n101 n101` (fakes modus ponens on an atom).

Hypothesis C: the model never assumes the negation of a *compound* goal (disjunction, implication) or uses the
assumed ¬G more than once (both reductio probes need ¬G twice: once inside the inner box, once outside); pretraining
reductios are atom-goal, single-use. Would be fixed by ≥ 7-line reductio proofs in pretraining (EM, Peirce, De Morgan
or→nand) — these are 9-liners, well inside the RL band, yet 0/80, so it is a missing primitive, not length.

## Class D/E: long and "packed" proofs (planning, no single token)

c family body 13–15 (all never solved): gold log-prob is diffusely low (mean −50 vs −38 for solved s-family 13-liners);
worst tokens are formula choices (`have n12 : ( Q …` argmax `P`), `∧` vs `→`, not a structural token. These proofs
chain `( T & R ) > ( Q > P )` packed premises through 3 box levels; the failure looks like search, not syntax. Every
target ≥ 20 lines is unsolved except 266 solved by some arm (max 24 lines). Not probed further here.

## What I'd test next on a GPU

1. Class A fix, cheapest: `Or.symm` trick or branch reordering in the renderer, then re-evaluate dev/old-dev. Expect
   +57 dev (class A never) and roughly +500 of the 931 pool targets with no retraining. Alternatively add ~200
   first-branch-nested proofs to pretraining and rerun the 3 h RL; measure class-A solve rate by band.
2. Class B/C: add short (7–10 line) targets to the RL pool / pretraining that require an intermediate ¬-lemma or a
   compound-goal reductio, in isolation (the synthetic templates here, relabeled), and check whether EI picks them up
   at 8 samples/round — my prior is that B needs a few hundred pretraining examples, C maybe fewer.
3. Confirm the counting mechanism: sample the model on a case split whose first branch contains *two* sequential
   inner Or.elims (four blocks) and on one whose second branch does; the flat-counter story predicts failure on
   both, a stack story predicts only the first.
4. c family: give the model the packed-premise lemma (`have (T ∧ R)`) as a prompt hint and see whether the rest is
   solvable; separates "search" from "syntax".

## Files

- `analyze.out`, `step2.out`–`step9.out`: set analysis, feature tests, minimal pairs, log-prob localisation.
- `probe_swap.py` → `probe_swap_results.json`, `probe_swap_looped_s0.json`, logs `probe_swap*.log`.
- `probe_synth.py` → `probe_synth_results.json`, `probe_synth_looped_s0.json`, logs `probe_synth*.log`.

## Appendix: synthetic probe on looped s0 (8 loops), same templates
looped s0: template -> samples ok /80, instances solved /10
impi_first                   ok   0/80  solved 0/10
impi_second                  ok   0/80  solved 0/10
negi_direct                  ok  72/80  solved 10/10
negi_inter                   ok   0/80  solved 0/10
excluded_middle              ok   0/80  solved 0/10
peirce                       ok   0/80  solved 0/10
nested_negi_constructive     ok  52/80  solved 10/10
impi_second_distract4        ok   1/80  solved 1/10
negi_direct_distract4        ok  60/80  solved 10/10
excluded_middle_distract3    ok   0/80  solved 0/10
