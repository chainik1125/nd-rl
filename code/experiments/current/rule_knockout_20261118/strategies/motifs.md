---
written_on: 2026-11-18
written_by: agent:claude
---

# Data-driven proof motifs as knockout candidates

Snapshot: control streams read 2026-11-18 20:40 PST (full-ei_s0: 64 rounds, 451,536 samples, 8,542 accepted proofs;
full-ei_s1: 72 rounds, 491,240 samples, 13,293 accepted proofs; 21,835 proofs on 5,462 distinct targets). The streams
were still growing, so control numbers drift by a few percent between scripts.

Corpora:
- pretraining: 154,990 proofs (`long_rl_20261014/data/pretrain.jsonl.gz`), no proof has an unused body line.
- RL references: 13,946 of 18,439 targets have a reference proof; 3,533 of those have body <= 14, the range the
  control reaches in ~75 RL minutes ("reachable"). The 4,493 ladder targets have no reference.
- control: Lean-accepted proofs in `artifacts/rule_knockout_20261118/live/full-ei_s{0,1}/found_stream.jsonl`.
  Model proofs are unpruned, so every detector looks only at lines backward-reachable from the final line.

## Code (all in this directory)

| file | what |
| --- | --- |
| `ndlib.py` | parser: line index, depth, formula tree, rule, citations, boxes, liveness |
| `motifs.py` | `motifs(Proof) -> set[str]`, every detector |
| `verify.py` | independent text-level re-implementation of the 8 recommended filters; agrees on all 154,990 records (0 mismatches) |
| `sanity.py` | rule argument conventions checked on every line (IMPE major-first, NEGE positive-first, ORE = disj + 2 boxes) |
| `ngrams.py` -> `ngrams.tsv` | 1,764 dependency bigrams/trigrams and box-nesting patterns, ranked |
| `tabulate.py` -> `motif_table.tsv` | every motif x corpus, plus first round seen in each control seed |
| `final.py` -> `final_out.tsv` | candidate table incl. avoidability, round-1 counts, pretraining-deletion overlap |
| `firsthit.py` -> `firsthit_out.tsv` | first-hit round, samples and spread per motif per control seed |
| `cost.py` -> `cost_out.txt` | signature-novelty cost proxy (weak, see below) |

Full path: `/private/tmp/claude-501/-Users-robbie-repos-nd-rl/72c64566-c4a9-4e6a-90f1-be85b62895cc/scratchpad/strategies/`

## Finding 1: pretraining and the RL targets barely share multi-step motifs

Pretraining is dominated by single introductions (ORI in 36.5% of records, IMPI 37.4%, mostly `A > A` or vacuous).
RL references are IMPE chains inside case splits. So most motifs that carry the RL targets sit at <= 1.5% of
pretraining, and several are exactly zero. A motif knockout therefore costs almost nothing in corpus size.

| dependency pattern | pretraining | RL refs | control proofs |
| --- | --- | --- | --- |
| IMPE result used as the minor premise of an IMPE | 1.53% | 77.9% | 25% |
| IMPE result used as the major premise of an IMPE | 0.14% | 77.3% | 9% |
| ANDE result used as major premise of IMPE | 0.05% | 61.7% | 18% |
| first ORE branch ends in IMPE | 0.01% (22) | 80.8% | 8.5% |
| three chained IMPE | 0 | 40.2% | 15% |
| ORE nested in an ORE branch | 0 | 60.8% | 0.2% |

## Finding 2: natural knockouts already measured in the control (no new run needed)

Motifs with zero pretraining records. "Attempts" = samples drawn before the first-hit round x share of the pool whose
reference (body <= 14) contains the motif; this assumes uniform target sampling and ignores ladder targets, so it is
an order-of-magnitude figure (inferred, not logged).

| motif | filter | pretraining | RL refs | reachable refs | first hit s0 | first hit s1 | control proofs now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| disjunctive syllogism | ORE branch ends in BOTE of a NEGE | 0 | 4,155 (29.8%) | 499 | r5, 43k samples (~1.2k attempts) | r1 | 1,283 (385 targets) |
| 3-chain | IMPE <- IMPE <- IMPE through minors | 0 | 5,602 (40.2%) | 1,447 | r9, 83k (~6.5k) | r10, 91k (~7.2k) | 3,318 (949) |
| dilemma | both ORE branches end in IMPE | 0 | 10,252 (73.5%) | 1,915 | r13, 121k (~12.5k) | r13, 114k (~11.8k) | 1,712 (738) |
| 4-chain | as 3-chain, four | 0 | 2,257 (16.2%) | 621 | r14, 133k | r17, 146k | 1,472 (486) |
| case split inside NEGI box | ORE whose enclosing box is closed by NEGI | 0 | 781 (5.6%) | 222 | r25, 210k | r5, 42k | 82 (42) |
| nested case split, second branch | ORE inside ORE.b2 | 0 | 5,257 (37.7%) | 433 | never (>452k) | r21, 174k | 34 (20) |
| nested case split, first branch | ORE inside ORE.b1 | 0 | 7,098 (50.9%) | 427 | never | never | 0 |
| case split aimed at F | ORE line whose formula is F | 33 | 719 (5.2%) | 458 | never | never | 0 |
| depth 4 | any line at depth >= 4 | 0 | 2,320 (16.6%) | 7 | r61 | never | 2 |

Read: three of these (syllogism, 3-chain, dilemma) emerged in both control seeds within 21 RL minutes and then spread to
hundreds of targets; two never appeared in ~940k samples. That is already a graded emergence ladder. Caveat: first-hit
round mixes first-hit cost with how many other hard steps the cheapest target needs (first targets had body 4-7).

## Finding 3: recommended knockouts (need a run)

Columns: del = pretraining records deleted; RL = reference proofs containing it (all / body <= 14); ctl = share of
control accepted proofs and distinct targets with such a proof (of 5,462); avoid = share of those targets for which the
control also holds an accepted proof without the motif (low = forced by the theorem, high = a choice); r1 = accepted
proofs with the motif in round 1 of the two control seeds (of 567), i.e. what the pretrained model does before RL.

| # | motif | exact filter (drop record if...) | del | RL all | RL <= 14 | ctl proofs | ctl targets | avoid | r1 | first-hit cost (inferred) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `mp_chain2` forward chaining | an IMPE line whose 2nd citation is an IMPE line | 2,377 (1.53%) | 10,860 (77.9%) | 2,615 (74.0%) | 25.3% | 1,416 | 5% | 2 | 1 line; no new token, no new context, only which line is cited |
| 2 | `depth>=3` triple nesting | any line with three `\|` | 5,664 (3.65%) | 7,436 (53.3%) | 487 (13.8%) | 19.2% (s0 2.7%, s1 29.8%) | 2,309 | 20% | 45 | 3 lines + two extra consecutive closes; a whole box wrapper at an unseen level |
| 3 | `bote_in_box` ex falso under a hypothesis | a BOTE line at depth >= 1 | 1,283 (0.83%) | 4,469 (32.0%) | 740 (20.9%) | 18.9% | 1,170 | 51% | 37 | 2 lines: F derived in a box whose goal is not a negation, then BOTE |
| 4 | `ore_any_work` non-degenerate case split | an ORE line `ORE d a1 e1 a2 e2` with e1 != a1 or e2 != a2 | 1,908 (1.23%) | 12,454 (89.3%) | 2,879 (81.5%) | 19.0% | 1,489 | 8% | 4 | 2 lines: a derived line inside a branch, and an ORE citing it as branch end |
| 5 | `ore_in_impi` case split under a conditional proof | an ORE line at depth >= 1 whose innermost enclosing box is closed by IMPI | 158 (0.10%) | 2,926 (21.0%) | 359 (10.2%) | 5.6% | 525 | 36% | 11 | 5 lines (two AS, two branch ends, ORE) all in an unseen context |
| 6 | `mp_curried` curried modus ponens | an IMPE line whose 1st citation is an IMPE line | 218 (0.14%) | 10,776 (77.3%) | 1,764 (49.9%) | 9.1% | 765 | 8% | 2 | 1 line |
| 7 | `negi_in_impi` reductio under a conditional proof | a NEGI line at depth >= 1 whose innermost enclosing box is closed by IMPI | 1,060 (0.68%) | 806 (5.8%) | 8 (0.2%) | 2.8% | 230 | 40% | 16 | 3 lines (AS at depth 2, NEGI at depth 1, IMPI over a NEGI) |
| 8 | `ore_on_derived` case split on a derived disjunction | an ORE line whose 1st citation is neither PR nor AS | 71 (0.05%) | 9,735 (69.8%) | 1,079 (30.5%) | 4.9% | 421 | 11% | 2 | 1 line |

Pretraining deletions of 1, 2, 3 are pairwise disjoint (0 shared records); 3 and 4 share 54.

### Is it a rule in disguise?

Share of each rule's pretraining records that the filter deletes:
- `mp_chain2`: 4% of IMPE records. Not a rule knockout. (But 79% of IMPE-using RL references contain it.)
- `depth>=3`: 10% of IMPI records.
- `bote_in_box`: 43% of BOTE records. 1,668 BOTE records stay, all at depth 0 (2,482 pretraining records have the trigram premise -> NEGE -> BOTE).
  In the RL references every BOTE is inside a box (4,469 of 4,469), so on the target side this is "all the BOTE the
  targets want". That is the point of the arm: the `.elim` token is known, its use under a hypothesis is not.
- `ore_any_work`: 84% of ORE records. Closest to a rule knockout. What stays (369 records) is only the degenerate
  `A v A |- A` shape with two empty branches. Flag this when presenting.
- `ore_in_impi` 7% of ORE; `negi_in_impi` 7% of NEGI; `mp_curried`, `ore_on_derived` < 1%.

### One real example each (live lines only)

1. `mp_chain2`, train_127246: `N8 P : IMPE N1 N6 ; N9 ( R > R ) : IMPE N7 N8`
2. `depth>=3`, control ladder_3081: `N2 | R : AS ; N3 | | R : AS ; N4 | | | S : AS ; N5 | | ( S > S ) : IMPI N4 N4 ; N6 | ( R > ( S > S ) ) : IMPI N3 N5 ; N7 ( R > ( R > ( S > S ) ) ) : IMPI N2 N6`
3. `bote_in_box`, control ladder_3537: `N1 ( ~ P ) : PR ; N4 | P : AS ; N5 | F : NEGE N4 N1 ; N6 | S : BOTE N5 ; N7 ( P > S ) : IMPI N4 N6`
4. `ore_any_work`, train_124535: `N4 | ( P & ( ~ S ) ) : AS ; N5 | ( S v R ) : AS ; N6 | ( P & ( ~ S ) ) : IMPE N2 N5 ; N7 ( P & ( ~ S ) ) : ORE N1 N4 N4 N5 N6`
5. `ore_in_impi`, train_146142: `N3 ( Q v Q ) : PR ; N5 | ( ~ Q ) : AS ; N6 | | Q : AS ; N7 | | Q : AS ; N8 | Q : ORE N3 N6 N6 N7 N7 ; N9 ( ( ~ Q ) > Q ) : IMPI N5 N8`
6. `mp_curried`, train_70396: `N2 ( Q > ( Q > S ) ) : PR ; N3 Q : PR ; N11 ( Q > S ) : IMPE N2 N3 ; N12 S : IMPE N11 N3`
7. `negi_in_impi`, train_124395: `N2 ( ~ ( ~ ( ~ S ) ) ) : PR ; N3 | ( ~ ( ~ S ) ) : AS ; N4 | | ( ~ R ) : AS ; N5 | | F : NEGE N3 N2 ; N6 | ( ~ ( ~ R ) ) : NEGI N4 N5 ; N7 ( ( ~ ( ~ S ) ) > ( ~ ( ~ R ) ) ) : IMPI N3 N6`
8. `ore_on_derived`, train_142206: `N3 ( P v P ) : IMPE N2 N1 ; N4 | P : AS ; N5 | P : AS ; N6 P : ORE N3 N4 N4 N5 N5`

Natural zero, disjunctive syllogism, control ladder_2100: `N1 ( ( P v Q ) v S ) : PR ; N2 ( ~ ( P v Q ) ) : PR ; N3 | ( P v Q ) : AS ; N4 | F : NEGE N3 N2 ; N5 | S : BOTE N4 ; N6 | S : AS ; N7 S : ORE N1 N3 N5 N6 N6`

## First-hit cost: what is measured and what is not

- Not measured: the model's probability of the motif. Nothing here loaded a checkpoint.
- The "first-hit cost" column is my count of lines in the cheapest instance whose pattern (rule, kind of line cited,
  box context) cannot occur in the filtered pretraining set. Inferred by reading instances.
- `cost.py` tried to automate that count with line signatures (rule + source rules, and rule + box type + depth) and
  "extra novel lines relative to full pretraining". Medians over control proofs containing the motif: `mp_curried` 1,
  `mp_chain2` 2, `bote_in_box` 2, `depth>=3` 2, `ore_in_impi` 3, `negi_in_impi` 4, `ore_any_work` 4. The minimum is 0
  for most, because control proofs already contain many signatures that are absent from the full pretraining set. So
  the proxy orders the motifs roughly as the hand count does, but it is weak evidence.
- Measured: first-hit rounds for natural zeros (Finding 2).

## Things that look like good motifs and are not

- Classical reductio (`DN` citing a `NEGI`): 10,554 pretraining records (6.8%, 81% of DN records), 268 RL references
  (1.9%), 43 control proofs (0.2%). Too rare in the targets to read, and it is Dan's earlier knockout.
- Modus tollens, hypothetical syllogism and De Morgan steps as textbook shapes: 58-123 pretraining records each,
  <= 0.2% of RL references (except `demorgan_andi`, 1.4%). The targets do not use them. Contraposition is row 7.
- "Lemma cited twice": 4,095 pretraining (2.64%), 49.2% of RL references but 6.8% of reachable ones, 4.3% of control,
  58% avoidable.
- Vacuous boxes (assumption unused): IMPI 39,693 pretraining records (25.6%), NEGI 7,807 (5.0%); 2.0% and 0% of RL
  references. A pretraining artefact, not a target strategy.
- `impi_in_impi` / `depth>=2`: 13.4% / 14.3% of pretraining, over the 10% budget.

## Caveats

- Reachability. The control solves almost nothing above 14 body lines in 75 minutes (s1: 1,764 at 0-6, 3,124 at 7-9,
  303 at 10-14, 12 at 15-19). "RL all" overstates what a 1-3 h run can show; use "RL <= 14" and the control columns.
- `depth>=3` has a large seed effect in the control itself (s0 227 proofs on 156 targets, s1 3,957 on 2,293), the same
  wall as the 2026-10-14 note. A knockout arm needs >= 3 seeds and the comparison is "0 of 3" against "1 of 2".
- 4,493 ladder targets have no reference, so RL-reference shares say nothing about them; the control columns do.
- Forced versus chosen: `mp_chain2`, `ore_any_work`, `mp_curried` are forced by the theorem (5-8% avoidable), so
  non-appearance means the targets stay unsolved. `bote_in_box` is 51% avoidable, so non-appearance can also mean the
  model took another route (the substitute problem already noted for the `exfalso` arm).
- Avoidability is measured on targets the control solved both ways; it is a lower bound on what is avoidable.
