# Textbook 72: weaknesses and where to diversify

What the 72 can't measure, and which kinds of real textbook exercises would fix that. It builds on
plan section 2 (families and length bins) and adds axes that the plan doesn't cover yet.

**Ground rules.**
- Every number here is aggregate, from the local, gitignored `textbook72/overview.md` and
  `tables.md`. No problem from the 72 is quoted or used as a template (`CLAUDE.md`).
- Every proposal is met by **selecting and tagging real exercises**, never by generating or
  editing them (plan section 1).
- Axes marked **(new tag)** need a measurement that `check.py` doesn't compute yet. Axes marked
  **(decide)** change what counts as a problem and are Charles's call.

Starting points (Charles, 2026-09-28):
1. Some proofs are easier forward than backward.
2. Can the model do proof strategy?
3. Ask about proof routine and formal logic.
4. A subset of proofs or fragments gets reused.

Sections 1–4 take these in turn. Section 5 lists further weaknesses. Section 6 ranks them.

## 1. Direction: forward versus backward proofs

**The weakness.** The 72 don't separate problems solved by working *backward* from the goal from
problems that need *forward* work on the premises.
- **Backward** means the goal's main connective dictates the next step: assume A to prove A→B;
  prove each conjunct of a conjunction.
- **Forward** means the goal gives no hint, and the premises must first be broken down with
  ∧-elim, →-elim or a case split.

The two call for different skills, and the model's pre-training data is mostly the backward kind:
→-intro and →-elim, and half its proofs have no subproof at all.

**What the 72 show (aggregate).**
- Goal main connective: → 21, ∨ 14, ¬ 10, ↔ 10, atom 9, ∧ 4, ⊥ 4.
  - Goals that are atoms or ⊥ (13) give no backward hint at all.
  - ∨ goals (14) are the textbook trap: ∨-intro applied too early commits to the wrong disjunct.
    The proof usually needs a forward case split or a reductio instead.
- Elimination rules are heavily over-represented against pre-training: ORE 20×, ∧-elim 18×.
- The model solves 0 of the problems whose shortest proof uses ORE.

The 72 therefore mix forward and backward difficulty inside the same families. A failure can't be
attributed to either.

**What to add.**
- Exercises whose goal is an atom or a ∨ and whose premises must be taken apart. Examples are the
  disjunctive syllogism, constructive dilemma and ∧/∨ distribution exercises that every textbook
  has.
- Enough purely backward exercises at the same length, such as nested conditionals and currying,
  to give the forward ones a control group.

**How to measure (new tag `direction`).** Two searches per problem, both with the fork's
`minlen` rule set and the same budget:
- **backward-first:** introduction rules chosen by the goal's shape, eliminations only when stuck.
  This is the order most textbooks teach as "proof strategy".
- **forward-first:** saturate the premises with eliminations, then introduce.

Tag the problem `backward`, `forward` or `either` by which search finds a proof within the budget,
and record the share of proof lines that are eliminations. Report the solve rate for each
`direction` × length bin.

## 2. Strategy: does the model pick the right one?

**The weakness.** The 72 score only whether a complete proof verifies. A 0 is the same whether
the model chose a strategy that can't work, or chose the right one and made a mistake later.
With 63 of 72 problems at 0, that difference is invisible.

**What the 72 show (aggregate).**
- **Classical problems:** 19, and every proof of them needs reductio or DN. The model solves 1,
  the easiest.
- **Case-split problems:** 21 use a case split and 6 need one on a derived disjunction. The model
  solves none of them.
- It is unknown whether the model even opens these proofs with the right assumption: ¬G for
  reductio, or the two disjuncts for a case split.

**What to add.**
- **Strategy labels on every problem:** direct, conditional proof, reductio, indirect proof
  (reductio + DN), case split, case split on a derived disjunction. The label is measured the way
  the plan's families are: "needs" is proven by restricted search or G4ip.
- **Real exercises where the surface points to the wrong strategy.** Examples are ∨ goals that
  need reductio (excluded middle, linearity) and ¬ goals that are best proved by case split.
  Textbook strategy chapters (e.g. *forall x: Calgary*, "proof strategy") often pair these on
  purpose.
- At least 20 per strategy (plan section 2 default), spread over length bins.

**How to measure (evaluation metric, no new data format).**
- **First-move strategy match:** parse the model's first assumption line and compare it with the
  set of strategies that can work (from the searches above). Score it on every sample, including
  failed ones.
- **Execution given the right strategy:** the solve rate among samples whose first move matched.

This splits "doesn't know what to do" from "knows but can't finish". It uses outputs the ws3
evaluator already saves.

## 3. Asking about proofs, not only for them (decide)

**The weakness.** Every one of the 72 has the same task: here is a sequent, write the whole proof.
Textbook exercise sets also ask about proofs:
- (a) **complete a proof:** some lines are given, finish it;
- (b) **justify:** a proof without rule names; supply each line's rule and citations;
- (c) **find the error:** a flawed proof; which line breaks a rule;
- (d) **formal logic questions:** is this sequent valid? Is this a tautology? Give a
  countermodel.

These test the proof routine directly, and each has a textbook source.

**What fits the current format.**
- **(a) fits.** The prompt is `THM <premises> SEQ <goal> PRF` and the model continues. Giving the
  printed first k lines after `PRF` is a partial-proof prompt that the verifier checks unchanged.
  Real "complete the proof" exercises give the prefix, so no line is invented. It also separates
  strategy (given) from execution (asked), which ties into section 2.
- **(b)–(d) don't fit.** They need a new output format and a new scorer, which the plan places out
  of scope. They are also a different skill from writing proofs, since the model was never trained
  to answer in words.

**Proposal.** Include (a) as a separate, labelled slice, `completion`, with its own headline
number, never pooled with full proofs. Record (b)–(d) as a later probe. **(decide):** whether (a)
is in v0.1.

## 4. Reused fragments and lemmas

**The weakness.** Textbook proofs are built from a small stock of recurring pieces:
- the derivation of excluded middle;
- double negation;
- modus tollens and contraposition;
- De Morgan;
- disjunctive syllogism.

Most textbooks prove these once as "derived rules", then use them inside longer proofs. The 72
treat each problem as unrelated. So they can't tell whether the model fails a long proof because
it lacks a piece or because it can't put pieces together.

**What the 72 show (aggregate).**
- **Missing pieces:** the 9 absent schemas are mostly these building blocks (contraposition in both
  directions, modus tollens, excluded middle, explosion, constructive dilemma).
- **Pieces without their compounds:** the laws that are present appear on their own (4 De Morgan
  directions, the DN laws), but rarely together with a longer proof that uses them.

**What to add.**
- **Lemma and compound pairs from the same book:** the derived-rule proof, plus the book's later
  exercises whose proofs contain it. Both are real exercises; the pairing is only a tag.
- The missing standard laws, each on its own (plan section 2 already asks for these).

**How to measure (new tags `fragments`, `lemma_of`).**
- Cut every reference proof into its subproofs and canonicalize each under renaming (`canon_key`).
  Count the fragments that recur across the set.
- Tag each problem with the fragments it contains, and each lemma exercise with the compounds that
  use it.
- **Composition gap:** the solve rate on compounds whose lemmas the model solves on their own,
  compared with compounds whose lemmas it fails.
- Report how much of the set the 10 most frequent fragments cover, so that one shared fragment
  can't quietly drive a result.

## 5. Other weaknesses

| Weakness | Evidence (aggregate, the 72) | Fix with real exercises |
| --- | --- | --- |
| **Too few per class to see a change** | 10–20 problems in each hard class. At n = 20, 0 solves still allows a true rate of up to 15% (95%, rule of three). | 50 or more per hard class in the new set; plan section 2 asks for at least 20. |
| **No difficulty gradient** | 9 solved, 63 at 0, nothing in between. | Include easy and middle exercises from the same chapters as the hard ones, so each family has a slope. |
| **Hollow middle lengths** | Shortest proof: ≤4: 7; 5–6: 11; 7–10: 19; 11–15: 25; 16+: 9. The frozen model's frontier is at 6 lines. | At least 60 at 5–10 lines (plan default). Textbooks' early exercise sets are mostly here. |
| **Length and shape confounded** | Every 7+ line problem is a case split, classical or nested. | Select long, purely conditional or conjunctive exercises (long hypothetical-syllogism chains, currying), so length can be varied without the hard shapes. |
| **Surface versus proof** | Some 1–2 line problems fail, probably because of unusual formulas. The ↔ encoding (A→B)∧(B→A) inflates both formula and proof: 10 goals. | Tag `formula_size` and `iff_count`. Include short exercises with large formulas and long ones with small formulas. |
| **Irrelevant premises** | 2 problems have unused premises, and 3 have inconsistent premises with a goal that isn't ⊥. | Keep textbook exercises with distractor premises as a tagged slice (`unused_premises`, `vacuous`), reported separately. |
| **Unbalanced rule coverage** | Rule use ranges from under 13% (BOTE) to 58% (NEGE) of proofs. | Report the rule-use histogram for each release. Select toward rules under 10%. |
| **One source** | Two catalogs, one curator. Style and notation are uniform. | Several books, results reported per book (plan section 4). |
| **Atom limit** | The tokenizer has only P, Q, R, S; 8 problems use all 4. | A fixed constraint. Log the exercises excluded for it, so the bias is visible. |
| **Theorems versus sequents** | 16 theorems and 56 sequents. | Tag `form`. Textbooks give theorems in their own sections, so select to about 25%. |
| **Reference proofs aren't canonical** | Dmitry's references were longer than the shortest found on 17 of 57. | Store both the reference and `minlen`'s shortest; bin by the shortest (plan section 3). |

## 6. Priorities

Ranked by how much each adds per unit of work, for v0.1:

1. **Size and middle lengths** (section 5): without these, nothing else can be measured.
   Already in the plan.
2. **Strategy labels and the first-move metric** (section 2): new data isn't required. It needs
   only the searches `check.py` runs anyway and a parser for the model's first assumption. This is
   the most direct answer to "can the model do proof strategy".
3. **The `direction` tag** (section 1): two extra searches per problem, then selection toward the
   thin cells.
4. **Lemma and compound pairs** (section 4): depends on which books T3 turns up. *forall x:
   Calgary*'s derived-rules chapter is the first candidate.
5. **The completion slice** (section 3a): valuable, but a new slice. Waits on Charles's decision.

**Changes to the plan this would bring, if adopted:**
- new measured fields `direction`, `strategy`, `fragments`, `lemma_of`, `formula_size`, `form`;
- the first-move metric in section 6;
- a `completion` slice, if (a) is approved.

## 7. Checklist: standard laws not present in the 72

9 of the 26 textbook schemas in the fork's `textbook_pool.py` match none of the 72, whether
directly, in theorem form, with conjoined premises, or as a biconditional. This is an aggregate
result from the local `textbook72/tables.md`. Every new source should be checked against this
list. Each law is worth including both on its own and inside longer proofs (section 4).

| Schema (fork name) | Form | Proof shape | Seen in open sources (`sources.md`, pass 1) |
| --- | --- | --- | --- |
| `contraposition` | A→B ⊢ ¬B→¬A | conditional + negation, intuitionistic | Calgary (interderivable pairs), Carnap ch. 6 |
| `contraposition_conv` | ¬B→¬A ⊢ A→B | classical (needs DN) | Calgary (IP strategy list), Carnap ch. 6 |
| `modus_tollens` | A→B, ¬B ⊢ ¬A | negation intro | Calgary and Magnus (derived-rules chapters) |
| `export` | (A∧B)→C ⊢ A→(B→C) | nested conditional | Calgary (strategy list) |
| `excluded_middle` | ⊢ A∨¬A | classical, reductio | Calgary (theorems; "Indirect proof of excluded middle"), Magnus |
| `explosion` | A, ¬A ⊢ B | BOTE | Carnap ch. 6 (a variant with a negated goal) |
| `constructive_dilemma` | A∨B, A→C, B→D ⊢ C∨D | case split, worked forward | Calgary (strategy list) |
| `negated_conditional_conv` | A∧¬B ⊢ ¬(A→B) | negation intro | Calgary (strategy list; interderivable pairs) |
| `dist_and_over_or_conv` | (A∧B)∨(A∧C) ⊢ A∧(B∨C) | case split | Calgary (strategy list) |

"Seen in" means only that a matching exercise was noticed during discovery. It hasn't been
transcribed or checked yet.

**Things to watch when adding these:**
- **Leak risk from plain-atom instances.** `validation_36` names the same schemas with atoms, and
  `rl_targets` / `transfer` hold instances of them. A textbook exercise in plain atoms (e.g.
  P→Q ⊢ ¬Q→¬P) is likely a renaming of a `validation_36` item. If so, the stage 6 audit removes
  it. Report how many are lost this way, and prefer exercises whose instances use compound
  subformulas.
- **Tag every one.** Each carries `schema` and `schema_in_rl_pool` (parent plan section 2), and
  results are reported with and without them. All 9 schemas are in the fork's pool, so an
  improvement on them may reflect RL exposure rather than new capability.
- **Mix the shapes.** The 9 cover five different shapes: classical, case split, nested
  conditional, negation intro and explosion. They add breadth, not only volume.
- **If a law turns up in no source,** record it in the S6 gap report. Don't write an instance
  ourselves.
