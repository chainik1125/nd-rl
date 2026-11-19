---
written_on: 2026-11-18
written_by: agent:claude
status: rough notes, cut short at the coordinator's request
---

# Proof-theoretic ways to split propositional proofs into strategies

Scratch code (read-only on the repo): `pt_work/flags.py` (per-proof flags), `pt_work/g4ip.py` (intuitionistic and
minimal prover, self-tested on 14 textbook formulas), `pt_work/classify.py`. All in this directory.

Status words: **measured** = I ran it on our data; **cited** = a search confirmed the reference; **inferred** = my own
reasoning, not checked against a source or data.

## 0. Framing

Proof theory separates three things that "one rule = one strategy" mixes up:

1. **Forced moves.** Invertible rules: applying them never loses provability. For a goal, IMPI, NEGI, ANDI. For a
   hypothesis, ORE, ANDE, BOTE. Not strategies.
2. **Real choices.** Non-invertible rules. In Dyckhoff's contraction-free calculus LJT/G4ip there are two: which
   disjunct to prove (ORI1 or ORI2), and using a hypothesis `(C → D) → B`, which means first proving the implication
   `C → D`. Classical logic adds one more: when to switch to reductio. (Dyckhoff, JSL 1992, cited. The "exactly two
   non-invertible rules" statement is from memory; the citation check did not confirm that sentence.)
3. **Optional moves.** Detours (lemmas, cut) and non-subformula steps. Normalization says no theorem needs them
   (Prawitz 1965, cited). They only shorten proofs.

Under Curry-Howard with `¬A := A → ⊥`, NEGE is IMPE and NEGI is IMPI. Our Lean rendering already merges them
(`nF nA`; a `fun` box). So modus tollens, hypothetical syllogism and contraposition are all one thing, function
composition (inferred; standard). This fits the knockout result: IMPE, whose surface form survives through NEGE, is
the one that came back.

## 1. Measured on our data

Per-proof flags, live lines only. Pretraining = 154,990 records. RL references = 13,946. Control = the 12,505
accepted proofs in `full-ei_s1/found_stream.jsonl`.

| flag (share of proofs having it) | pretraining | RL references | control run |
| --- | --- | --- | --- |
| box assumption cited twice or more (contraction) | 1.24% | 83.11% | 12.83% |
| any premise or assumption cited twice or more | 8.32% | 91.07% | 36.31% |
| vacuous discharge (IMPI/NEGI box never uses its assumption) | 30.65% | 2.01% | 58.41% |
| derived line cited twice or more (sharing) | 3.88% | 49.25% | 6.40% |
| detour (intro then matching elim) | 0.17% | 0.04% | 0.09% |
| argument box (IMPI/NEGI line feeds a minor premise) | 0.93% | 0.94% | 2.29% |
| non-analytic line (not a subformula, up to two negations) | 0.17% | 0.00% | 0.30% |
| reductio (DN on a NEGI-built double negation) | 6.81% | 1.92% | 0.26% |
| reductio that never uses its assumption | 2.81% | 0.00% | 0.03% |
| DN on a double negation not built by NEGI | 1.55% | 0.00% | 7.42% |
| dead lines | 0.00% | 0.06% | 19.32% |

Max box depth: pretraining 0 to 3, RL references 0 to 6, control 0 to 3.

Theorem labels on the 18,439 RL targets, by G4ip, in 2 seconds with no timeouts:

| label | targets | of which have a reference proof |
| --- | --- | --- |
| minimal-provable | 13,361 | 9,651 |
| intuitionistic, not minimal | 4,509 | 4,051 (all use BOTE) |
| classical only | 569 | 244 (all use DN) |

397 references use BOTE on a minimal-provable theorem and 24 use DN on one, so rule use overstates need.
The same classification of the pretraining file was stopped unfinished.

Caveats on the table:

- The "used twice" counts are citation counts. They count one use in each ORE branch as two. The two RL reference
  reductios I inspected that cite their assumption twice were both of this kind. A linear-logic-correct count
  (maximum over branches) is not implemented.
- The control column is one seed.
- G4ip with the nested-implication rule switched off is not the same as "no argument box" in Fitch: it proves
  `¬¬(A ∨ ¬A)` without that rule, while the Fitch proof has an argument box. Do not use it as the need test.

## 2. Candidates

### A. Structural flags: contraction and vacuous discharge

- **Definition.** For each box assumption: cited zero times (weakening, K), once (linear), or twice or more
  (contraction, W). Two binary flags per proof. Sources: Church's λI against λK (1941); BCI = linear, BCK = affine
  (nLab, cited); BCIW = relevant implication and SK = intuitionistic (SEP "Combinatory Logic", cited); Girard 1987.
  Hindley 1997 was not confirmed for the relevant-logic part.
- **Complete and unique.** Yes: every proof falls in exactly one of four classes, and the two flags are independent.
- **Choice or forced.** Partly forced by the theorem (`(A → A → B) → A → B` needs contraction, `A → B → A` needs
  weakening), partly free.
- **Detect and remove.** Count citations of each AS line in the live cone. Drop proofs with the flag. Dropping
  assumption-contraction costs 1.24% of pretraining.
- **Need test.** Bounded search for a linear proof. Cheap at our sizes; not built.
- **First-hit cost.** One: cite a name a second time. No new token, no new syntax.
- **Why it matters.** 83% of RL references reuse an assumption against 1.24% of pretraining. This is the largest
  pretraining-to-RL shift I measured.

### B. Logic ladder on theorems

- **Definition.** Label each theorem minimal, intuitionistic-only or classical-only. Sources: Johansson 1937, Glivenko
  1929, Statman 1979 (all cited). Minimal logic plus DNE is full classical logic (Diener and McKubre-Jordens 2020,
  cited), so in our rule set BOTE is never needed once DN is present.
- **Complete and unique.** Yes, on theorems. Three nested classes, one label each.
- **Choice or forced.** Forced by the theorem, which is the point: it measures need, not use.
- **Detect and remove.** Run G4ip on the statement. Remove by theorem label, not by rule token. That gives a 2 x 2
  design: token seen or not, needed use seen or not.
- **Need test.** Decidable, PSPACE in general, 2 seconds for our 18,439 targets.
- **First-hit cost.** Classical: about three joint decisions (assume the negated goal, reach ⊥, apply DN). Ex falso:
  one.
- **Finer layers** (weak excluded middle, Jankov 1968 cited; Gödel-Dummett, not checked) label theorems but do not
  map to any proof pattern in our rule set. Not usable for knockouts.

### C. Argument boxes (the nested-implication choice point)

- **Definition.** A box built because a hypothesis needs an implication or negation as its argument, not because the
  goal is one. In Fitch: an IMPI/NEGI line that reaches the minor premise of IMPE or NEGE. In lambda terms: a λ in
  argument position. In Prawitz/Troelstra-Schwichtenberg terms: an introduction on a track of order 1 or more (the
  book's exact wording was not seen). Formulas of order 2 or less are in P, order 3 is already PSPACE-complete
  (Schubert, Urzyczyn and Zdanowski 2024, cited).
- **Complete and unique.** One binary flag, or an integer (maximum track order). Not a partition.
- **Choice or forced.** A real choice: this is the non-invertible left rule.
- **Detect and remove.** `flags.py` field `argbox`. Removal costs 0.93% of pretraining.
- **Need test.** Not built. See the G4ip caveat above.
- **First-hit cost.** About two: state a `have` whose formula is copied from a hypothesis's antecedent, not from the
  goal, then apply the hypothesis to it.
- **Link to earlier work.** This is the "non-goal boxes" shape that the long-RL note lists as unsolved.

### D. Normality: detours and analytic proofs

- **Definition.** A detour is an intro-derived line used as the major premise of the matching elim. A non-analytic
  line is one whose formula is not a subformula of the statement (classical proofs: up to negation). Prawitz 1965,
  cited; Stålmarck 1991, cited.
- **Complete and unique.** Binary flag per proof. Every proof has a normal form.
- **Choice or forced.** Never forced. Purely optional.
- **Detect and remove.** `flags.py` fields `detour`, `nonanalytic`. Both are already near zero in pretraining (0.17%).
- **Need test.** Trivially "never needed".
- **First-hit cost.** One for a detour (apply a line that an intro produced). High for a non-analytic lemma: the model
  must write a formula it cannot copy from the prompt.
- **Free readout.** The control run found detours in 0.09% of accepted proofs, below the pretraining rate. A cheap
  pattern with no need behind it did not grow.

### E. Shape of classical proofs

- **Definition.** For each DN: is the double negation built by NEGI (reductio) or not (plain DNE)? How many times is
  the reductio assumption used? Seldin 1989 for "one classical step at the end" is only confirmed second-hand.
- **Status.** Dan's create-or-elicit knockout is the first split. The use count is my own idea; the citation check
  found no result relating it to provability. The control run writes DN almost only as plain DNE (7.42% against
  0.26% reductio), the reverse of the pretraining mix.

## 3. Directions that do not give a usable split

- **Forward against backward reasoning.** A finished proof does not record search direction, only a line order that
  the generator fixed. Chaudhuri, Pfenning and Price (2008, cited) show the difference is a polarity choice in the
  search calculus, not a property of the proof.
- **Proof nets and canonical forms.** They answer "are these two proofs the same", not "what strategy is this". No
  agreed notion exists for classical propositional logic (Straßburger 2006, cited). Useful only for deduplicating
  accepted proofs (Scherer 2017, cited).
- **Proof-complexity separations.** Cut against cut-free and tree against dag gaps are asymptotic. At 80 lines or
  fewer only raw measures remain: size, box depth, sharing.
- **Combinator bases taken literally.** Nobody writes S/K terms. The usable residue is candidate A.
- **Intermediate logics.** See B.
- **Focusing as a label set.** It gives the forced/chosen split in section 0 and candidate C, but its own labels
  (synthetic rules) are an unbounded family, not a short list.

## 4. What I would run first

Candidate A, the assumption-contraction knockout. It removes 1.24% of pretraining, so the arm is size-matched for
free. All tokens and all syntax stay seen. First-hit cost is one. 83% of RL references need it. If the first-hit
hypothesis is right it should appear within the first rounds; if it does not, "never seen means never appears" holds
even at cost one, which would be a much stronger claim than the current data supports.

Before launch: fix the ORE-branch overcount, and scan the existing knockout runs' `found_stream.jsonl` with
`flags.py`, which costs nothing.

## 5. Citation status

Checked by a search agent; full statuses were returned to me, the main points are:

- Confirmed: Prawitz 1965; Andreoli 1992; Liang and Miller 2009; Dyckhoff 1992; Statman 1979; Johansson 1937;
  Glivenko 1929; Stålmarck 1991; Ariola and Herbelin 2003; Diener and McKubre-Jordens 2020; Griffin 1990; Parigot
  1992; Chaudhuri, Pfenning and Price 2008; Scherer and Rémy 2015; Scherer 2017; Jankov 1968; Schubert, Urzyczyn and
  Zdanowski 2024 (arXiv:2405.05670).
- Partly confirmed: Seldin 1989 (the "once, at the end" claim is second-hand); Troelstra and Schwichtenberg 2000
  (track order, from course notes); Hindley 1997; Chaudhuri, Miller and Saurin 2008 (which fragment matches proof
  nets); Rips 1994 (forward/backward rule lists confirmed only for his 1983 model).
- Not checked: Church 1941 and 1951; Anderson and Belnap 1975; Girard 1987; Dummett 1959; Nishimura 1960; Urquhart
  1984; Sørensen and Urzyczyn 2006.
- Correction: the "decidability of variables" note is Katsumasa Ishii, MLQ 2018; the underlying theorem is Ishihara,
  LMCS 2014.
