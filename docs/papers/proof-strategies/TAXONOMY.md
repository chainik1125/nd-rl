---
written_on: 2026-10-18
written_by: agent:claude
---

> **Reading copy (librarian, 2026-10-19).** This is `proof_strategies/TAXONOMY.md` of run `proof-strategies` (fork
> `dan-pandori/nd-takehome`, branch `dan_proof-strategies`, DONE `441de472`) with the corrections from its review
> applied, each marked *[review]*. In eight places a rule schema or law formula, read with its letters as atoms, is in
> a textbook72 renaming class; these are restated in words and marked *[librarian]*. File names in `code` refer to the
> fork. See [README.md](README.md).

# A taxonomy of the proof strategies in our pretraining data

Run `proof-strategies` (executor, 2026-10-18; pre-registration `preregistration/proof-strategies.md`). Taxonomy **v1.2**
(v0 = the pre-registered list, code at commit 82fa2c47; v1.1 = after reading the data; v1.2 = after the blind spot-check:
disjunctive syllogism and excluded middle tightened, wording clarified; changes and reasons in `log.md`, 2026-10-18,
except the last disjunctive-syllogism change, K12 47 → 43 records, commit e0511ae6, which is not logged *[review]*). Code:
`proof_strategies/strategies.py` (labels), `theorem_level.py`, `necessity_ps.py`; numbers: `TABLES.md`; examples:
`EXAMPLES.md`. Checker: Lean alone (every proof shown or searched here was accepted by Lean; nothing is judged by
`nd_verify`).

**Thirteen strategies** (ten intuitionistic, three classical) plus "trivial", each with sub-types. Every proof gets two
kinds of label: **uses** (each strategy that occurs anywhere in it) and one **primary** (how its conclusion is reached).
The table gives the share of proofs that use each strategy in K12 (155,000 generator proofs) and in Charles's 187
textbook problems (reference proofs); in parentheses, the share where the step does logical work (§ 2). All shares,
comparison sets and breakdowns: `TABLES.md`.

| # | strategy (id) | standard names | classical? | K12 % (working) | textbooks % | the knockout's σ |
|---|---|---|---|---:|---:|---|
| 1 | Direct reasoning (`direct`) | forward chaining; modus ponens / →E, simplification / ∧E | no | 51.2 | 84.5 | – |
| 2 | Conjunction introduction (`and_intro`) | ∧I, adjunction, "prove each conjunct" | no | 32.6 | 47.6 | – |
| 3 | Conditional proof (`cond_proof`) | →I, conditional proof (CP), deduction theorem | no | 42.9 (20.8) | 52.4 (49.2) | – |
| 4 | ∨-introduction (`or_intro`) | ∨I, addition | no | 44.3 | 31.0 | – |
| 5 | Proof by cases (`cases`) | ∨E, argument by cases | no | 44.2 (8.8) | 33.7 (33.7) | – |
| 6 | Disjunctive syllogism (`disj_syll`) | DS, modus tollendo ponens | no | 0.03 | 14.4 | – |
| 7 | Proof of a negation (`neg_intro`) | ¬I; "proof of negation"; intuitionistic reductio | no | 4.7 (2.1) | 38.0 (38.0) | – |
| 8 | Modus tollens / contraposition (`mod_tollens`) | MT (modus tollendo tollens); contraposition | no (a classical variant exists) | 0.05 | 11.8 | – |
| 9 | Ex falso (`ex_falso`) | explosion, ex falso quodlibet, ⊥E | no | 8.0 | 24.1 | – |
| 10 | **Proof by contradiction** (`contradiction`) | classical reductio ad absurdum (RAA), indirect proof (IP), ⊥c | **yes** | 10.8 (0.04) | 20.9 (20.9) | **σ = #10 (via ¬I); σ⁺ = #10 ∪ part of #11** |
| 11 | Double-negation elimination (`dne`) | DNE on a double negation the theorem supplies | **yes** | 3.1 | 3.2 | part of σ⁺ |
| 12 | Excluded middle (`lem`) | LEM, tertium non datur; cases on A ∨ ¬A | **yes** | 0.03 | 0 | – |
| 13 | Lemma reuse (`lemma_reuse`) | a lemma used twice (a shared derived line) | no | 11.7 | 16.0 | – |
| – | Trivial | the conclusion is a premise | no | 0 (primary) | 0 | – |

References for the names: Gentzen (1935) and Prawitz (1965) for the introduction / elimination rules, ⊥ and the
classical absurdity rule; Fitch (1952) for subproofs and reiteration; *forall x: Calgary* (Magnus, Button, Trueman and
Zach, 2023 ed.; quick reference "Basic deduction rules for TFL" and "Derived rules for TFL"), whose basic rules are R,
∧I/E, →I/E, ∨I/E, ¬I/E, X (explosion) and IP (indirect proof) and whose derived rules are DS, MT, DNE, LEM (as a
cases rule, "TND") and De Morgan; Copi, Cohen and McMahon, *Introduction to Logic* (14th ed., 2014) for MP, MT, DS,
Simplification, Conjunction, Addition, constructive dilemma, hypothetical syllogism, conditional and indirect proof;
Velleman, *How to Prove It* (3rd ed., 2019, ch. 3) for goal-directed strategies; Bauer, "Proof of negation and proof by
contradiction" (*Mathematics and Computation*, 2010) for the line between #7 and #10; van Dalen, *Logic and Structure*
(5th ed., 2013) for RAA; Dyckhoff (1992) for the G4ip decision procedure used at theorem level.

## 1. The setting

Proofs are in the take-home's Fitch-style natural deduction (`spec.md`): numbered lines, each with a box depth, a formula,
a rule and cited lines. Rules: PR (premise), AS (assumption, opens a box), R (reiteration), ANDI, ANDE1/2, IMPE, IMPI,
ORI1/2, ORE, NEGE, NEGI, BOTE (ex falso), DN (classical double-negation elimination; the only classical rule). IMPI and
NEGI cite one box (s, e) by its assumption line s and last line e; ORE cites (j, s1, e1, s2, e2): the disjunction j
and two boxes. F is ⊥. In Lean, ¬A is A → False, so NEGI is a `fun`, NEGE an application, BOTE `False.elim`, and DN
`Classical.byContradiction`.

Every predicate is evaluated on the **dependency-pruned** proof: the premises plus every line the last line transitively
cites (a box citation cites s and e). Notation used below:

- **origin(i)**: follow R lines from line i back to the line that produced the formula.
- **hyps(i)**: the open assumptions line i depends on. AS line: {i}. IMPI / NEGI citing (s, e): hyps(e) − {s}. ORE:
  hyps(j) ∪ (hyps(e1) − {s1}) ∪ (hyps(e2) − {s2}). Any other line: the union over the lines it cites.
- **deps(i)**: every line i transitively cites.
- **citers(x)**: the non-R lines y such that x = origin(r) for some line r that y cites.
- **root**: origin(last line). **Sub**: all subformulas of the theorem's premises and conclusion.
- "x **depends on** assumption s" means s ∈ hyps(x); "x depends on line y" means y ∈ deps(x).
- A line x is **inside** box (s, e) iff s ≤ x ≤ e.

A proof **uses** a strategy if at least one line of the pruned proof instantiates it ("instances" below). Strategies
nest: a disjunctive syllogism is also a proof by cases and uses ex falso; a modus tollens is also a proof of a negation
(or, classically, sits inside a proof by contradiction); "uses" counts are therefore not additive.

## 2. Definitions (instances, then sub-types)

**#1 Direct reasoning.** Instances: every IMPE line (modus ponens) and every ANDE1/2 line (∧-elimination); also a root
NEGE line (the goal is ⊥, obtained from A and ¬A). Sub-types: `mp`, `ande`, `mp_chain3` (an IMPE citing an IMPE citing
an IMPE, as `patterns2.impe_chain` ≥ 3), `neg_elim_root`. A proof with no AS line at all is **box-free** (reported
separately).

**#2 Conjunction introduction.** Instances: ANDI lines. Sub-type `root`.

**#3 Conditional proof.** Instances: IMPI lines citing (s, e). Sub-types: `vacuous` if s ≠ e and s ∉ hyps(e) (the
assumption is never used: weakening, a conditional whose consequent holds without its antecedent *[librarian]*), else `working` (a one-line box, s = e, the identity A → A, counts as
using its assumption); `explosive` if origin(e) is a BOTE line inside the box
that depends on s (¬A ⊢ A → B); `curried` if origin(e) is an IMPI line inside the box (A → (B → C)); `hyp_syll` if
origin(e) is an IMPE line inside the box reached from s by ≥ 2 steps of modus ponens (hypothetical syllogism, chaining
conditionals *[librarian]*; the steps
are those of the refutation route below, all of kind `mp`).

**#4 ∨-introduction.** Instances: ORI1/2 lines. Sub-types (a line may have several): `root`; `case_branch` (it is
origin(e_k) of a branch of some ORE and lies inside that branch); `feeds_ore` (it is origin(j) of some ORE); `other`
(none of the three, e.g. the last line of a →I box or a conjunct of an ∧I).

**#5 Proof by cases.** Instances: ORE lines (j, s1, e1, s2, e2). Sub-types: **`genuine`** if every branch uses its case
(s_k = e_k or s_k ∈ hyps(e_k)), else **`redundant`** (a branch proves the goal without its case, so the split can be
deleted); `on_premise` / `on_hypothesis` / `on_derived` by the rule of origin(j) (PR / AS / other); `nested` (the ORE line
lies inside a branch of another ORE); `constructive_dilemma` (each branch ends, inside the branch, in an ORI of an IMPE
whose minor premise is the branch assumption: the textbook constructive dilemma, one conditional applied to each
disjunct *[librarian]*); `trivial_merge` (the disjunction is X ∨ X);
`or_intro_decoration` (origin(j) is an ORI line; `patterns2.ori_ore`); `all_refuted` (both branches refuted, below);
`bot_case` (a case whose assumption is ⊥ itself: an empty case, never "refuted"); `on_lem` (origin(j) is an
excluded-middle line, #12).

**#6 Disjunctive syllogism.** Instances: ORE lines with exactly one **refuted** branch whose other branch uses its case
(s = e or s ∈ hyps(e)) and is not itself a ⊥ case. Branch (s, e) is refuted iff its assumption s is not ⊥ itself and origin(e) is a BOTE line
inside the branch whose cited ⊥ depends on s (the case leads to a contradiction). From a disjunction and a refutation
of one disjunct, conclude the other *[librarian]*. (v1.1 also counted a
case that is ⊥ itself, A ∨ ⊥ ⊢ A, and splits whose other branch ignored its case; the spot-check found these were most
of the K12 instances.) Explosion done another way (a vacuous ¬I or reductio closing the case) is not counted here.

**The refutation route** of a NEGI line n citing (s, e) (used by #7, #8 and #10): how the box refutes its assumption s.
- `vacuous` if s ≠ e and s ∉ hyps(e) (⊥ without using the assumption). Otherwise let f = origin(e); if f is not inside
  the box, `other`.
- If f is NEGE citing (a, b): if both a and b depend on s, `self`; otherwise trace the side that depends on s.
- If f is IMPE citing (m, k) (⊥ from a given X → ⊥): if k depends on s and m does not, trace k; else `mixed`.
- If f is ORE, `cases` (⊥ by cases). Otherwise `other`.
- **Trace** from x = origin(side) back to s, recording one step per line: IMPE whose minor premise alone depends on s →
  `mp`, continue at origin(minor); IMPE whose major premise alone depends on s → `mp_major`, continue at origin(major);
  ANDE → `ande`; ORI → `ori`; ANDI with exactly one dependent side → `andi`, continue there; DN → `dn`; stop at s. Any
  other case (both or neither premise dependent, another rule) ends the trace with `mixed`.
- Route from the steps, first match: no steps → **`immediate`** (the assumption itself contradicts a line); any `mixed`
  → `mixed`; last step (the one applied to the assumption) `mp_major` → `imp_hyp` (refuting a conditional); steps ⊆
  {mp, ande} with at least one `mp` → **`mt`** (consequences of the assumption by modus ponens); only `ande` →
  `and_elim`; only `ori` → `or_weaken` (De Morgan-style, ¬(A ∨ B) ⊢ ¬A); only `andi` → `and_intro`; else `mixed`.

**#7 Proof of a negation.** Instances: NEGI lines that are the root or have at least one citer that is not a DN line
(a NEGI line cited only by DN lines belongs to #10). Sub-types: the refutation route (`immediate`, `mt`, `or_weaken`,
`and_elim`, `and_intro`, `imp_hyp`, `cases`, `self`, `mixed`, `vacuous`, `other`), and `working` = route ≠ `vacuous`.

**#8 Modus tollens / contraposition.** Instances: (a) any NEGI line (a #7 instance or one consumed by DN) whose
refutation route is `mt`; sub-types `strict` (one `mp` step, the dependent side is the positive premise of the ⊥ line:
NEGE's first premise, or the minor premise of an IMPE from a given X → ⊥; A → B, ¬B ⊢ ¬A), `neg_side` (one `mp`, the
dependent side is NEGE's negative premise: A → ¬B, B ⊢ ¬A), `chain` (≥ 2 steps), `classical` (the NEGI is cited
by DN: ¬A → B, ¬B ⊢ A; it can hold together with `strict`, `neg_side` or `chain`); (b) an IMPI line citing (s, e) whose assumption is a negation ¬B and whose origin(e) is a NEGI
line inside the box with route `mt` whose independent side has origin s: A → B ⊢ ¬B → ¬A (`contraposition`).

**#9 Ex falso.** Instances: BOTE lines (explosion done by a vacuous ¬I or a vacuous reductio is counted under #7 /
#10 as `vacuous`, not here). Sub-types by where the ⊥ comes from (several may hold): `inconsistent_premises` (the
cited ⊥ line depends on no assumption: it follows from the premises alone, which includes a premise with ⊥ as a
conjunct), `refuted_case` (the BOTE is the last line, after R, of a refuted ∨E branch), `explosive_cond` (it is the
last line, after R, of a →I box and its ⊥ depends on that box's assumption), `other` (none of the three); and `root`.

**#10 Proof by contradiction (classical).** Instances: DN lines d (conclusion G) whose cited line c = origin(cited) is
either a NEGI line (**σ**: assume ¬G, derive ⊥, conclude ¬¬G, eliminate it) or a line derived by another rule (not PR,
AS, NEGI) whose ¬¬G is not in Sub (**`built_other`**: a disguised reductio, e.g. ¬¬G built in both branches of a ∨E).
Its route is the refutation route of c (σ); for `built_other` it is `vacuous` when c is BOTE (going from ⊥ to ¬¬G and then to G is explosion *[librarian]*),
and when c is an ORE whose two branches each build ¬¬G vacuously (by BOTE or a vacuous ¬I, recursively), `cases` for
any other ORE, `other` otherwise.
One **kind** per instance, first match: `vacuous` (route vacuous: ⊥ without ¬G, explosion in disguise) > `pointless`
(route immediate and the independent premise of the clash has formula G: G was already derived) > `scaffolded` (¬¬G or ¬G → ⊥ is in Sub:
the theorem supplies the double negation) > `neg_goal` (G is a negation: reducing a triple negation to a single one is intuitionistic *[librarian]*) > **`free`**.
An instance is a **template** if its kind is vacuous or pointless, or scaffolded with route immediate (the reductio does
no logical work); otherwise it is **working**. Position sub-types: `root` (the DN line is the root), `lemma` (it is
not); `via_r` (DN cites an R copy).

**#11 Double-negation elimination.** Instances: DN lines that are not #10: c = origin(cited) is a premise (`on_premise`),
an assumption (`on_hypothesis`, e.g. proving double-negation elimination as a conditional *[librarian]*), or a derived line whose ¬¬G is in Sub (`on_derived`).

**#12 Excluded middle.** Instances: DN lines whose formula is X ∨ ¬X or ¬X ∨ X: excluded middle proved by
contradiction, the classical lemma. (v1.1 also counted any X ∨ ¬X line depending on some DN step, e.g. ∨I of an X got by
DNE; the spot-check showed that is not the strategy.) Sub-type `cases_on_lem`: an ORE whose disjunction (after R) is
such a line, *forall x*'s TND rule; only this ORE counts as an #12 instance for the primary label. A premise X ∨ ¬X is
supplied, not classical reasoning: a case split on it is plain #5.

**#13 Lemma reuse.** Instances: lines (not PR, AS, R) with ≥ 2 distinct citing lines (one line citing it twice, as in
∧I of X with itself, is one citer). Sub-types: `across_boxes` (the citers lie in ≥ 2 different innermost boxes, the top
level counting as one box; e.g. a lemma proved before a case split and used in both cases), `multi_step` (the
reused line depends on at least one line that is not PR, AS or R).

**Trivial.** The root is a PR line (the conclusion is a premise). Never a "uses" label.

## 3. The primary label and its precedence rule

The primary strategy is that of the **root inference**: among the strategies the root line itself instantiates (its own
instance of #1–#12, not those of the lines it cites), take the first in

`excluded middle (cases on LEM) > proof by contradiction > DNE > disjunctive syllogism > modus tollens / contraposition >
proof by cases > proof of a negation > conditional proof > ex falso > ∨-introduction > ∧-introduction > direct
reasoning`;

a PR root is "trivial"; anything else would be "unlabelled" (none occurs). Lemma reuse is never primary. Since the root
line has one rule, precedence only decides between a basic rule and the named pattern built on it (a ∨E that is a
disjunctive syllogism, a ¬I that is a modus tollens, a →I that is a contraposition, a DN that is a reductio).

**The primary label is this root label.** A second label is reported beside it: the **curried primary**. Goals shaped
A → (B → C) make "conditional proof" the primary of most proofs whatever they do inside. The curried primary first descends through root →I boxes (r := origin(e) while r is IMPI; their assumptions
become premises, as in Charles's overlap rule that A ⊢ B and ⊢ A → B are one problem) and then applies the same rule
(a NEGE line reached this way is the curried proof's root ¬E, so direct reasoning, #1); if the descent reaches an
assumption or premise, the curried proof is trivial.

## 4. Theorem-level labels (`theorem_level.py`, `necessity_ps.py`)

A proof shows how the generator proved a theorem; these say what the theorem forces.
- **classical-only**: provable classically (a proof exists) but not intuitionistically (G4ip, `intuit.py`); every proof
  then uses DN, i.e. #10, #11 or #12.
- **dneg**: the sequent contains a double negation, semantically (¬¬X, ¬X → ⊥, ¬(X → ⊥) or (X → ⊥) → ⊥).
- **σ-covered**: classical-only and no dneg, with dneg semantic as above *[review]*. This is not create-or-elicit's
  definition, which is also the pre-registered one: create-or-elicit counts only a literal ¬¬ subformula. Under that
  rule K12 has 2,340 σ-covered theorems (1.51 %), all with a (¬X → ⊥) subformula; cap-6 has 2,235 and the dose set 232.
  The RL pools, Charles's problems and textbook72 agree under both rules. The switch is not logged in `log.md` *[review]*.
  Every proof must then eliminate a double negation it built itself. That need not be a #10 *[review]*: at least 573
  Lean-accepted RL proofs of σ-covered targets (create-or-elicit ladders, rounds 1–16) eliminate a double negation they
  built themselves by applying DN to a ¬¬X that a box assumes: an ∨E case, or the assumption of a →I lemma that states
  double-negation elimination (create-or-elicit's "lemma route"). The classifier labels these #11 `on_hypothesis`, with no #10. The review's fix:
  extend #10 to DN on any box hypothesis ¬¬X that is not in Sub, when the proof builds ¬¬X by ¬I *[review]*.
- **inconsistent premises**: no truth assignment satisfies all premises (so ex falso proves the conclusion).
- **requires X** (bounded, on samples): the shortest-proof search of `minlen.py` finds a proof of ≤ 12 lines, and the
  same search with rule X banned finds none within 12 lines and does not time out (60 s). Bans: ORE (#5), NEGI (#7, #8
  and #10 all need it), BOTE (#9), DN (all classical strategies). Relative to minlen's restricted formula space.

## 5. How this maps to σ, `patterns.py` and `patterns2.py`

| earlier label | definition there | here | K12 records (both) |
|---|---|---|---|
| σ (`coe_sigma`: strict ∪ chain, written or pruned) | DN citing (after R) a NEGI line | #10 instances whose c is NEGI | 16,703 = 16,703 |
| `patterns.reductio` (P2, σ strict) | DN citing a NEGI line directly | same, no R | 16,703 |
| σ⁺ (`coe_sigma.dn_derived`, the knockout's removal rule) | DN citing a line that is not PR / AS | #10 ∪ #11 `on_derived` | 16,961 = 16,961 (0 differ either way) |
| `coe_rounds.sigma_subtype` scaffold / free | the prompt contains ~~G or ~G > F | #10 kinds `scaffolded` / `free` (+ `vacuous`, `pointless`, `neg_goal` split out) | – |
| σ-covered (create-or-elicit) | classical-only, no ¬¬ in the sequent | theorem-level σ-covered (semantic dneg) | 0 here; 2,340 under the definition in column 2 *[review]* |
| `patterns.derived_ore` (P1) / `derived_ore_strict` | ORE whose disjunction line is not PR (strict: not PR / AS, X ≠ Y) | #5 `on_derived` (origin after R) | in `TABLES.md` |
| `patterns2.nested_ore`, `ori_ore`, `impe_chain4`, `negi_ande_hyp` | ORE in an ORE branch; ORE on an ORI line; IMPE chain ≥ 4; ANDE on a NEGI box's assumption | #5 `nested`, #5 `or_intro_decoration`, #1 `mp_chain3` (≥ 3), #7 `and_elim` (narrower: the refuting path is ∧E only) | in `TABLES.md` |

## 6. Validation

- **Lean.** Every proof this run searched was checked by Lean (`nd2lean` + Lean core): 1,610 textbook-schema, 58
  textbook72 and 990 K12 shortest proofs, and 3,610 necessity-search proofs; 6,268 in all, 0 rejected. Harness
  controls: 200 / 200 accepted as found, 0 / 200 with the conclusion negated, 0 / 200 with the last line mis-cited
  (`artifacts/ps/lean_negative_control.txt`). Rendered for reading and checked by `lean_check` (allowlist, axioms, no
  errors): the 30 canonical examples, a 587-proof renderer test, the 12 snippets in `blog.md`. The 39 unit tests and 25
  fixtures below are Lean-checked proofs.
- **Term size.** The per-rule formula equals `lean_check`'s elaborated inference-node count on 400 / 400 pruned proofs
  (`artifacts/ps/term_size_check.json`).
- **Blind spot-check** (three independent subagents, by hand from this document; 260 proofs, 20 per strategy: 12
  random positives, 4 random negatives, 4 borderline; seed 20261018; `data/ps/validation/`). Agreement with the v1.1
  labels: uses 260 / 260, primary 260 / 260 (20 / 20 for every strategy; `artifacts/ps/validation/spot_compare_v1.1.json`).
  Their notes produced v1.2: 10 of the 12 sampled disjunctive-syllogism positives had a ⊥ case or an idle other case,
  and 5 of the 20 excluded-middle items were ∨I of a DNE'd X or rested on a vacuous reductio. Against the final v1.2
  definitions the same blind labels agree on 245 / 260 uses and 258 / 260 primary: disjunctive syllogism 10 / 20 and
  excluded middle 15 / 20 on uses (`spot_compare_v1.2.json`). The final #6 and #12 were never hand-checked blind; only
  the second implementation validates them *[review]*.
- **Second implementation** (an independent subagent; own parser; written from this document and `spec.md` only;
  counts frozen before it read our labels; `proof_strategies/validation/second_impl.py`). It covers proof by
  contradiction (instances, kinds, routes, working / template), DNE, proof by cases (genuine / redundant / `bot_case`),
  disjunctive syllogism, proof of a negation (routes), modus tollens (sub-types), excluded middle and the primary label.
  - v1.1: every uses label and the primary agreed on 155,000 / 155,000 K12 and 1,610 / 1,610 textbook-schema proofs.
    20 K12 records differed in a sub-type: two gaps in the text and one id missing from our output, all fixed.
  - v1.2 (final): every compared label agrees on every record of both sets.
  - Its 25 hand-built fixtures cover sub-types the data never exercise (`cases_on_lem`, `all_refuted`, `via_r`,
    `bot_case`, …): 25 / 25 are Lean-accepted and get exactly the labels it expected. (`bot_case` is listed wrongly
    here: 2,070 K12 records have it *[review]*.)
  - Reports: `artifacts/ps/validation/second_impl.md` and `second_impl_v1.2.md`.
- **Coverage.** Every K12 proof has a primary label (0 unlabelled) and at least one "uses" label. Under v0, 277 K12
  proofs had none: each proved ⊥ by one ¬E from contradictory premises, and v1 counts a root ¬E as direct reasoning.
  The curried primary is "trivial" for 3.26 %: the curried goal is an assumption or premise (a consequent that is a
  premise, or the identity conditional *[librarian]*).
- **Not exercised by the data.** No K12 or schema proof has a dead line, so pruning is tested only by unit tests.
  Sub-types with no data instances are tested only by the fixtures.

## 7. Limits

- Labels describe proofs as written by the generator (K12, cap-6, K12⁻, pool generator proofs), by a shortest-proof
  search (textbook-schema instances, textbook72, a K12 sample), or by Charles's search (his reference proofs). Style
  differs between these sources; the K12 sample re-proved by the same search separates theorem from style.
- The definitions are syntactic. "Working" and "template" are about whether a step does logical work in this proof,
  not about whether a cleverer proof exists; the theorem-level labels answer that, within bounds.
- "Requires" is bounded (12 lines, minlen's formula space); classical-only is exact (G4ip).
