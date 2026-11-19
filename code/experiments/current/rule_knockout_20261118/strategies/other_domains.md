---
written_on: 2026-11-18
written_by: agent:claude
---

# Proof strategies outside natural deduction: which system splits cleanly

Scope: systems other than Fitch natural deduction where "strategy" is crisp, judged against the group's design
(remove one strategy from pretraining, run RL, see whether it comes back). Three sibling notes cover decompositions
inside natural deduction, corpus motifs and the earlier taxonomies; this note does not.

Status labels used below: **[checked]** = I ran it today (Lean 4.34 core or a script, files listed at the end);
**[cited]** = a source found by web search today; **[memory]** = standard result I did not find a source for today;
**[inferred]** = my reasoning, not checked.

## 1. What the requirement means

A system is useful for the knockout design if it has all of:

1. **Complete decomposition.** Every proof step carries exactly one strategy label.
2. **Independence.** No strategy can be derived from the others. Then a knockout cannot be simulated, which is what
   fails for ex falso today (reductio plus double-negation elimination replaces it).
3. **Decidable need.** "Theorem T cannot be proved without X" is cheap to decide. Then the readout can be "the model
   solved a target that provably needs X", which is stronger than "X appears in an accepted proof".
4. **Real choice.** The goal's shape does not dictate the rule.
5. **Tunable first-hit cost.**

The definition of need that works everywhere: **T needs X iff T is not a theorem of the system with X deleted.**
The set of needed axioms N(T) is then unique for every theorem, by definition. What is not guaranteed is that N(T)
alone suffices: a theorem can be provable from {X, Y} and from {X, Z} and need neither Y nor Z. Those theorems are
detectable and rare in both censuses below (0.3% of identities, 0.7% of theorems), and are dropped.

One point that applies to every candidate. Independence is exactly what makes a **named** strategy expensive to
rediscover. If strategy X is carried by its own token and X is independent, a base model that never saw X has never
emitted that token. Today's two token knockouts (`.elim`, `Classical.byContradiction`) had no hit in more than 440,000
accepted attempts. So each candidate is assessed in two encodings:

- **named:** the strategy has a fixed token (`axW`, `comm`). First hit = emit a never-trained token, then instantiate it.
- **in-prompt:** the strategy is a hypothesis in the theorem statement, cited as `hK` with K varying per problem, as
  premises are today. No token is new. The base has never seen a hypothesis of that shape, or has never used one.

The opposite risk for the in-prompt encoding: applying a hypothesis is one generic skill, so an unseen axiom may work
zero-shot. Then the knockout removed nothing. PrOntoQA-OOD (Saparov et al., 2023) reports that LLMs generalise to
unseen compositions of deduction rules but need demonstrations for proof by cases and by contradiction [cited]; INT
(Wu et al., 2021) reports that a transformer trained on 25 axiom combinations proves 29% of theorems with unseen
combinations and 91% when trained on 300 [cited]. Both suggest the in-prompt encoding sits in a measurable middle,
not at 0 or 1. Which regime a 9.6M model lands in is the first thing to measure (section 5).

## 2. Ranked candidates

### 2.1 Equational reasoning in a one-operation algebra (rank 1)

**System.** A type with one binary operation (later a unit and an inverse). Strategies are the axioms: associativity
A, commutativity C, idempotence I, then unit U and inverse N. A proof is a chain of rewrites, each step one axiom at
one position in one direction. Every subset of the axioms is a named variety: magma, semigroup, commutative
semigroup, band, semilattice, monoid, group, abelian group.

1. **Strategies and uniqueness.** 3 to 6. Each proof step has exactly one label, so every proof decomposes completely
   and uniquely into a sequence of axiom uses. Theorem-level need is decidable: T needs X iff the identity fails in
   some algebra satisfying the other axioms. Prototype **[checked]**: with all magmas of size at most 3 as
   countermodels (18, 28 and 58 magmas for A, C, I), 20,000 random semilattice identities over 3 variables were
   labelled in 4.5 s. All seven non-empty need sets occur: A 188, C 750, I 1,941, AC 1,037, AI 1,895, CI 1,088, ACI
   13,045, and 56 with no certified need. The small-model test agreed with exact normal forms on 20,000 of 20,000
   identities for I and 19,983 of 20,000 for A. The Equational Theories Project (Tao et al., 2025) reports the same
   thing at scale: finite magmas of size up to 4 refute more than 96% of the false implications among 4,694 laws
   [cited].
2. **Independence.** Yes, and it is checked by the same finite models: for each axiom there is a small magma
   satisfying the others and violating it **[checked for A, C, I]**. Adding absorption or a second operation needs
   care: in lattices idempotence follows from the two absorption laws [memory], so lattices are not an independent
   basis as usually written.
3. **Real choice.** Yes. Any axiom applies at any matching subterm in either direction; the goal is an equation
   whose shape names no axiom.
4. **Generation and checking.** Generate by random walk: random term, apply random axiom instances, keep (start, end,
   path), shorten the path by breadth-first search in the term graph. Label by countermodels and normal forms. Lean 4
   core checks it with `rw` and no Mathlib **[checked]**:

   ```lean
   theorem e1 ( α : Type ) ( op : α → α → α ) ( a b c : α )
       ( assoc : ∀ x y z , op ( op x y ) z = op x ( op y z ) ) ( comm : ∀ x y , op x y = op y x ) ( idem : ∀ x , op x x = x ) :
       op a ( op b a ) = op a b := by have n1 : op a ( op b a ) = op a ( op a b ) := ( by rw [ comm b a ] ) ; have n2 : op a ( op a b ) = op ( op a a ) b := ( by rw [ assoc a a b ] ) ; have n3 : op ( op a a ) b = op a b := ( by rw [ idem a ] ) ; exact n1.trans ( n2.trans n3 )
   -- terse form, rule instances only:   := by rw [ comm b a , ← assoc a a b , idem a ]
   ```

   The nested `by` needs its parentheses in the one-line format; without them the following `have` is parsed into the
   inner block and Lean rejects the proof **[checked]**. The vocabulary must not contain `simp`, `ac_rfl`, `omega`,
   `grind` or `decide`, each of which would bypass the axioms; the group's closed tokenizer already guarantees that.
5. **First-hit cost.** Named encoding: one unseen token plus two or three argument tokens, and in the long form the
   next term, which is a copy with one local edit. In-prompt encoding: no unseen token; one unseen combination (cite
   a premise of unfamiliar shape inside `rw`). Tunable three ways: the encoding, the minimum number of applications of
   X a target needs (computable by search at these sizes), and whether the model writes intermediate terms.
6. **Transfer.** "Apply identity X here" is the unit operation of LM mathematics, and "RL makes the model start using
   a lemma it has never used" is a fair small copy of the LM question. It says less about structural strategies such
   as case analysis.

**Weakness.** Strategy = axiom is the same granularity as rule = strategy in natural deduction. What changes is that
the three defects go away: rules are not forced by the goal, are not nested, and need is decidable. Commutativity and
associativity proofs can be long (sorting a term), so length has to be capped in generation.

### 2.2 Structural rules in implicational logic: BCI, BCK, BCIW, BCKW, plus double negation (rank 2)

**System.** Formulas from → and an uninterpreted falsum F. Strategies are **weakening** K (discard a hypothesis),
**contraction** W (use a hypothesis twice) and **classical** DNE (((a → F) → F) → a). The base, with none of them,
is linear logic's implicational fragment (combinators B, C, I). This is the rigorous version of "restrict to negation
and implication". Under Curry–Howard the combinator bases match variable-use disciplines on λ-terms: BCI terms use
each variable exactly once, BCK at most once, BCIW (relevant, λI) at least once, BCKW is unrestricted [cited: nLab
"combinatory logic", accessed 2026-11-18; Hindley 1993].

1. **Strategies and uniqueness.** Three independent flags, 8 logics. Flags of a proof are computed from the text: a
   bound name never cited (K), a name cited twice (W), `dn` used (DNE). Theorem-level need is decidable: affine and
   linear provability by terminating search; intuitionistic by loop-checked search (PSPACE-complete in general,
   Statman 1979 [cited]); relevant implication is decidable (Kripke 1959 [memory]) and 2-EXPTIME-complete (Schmitz
   2016 [cited]) but cheap at these sizes. Prototype **[checked]**: all 103,035 implicational formulas with at most
   7 atom occurrences over at most 4 atoms, classified in 7 s:

   | class | count |
   | --- | --- |
   | not a tautology | 74,254 |
   | needs K only | 25,398 (+ 1,572 where no relevant proof was found but no matrix refutes; e.g. P → P → P) |
   | needs W only | 125 |
   | needs K and W | 194 (+ 22 uncertified) |
   | needs neither (linear) | 126 |
   | K alone or W alone suffices (no unique need) | 191 |
   | classical only (needs DNE or Peirce) | 1,153 |

   Contraction-needing theorems are rare among random small formulas, so targets must be generated forward from
   proofs, not filtered from random formulas. For balanced formulas (each atom at most twice) the βη-normal proof is
   unique (Babaev and Solov'ev 1979, English 1982; Hirokawa 1992 [cited]), so there the flags are a function of the
   theorem; balanced formulas are all affine, so this does not cover contraction.
2. **Independence.** K, W and DNE are mutually independent over B, C, I: each has a 3-element matrix validating the
   others and modus ponens and refuting it **[checked]** (Łukasiewicz L3 refutes W, the Gödel 3-chain refutes DNE,
   Sugihara RM3 refutes K). Two traps **[checked by the same search, plus a hand derivation]**:
   - **Do not use Peirce's law as the classical flag.** Contraction is derivable from K, B and Peirce (the
     Tarski–Bernays axioms; the Isabelle AFP entry "Implicational Logic" proves (p → p → q) → p → q from them
     [cited]). No 3-element matrix separates K or W from the rest once Peirce is present.
   - **Do not use Lean's `False` with ex falso.** K is derivable from B, C, I, ex falso and DNE with every variable
     used once: `fun a b => dne (fun k => efq (k a) b)`. Keep F a `Prop` variable and give `dn` as a hypothesis.
   With distractor premises, as in today's corpus, every theorem already uses weakening at the premise level, so the
   K flag is only meaningful if distractors are dropped.
3. **Real choice.** Partly. Whether to reuse a hypothesis is the prover's decision and no goal shape forces it, but
   for a theorem that needs W there is no alternative. Theorems provable both ways exist (P → P is `I` or `S K K`).
4. **Generation and checking.** Generate random λ-terms under a chosen use discipline, take the principal type, add
   atoms' identifications. The Lean text is a strict subset of today's format (`fun` boxes and application), so the
   tokenizer, gate and harness run unchanged **[checked]**:

   ```lean
   theorem l1 ( P Q R : Prop ) : ( P → ( ( P → ( P → Q ) ) → Q ) ) := by have n9 : ( P → ( ( P → ( P → Q ) ) → Q ) ) := ( fun ( n1 : P ) => by have n8 : ( ( P → ( P → Q ) ) → Q ) := ( fun ( n2 : ( P → ( P → Q ) ) ) => by have n3 : ( P → Q ) := n2 n1 ; have n4 : Q := n3 n1 ; exact n4 ) ; exact n8 ) ; exact n9
   theorem l2 ( P Q F : Prop ) ( dn : ∀ { a : Prop } , ( ( a → F ) → F ) → a ) : ( ( ( P → F ) → F ) → P ) := by have n1 : ( ( ( P → F ) → F ) → P ) := dn ; exact n1
   ```
5. **First-hit cost.** As λ-terms: W is one already-seen name cited a second time, the cheapest knockout available,
   and likely elicited within minutes [inferred]. As combinators (2.4): `axW` is a never-seen token plus a correct
   schema instance. The same logical strategy at two surface costs is a direct test of the first-hit hypothesis.
6. **Transfer.** Same as today's setting; the strategies are resource disciplines, thinner than what people mean by
   a proof strategy.

### 2.3 Horn rules over relations (rank 3)

**System.** A few relations on a finite set of constants, facts as premises, rules such as symmetry, transitivity,
reflexivity, monotonicity of a function, composition of two relations. Proof = forward chain.

1. **Strategies.** As many as wanted (10 to 30 rules). One rule per line. "T needs rule X" is Datalog saturation
   without X: polynomial, exact, milliseconds [inferred from standard Datalog; not prototyped].
2. **Independence.** Decided per theorem by the same saturation; refl, symm, trans are the textbook independent
   triple [memory].
3. **Real choice.** Yes: which rule on which facts, among distractors. The goal is an atom and names no rule.
4. **Generation and checking.** Random facts and rules, saturate, sample a derivation, minimise. Lean **[checked]**:

   ```lean
   theorem r1 ( α : Type ) ( R : α → α → Prop ) ( a b c d : α )
       ( symm : ∀ { x y } , R x y → R y x ) ( trans : ∀ { x y z } , R x y → R y z → R x z ) ( refl : ∀ { x } , R x x )
       ( h1 : R a b ) ( h2 : R c b ) : R a c := by have n1 : R b c := symm h2 ; have n2 : R a c := trans h1 n1 ; exact n2
   ```
5. **First-hit cost.** Named: one unseen token plus a three-token conclusion. In-prompt: likely near zero, because
   every rule has the same form and "apply the rule" is one skill.
6. **Transfer.** Weakest of the top three for "strategy"; closest to "use a tool never seen". Zhang et al. (2022)
   show BERT trained on Horn problems learns distribution-specific statistical features, not forward chaining
   [cited], a known hazard of this domain.

### 2.4 Hilbert systems: one axiom schema per strategy (rank 4)

**System.** Axiom schemas plus modus ponens. Two bases: Łukasiewicz's three (K, S, contraposition), which is
Metamath's `ax-1`, `ax-2`, `ax-3` [cited: Metamath Proof Explorer, accessed 2026-11-18], or Curry's B, C, K, W (+ I).

1. **Strategies.** 3 (Łukasiewicz) or 5 (BCKW + DNE). Each proof leaf is one axiom instance. Metamath prints the
   axiom set under every theorem ("This theorem was proved from axioms: ...") [cited].
2. **Independence.** K and S are independent **[checked]**; K, W, C, DNE are independent **[checked]**; B's
   independence was not separable with 3 elements **[unverified]**; I is derivable (`C K K` **[checked in Lean]**).
   Theorem-level need is standard only for the BCKW basis (the sublogics are 2.2's). For Łukasiewicz's basis, "needs
   contraposition" is intuitionistic provability with negations read as atoms [inferred], but "provable without K"
   and "without S" are non-standard logics with no decision procedure I know of.
3. **Real choice.** Fully: no rule is forced by the goal, there is only modus ponens and the choice of instance.
4. **Generation and checking.** Condensed detachment enumerates all proofs; pmGenerator (Discher, 2023; user-defined
   axiom systems since v1.2, 2024) does this exhaustively for arbitrary Hilbert systems [cited]. Lean: axioms as
   prelude theorems with implicit propositions, modus ponens as application, the stated formula on each `have` fixes
   the instance **[checked]**:

   ```lean
   theorem axK {a b : Prop} : a → b → a := fun x _ => x
   theorem axW {a b : Prop} : (a → a → b) → (a → b) := fun f x => f x x
   theorem h1 ( P Q : Prop ) : ( ( P → ( P → Q ) ) → ( P → Q ) ) := by have n1 : ( ( P → ( P → Q ) ) → ( P → Q ) ) := axW ; exact n1
   -- anonymous variant: no schema has a name token; the model only writes an instance
   macro "ax" : term => `(by first | exact axB | exact axC | exact axK | exact axW | exact axI)
   ```
   Bare combinator terms without the stated type do not always elaborate (in `S K K` one proposition is undetermined).
5. **First-hit cost.** Highest of all: an unseen token and a correct instance of a schema with four metavariable
   occurrences. The anonymous variant removes the token and leaves "write an instance of an unseen shape".
6. **Transfer.** Poor proof style (the deduction theorem inflates everything; `P → P` takes five lines), and Discher
   reported in 2026 that LLMs do badly at condensed detachment [cited, a mailing-list post]. Useful as the expensive
   encoding of 2.2, not as a domain of its own.

### 2.5 Permutation puzzles (rank 5)

Strategies are generators (move types). A "theorem" is a target arrangement, a proof is a word, Lean checks it with
`decide` on lists **[checked]**. "Target needs generator g" is membership in the subgroup generated by the others,
polynomial by Schreier–Sims [memory]. Independence holds when the generating set is irredundant. First-hit cost is
one unseen token with no instantiation, the cheapest token knockout there is. Transfer is weak: there is no
statement-dependent reasoning, and the solution hides whatever plan produced it.

## 3. Candidates that fail

| candidate | verdict | why |
| --- | --- | --- |
| Sequent calculus, focused calculi | fails as a taxonomy | The rules are natural deduction's again (right = introduction, left = elimination). Focusing (Andreoli 1992) splits rules into invertible ones, which are forced, and non-invertible ones, which are the real choices; maximal multi-focusing gives canonical proofs but only for multiplicative linear logic (Chaudhuri, Miller and Saurin 2008) [cited]. Keep it as a tool: Dyckhoff's terminating calculus (1992) decides intuitionistic need. |
| Minimal ⊂ intuitionistic ⊂ classical | fails | Nested by construction; it is the present situation. 2.2 is the non-nested replacement. |
| Resolution, tableaux | fails | One rule. Strategies are refinements (unit, input, linear, set of support) that are nested: unit refutations exist iff input refutations do, and unit resolution is complete only for Horn sets [cited: Wikibooks, accessed 2026-11-18]. "Needs a case split" (unit propagation fails) is crisp and linear-time but is one flag. |
| Arithmetic identities and inequalities with a lemma set | fails | Lemmas are interderivable and "needs lemma X" is not decidable in general. Lean core's `omega`, `simp` and `grind` would have to leave the vocabulary. INT (Wu et al., 2021) is the precedent, with 18 ordered-field axioms and its own checker, and measures generalisation to unseen axiom combinations and orders, not need [cited]. |
| Sudoku-style techniques | fails | Named techniques overlap and simulate each other (an X-wing is a special forcing chain) [memory]. Lean sees only the final grid unless a step trace is designed and checked. Shah et al. (2024) train transformers on a seven-strategy solver's traces [cited]; no knockout. |
| Modal cube (axioms T, B, 4, 5, D) | partial | Frame conditions are crisp but not independent (the five axioms give 15 logics, not 32) [memory]; a shallow embedding reduces to 2.3 with quantifiers. |
| Łukasiewicz three-axiom system | partial | See 2.4: independent, but need is undecidable in practice for two of the three. |

## 4. Comparison

| system | strategies | every proof labelled uniquely | independent | need decidable | goal forces rule | Lean core embedding | first hit, named / in-prompt | build (days) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equational, one operation | 3 to 6 axioms | yes, per step | yes [checked A, C, I] | yes, small models + normal forms [checked] | no | premises + `rw` chain [checked] | token + 3 args / unseen premise shape | 3 to 5 |
| structural rules (→, F) | 3 flags | yes, flag set | yes [checked] | yes [checked, 6% uncertified for K] | no, but need leaves no alternative | subset of today's text [checked] | token + schema / one repeated name | 2 to 3 |
| Horn rules | 10 to 30 | yes, per line | per theorem | yes, saturation [inferred] | no | premises + `have` chain [checked] | token + 3 / near zero [inferred] | 2 to 3 |
| Hilbert, BCKW | 4 or 5 | yes, per leaf | K, W, C, DNE [checked]; B unverified | only for K, W, DNE | no | prelude constants + application [checked] | token + schema instance | 3 to 4 |
| permutation words | any | yes, per move | by choice of generators | yes [memory] | no | `decide` [checked] | one token | 1 to 2 |

## 5. Recommendation and pilot

Switch target: **2.1, the one-operation equational domain.** It is the only candidate that meets all five
requirements with more strategies than natural deduction has, and each of its claims above was cheap to check.

The pilot needs no RL. The working hypothesis says emergence is governed by first-hit cost, so measure that
directly:

1. Build the generator, the need labeller and the Lean gate for A, C, I (about three days; the labeller prototype is
   `od_work/eq_census.py`).
2. Pretrain four bases, 10 minutes each: knock out C, knock out I, each in the named and the in-prompt encoding.
3. From each base sample about 100,000 attempts on targets certified to need the removed axiom exactly once, and
   count Lean-accepted proofs. No RL. About 1.5 pod-hours in total [estimate].

Reading: the domain works if the hit rate lands between about 1e-5 and 1e-2 in at least one cell and moves with the
number of required applications. If the named cells are 0 and the in-prompt cells are near 1, there is no middle
regime and the domain only reproduces today's two outcomes. The same four-cell measurement on 2.2 (knock out
contraction; λ form against `axW`) costs two more days and uses the existing harness.

## 6. Files

All under `/private/tmp/claude-501/-Users-robbie-repos-nd-rl/72c64566-c4a9-4e6a-90f1-be85b62895cc/scratchpad/strategies/od_work/`:

| file | what |
| --- | --- |
| `embed.lean` | the five embeddings; `lean embed.lean` passes on Lean 4.34.0 with no imports |
| `embed_bad.lean` | the same with a wrong axiom instance enabled; Lean rejects it |
| `matrices.py` | brute-force 3-element matrices for axiom independence |
| `census.py`, `census2.py` | structural-rule need census for implicational formulas |
| `eq_census.py` | axiom-need labeller for A, C, I by small countermodels, compared with normal forms |

Not tested: the `step` macro in `embed.lean` (anonymous equational steps) is defined but never used; treat anonymous
equational steps as unbuilt.

## 7. Sources

- Wu, Jiang, Ba, Grosse. "INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving." ICLR 2021. https://arxiv.org/abs/2007.02924
- Saparov et al. "Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples." NeurIPS 2023. https://arxiv.org/abs/2305.15269
- Zhang, Li, Meng, Chang, Van den Broeck. "On the Paradox of Learning to Reason from Data." arXiv 2022, IJCAI 2023. https://arxiv.org/abs/2205.11502
- Tao et al. "The Equational Theories Project: Advancing Collaborative Mathematical Research at Scale." arXiv 2025. https://arxiv.org/abs/2512.07087
- Discher. pmGenerator (2023; v1.2 2024). https://github.com/xamidi/pmGenerator ; 2026 mailing-list post: https://groups.google.com/g/metamath/c/NATmJ843OSQ
- Metamath Proof Explorer, propositional axioms (accessed 2026-11-18). https://us.metamath.org/mpeuni/mmtheorems1.html
- Wikipedia, "Implicational propositional calculus" and "B, C, K, W system" (accessed 2026-11-18). https://en.wikipedia.org/wiki/Implicational_propositional_calculus , https://en.wikipedia.org/wiki/B,_C,_K,_W_system
- Isabelle AFP entry "Implicational Logic" (accessed 2026-11-18; publication year not checked). https://www.isa-afp.org/browser_info/current/AFP/Implicational_Logic/document.pdf
- nLab, "combinatory logic" (accessed 2026-11-18). https://ncatlab.org/nlab/show/combinatory+logic
- Hindley. "BCK and BCI logics, condensed detachment and the 2-property." Notre Dame Journal of Formal Logic 34(2), 1993. doi:10.1305/ndjfl/1093634655
- Hirokawa. "Balanced formulas, BCK-minimal formulas and their proofs." 1992. https://link.springer.com/doi/10.1007/BFb0023874
- Babaev, Solov'ev. "A coherence theorem for canonical morphisms in cartesian closed categories." 1979 (Russian), J. Math. Sci. 20, 1982. https://link.springer.com/article/10.1007/BF01629434
- Statman. "Intuitionistic propositional logic is polynomial-space complete." Theoretical Computer Science 9(1), 1979 (seen in reference lists only).
- Schmitz. "Implicational Relevance Logic is 2-EXPTIME-Complete." Journal of Symbolic Logic, 2016. https://arxiv.org/abs/1402.0705
- Arthan, Oliva. "On Affine Logic and Łukasiewicz Logic." arXiv 2014. https://arxiv.org/abs/1404.0570
- Andreoli. "Logic programming with focusing proofs in linear logic." Journal of Logic and Computation 2(3), 1992 (seen in reference lists only).
- Chaudhuri, Miller, Saurin. "Canonical Sequent Proofs via Multi-Focusing." IFIP TCS 2008. https://www.lix.polytechnique.fr/~dale/papers/tcs08trackb.pdf
- Dyckhoff. "Contraction-free sequent calculi for intuitionistic logic." 1992 (year and result confirmed via Wikipedia's "Roy Dyckhoff"; venue not confirmed).
- Shah, Dikkala, Wang, Panigrahy. "Causal language modeling can elicit search and reasoning capabilities on logic puzzles." NeurIPS 2024. https://arxiv.org/abs/2409.10502
- Wikibooks, "Logic for Computer Scientists: Input and Unit Resolution" (accessed 2026-11-18). https://en.wikibooks.org/wiki/Logic_for_Computer_Scientists/Predicate_Logic/Strategies_for_Resolution/Input_and_Unit_Resolution
- From memory, no source found today: Kripke, "The problem of entailment" (abstract, 1959) for decidability of relevant implication; the Schreier–Sims algorithm; absorption implying idempotence in lattices; the count of 15 normal modal logics in the modal cube.
