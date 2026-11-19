---
written_on: 2026-11-18
written_by: agent:claude
---

> Markdown export of the shared doc at https://claude.ai/artifact/LTdzaX9RjPZxGZsYeV4w5F (2026-11-18, 10:55 pm PT). The doc is the live copy.

# A generator basis for proof strategies

Nov 18, 2026 · @Robbie · Written by Robbie's Claude

## Summary

Restricting to implication gives a complete basis of four generators, and three switches in it are fully independent of each other.

- **Dmitry's suggestion has a precise form.** In implication-only logic, with ¬a read as a → F, every proof is built from four generators (B, C, K, W) and modus ponens.
- **Three switches are independent.** K (ignore a hypothesis), W (use one twice) and double-negation elimination cannot simulate each other. That gives 8 exact classes of theorem, not a ladder.
- **"Only four" is the price of a clean basis.** B and C are needed almost everywhere, so the real switches are those three. More strategies come from short combinations of generators, graded by how many steps the first success takes.
- **We can test it this week without building anything.** In our current proof format W is "an assumption cited twice". Removing it costs 1.24% of pretraining.
- **Recommendation.** Run the assumption-reuse knockout on the current harness first. Build the implication-only dataset (2 to 3 days) only if that result is interesting.

## Why rules were the wrong grain

Tonight's rule knockouts showed that "one rule = one strategy" gives too few strategies, and that what predicts emergence is the cost of the first accidental success.

We removed one natural-deduction rule from pretraining per arm, ran RL, and counted whether the rule appeared in Lean-accepted proofs. Numbers are final, from 2026-11-18: about 3 hours of RL for the expert-iteration runs and 2 hours for GRPO.

| Rule removed | Runs | Result |
| --- | --- | --- |
| Modus ponens | 4 | Appeared in 1 run at 34 minutes; 10,185 distinct proofs on 2,024 targets by the end. The other 3 runs: zero. |
| Conditional proof | 4 | Zero in all 4. |
| Ex falso | 4 | Zero. The arm solves ex-falso targets another way, apparently by proof by contradiction. |
| Double-negation elimination | 4 | Zero. |

The three rules that never appeared account for about 1.6 million accepted attempts across 12 runs.

Four things are wrong with rules as the unit:

- **There are only four or five of them.**
- **Introduction rules are forced.** A goal shaped a → b tells you to use conditional proof. That is not a choice.
- **They are nested, not independent.** Proof by contradiction can stand in for ex falso, so removing ex falso leaves no gap.
- **They are expensive to remove.** Deleting every proof with modus ponens or conditional proof drops about 38% of pretraining.

Modus ponens was the cheapest first success: two familiar tokens in a new arrangement. Ex falso and double-negation elimination needed tokens the model had never emitted.

Final results are on Robbie's [Rule Knockouts page](https://claude.ai/artifact/GRRoyzSR77mVWUtQByUx5Y).

## The four generators

Keep only implication, and every proof is a combination of four axiom schemas and modus ponens.

Negation is not a separate connective here. ¬a is a → F, where F is an ordinary atom with no special rule.

| Generator | Formula | What it lets a proof do | How it shows up in our proof format |
| --- | --- | --- | --- |
| B | (b → c) → (a → b) → (a → c) | Chain two implications | Feeding one derived line into the next |
| C | (a → b → c) → (b → a → c) | Use hypotheses in a different order | Invisible, because lines are cited by name |
| K | a → b → a | Ignore a hypothesis | An assumption that is never cited |
| W | (a → a → b) → (a → b) | Use a hypothesis twice | An assumption cited more than once |

One more schema gives classical logic: double-negation elimination, ((a → F) → F) → a.

The identity a → a needs no generator of its own. It is derivable from C and K, which the agent checked in Lean.

Under the proofs-as-programs reading these are habits of variable use. A proof with no K and no W uses every hypothesis exactly once. K allows "at most once", W allows "at least once", and both together allow anything.

## Why the basis is exhaustive and independent

The four generators produce every intuitionistic proof in the fragment, and switching K, W and double-negation elimination on or off gives a lattice of logics, not a ladder.

- **Exhaustive.** B, C, K and W with modus ponens are a complete basis for intuitionistic implication. Nothing else is needed.
- **Independent.** For each of K, W and double-negation elimination there is a 3-valued truth table that validates the other two and refutes it. The agent found all three by search. B's independence was not verified.
- **Exact need.** "This theorem needs W" is decidable, so every theorem gets a label that measures need and not use.

(Diagram in the shared doc: a diamond with linear logic at the bottom, affine (K only) and relevant (W only) side by side above it, intuitionistic (K and W) above those, and classical on top.)

Read it bottom up. Affine and relevant logic sit side by side because neither K nor W gives you the other.

The need labels were computed for all 103,035 implication-only formulas with at most 7 atom occurrences over at most 4 atoms, in 7 seconds:

| Class | Formulas |
| --- | --- |
| Not a tautology | 74,254 |
| Needs K only | 25,398 (plus 1,572 not certified) |
| Needs W only | 125 |
| Needs K and W | 194 (plus 22 not certified) |
| Needs neither | 126 |
| Either K or W alone is enough | 191 |
| Classical only | 1,153 |

Two things stand out. Theorems that need W alone are rare among random small formulas. And 191 theorems have no unique need: K alone proves them and so does W alone.

## Only four?

Yes at the level of generators, and that is what a clean basis costs. A larger set of strategies comes from combining the generators, not from finding more of them.

There are three ways to get more than four.

| Route | What a strategy is | How many | Clean? |
| --- | --- | --- | --- |
| Need classes | The set of switches a theorem needs (K, W, classical) | 8 | Exact and independent |
| Words in the generators | A short combination that pretraining never contained | As many as we define | Graded by length, not unique |
| More connectives | The introduction and elimination moves for ∧ and ∨ | About 6 more | Mostly forced by the theorem |

**Need classes.** Knock out a class of theorem, not a rule. The model keeps seeing every token, but never sees a proof that had to use a hypothesis twice.

**Words in the generators.** This is where a long list of strategies comes from, with the generators as the alphabet. A strategy is a combination such as "reuse an assumption inside a nested box", and its length is the number of steps the first accidental success takes. Tonight's control runs already show such a ladder. Patterns with zero pretraining proofs first appeared at very different times:

- Disjunctive syllogism: sampling round 1 to 5.
- Three chained modus ponens: rounds 9 and 10, about 15 minutes of RL.
- A case split nested in the first branch of another: never, though 51% of reference proofs contain one.

Those three are the corpus agent's counts and have not been hand-checked.

**More connectives.** Adding ∧ and ∨ back brings pairing, projection, injection and case analysis. Whether a theorem needs them is mostly decided by which connectives it contains, so they add generators without adding choices. The exception is a case split on a disjunction the proof derived, which is a real choice.

A fourth option is a dial, not a list: the number of times a hypothesis must be reused. That is a proposal and nobody has checked that it gives a usable scale.

## What our current data already says

Two of the three switches can be read off the proofs we already have, and W is the one with a large gap between pretraining and the RL targets.

| Switch | Detector on a proof | Pretraining proofs | RL reference proofs | Control model's accepted proofs |
| --- | --- | --- | --- | --- |
| W, reuse | A box assumption cited twice or more | 1.24% | 83.1% | 12.8% |
| K, ignore | A box whose assumption is never cited | 30.7% | 2.0% | 58.4% |
| Classical | The double-negation rule | 8.4% | 1.9% | about 5 to 7% of accepted attempts |

- The W and K columns are the proof-theory agent's counts. The control figures are one seed.
- The W detector counts one use in each branch of a case split as two. That is not fixed yet and inflates the 83%.
- The classical row is from tonight's knockout runs.

The RL targets can also be labelled by the weakest logic that proves each one. An exact prover did all 18,439 in 2 seconds: 13,361 need neither ex falso nor classical reasoning, 4,509 need ex falso, 569 need classical reasoning.

So the cheapest test of the whole idea needs no new dataset. Remove the 1.24% of pretraining proofs that reuse an assumption, and see whether reuse appears in RL. Every token and every syntax form stays visible, and the first success is one step: citing a line a second time.

## Traps

Five things break the clean picture if we are careless, and the last one is the most important.

- **Peirce's law cannot be the classical generator.** With B and K it gives you W for free, so the switches stop being independent. Use double-negation elimination.
- **Lean's built-in `False` cannot be F.** Ex falso plus double-negation elimination gives you K for free. F has to be a plain proposition variable, with double-negation elimination supplied as a hypothesis.
- **Our distractor premises switch K on everywhere.** A premise the proof never uses is an ignored hypothesis. For K to mean anything the distractors have to go.
- **Targets must be generated, not filtered.** Only 125 of the 28,781 small tautologies need W alone. A target set has to be built forward from proofs that use W.
- **A named generator is an unseen token.** If proofs are written as combinator terms, a removed generator is a symbol the model has never emitted. That is the case that never emerged tonight. Written in our current format, K and W are citation patterns and no token is new.

## Proposed experiments

Run the cheap test on the current harness first, and build the new dataset only if it pays.

| Order | Experiment | Cost | What each outcome means |
| --- | --- | --- | --- |
| 1 | Assumption-reuse knockout on the current harness, 3 seeds plus a control | About 4 pod-hours a run at $0.59, so roughly $10 in all | Reuse appears fast: a second ignition beside modus ponens, and the first-hit-cost account holds. Reuse never appears: "never seen means never appears" holds even when the first success is one step. |
| 2 | Read the first-hit rate straight off tonight's checkpoints, no training | About 1.5 pod-hours | Predicts when each arm should ignite, and tests that against the three modus-ponens runs that never ignited. |
| 3 | Build the implication-plus-F dataset with exact need labels for all 8 classes | 2 to 3 days of build, by the agent's estimate | Lets us knock out a need class cleanly and compare K, W and classical on equal terms. |
| 4 | A ladder of combinations by length, in either dataset | 3 seeds a rung | Tests whether time to emergence grows with the length of the shortest unseen combination. |

Before experiment 1 runs, two things need doing:

1. Fix the reuse detector so one use per case-split branch does not count as reuse.
2. Measure how often each knockout base writes the removed form before any RL. Earlier work in the group found that guessing acquisition from syntax fails.

## What this does not give us

The basis is clean because it is small and structural, and both of those are limits.

- **These are not textbook strategies.** We get "reuses a hypothesis", not "argues by cases" or "argues by contradiction".
- **Three independent switches is still few.** A long list of strategies only comes from combinations, and combinations are graded, not unique.
- **Need can be ambiguous.** 191 small theorems are provable with K alone or with W alone.
- **Transfer to language models is unargued.** Nobody has said what "uses a fact twice" corresponds to in a model trained on text.

Open questions for the group:

- [ ] Is a strategy a property of a theorem (need) or of a proof (use)? The y-axis of the emergence plot depends on it.
- [ ] Are 8 exact classes enough for the paper, or do we need the graded combinations as well?
- [ ] Do we drop distractor premises so that K means something?
- [ ] Does the classical switch belong in the first version, given that an unseen token never emerged tonight?

## Sources

The measurements in this note come from four research agents run on 2026-11-18. Their counts on our data have not been independently re-checked.

Opened and checked by the agents:

- Hindley, "BCK and BCI logics, condensed detachment and the 2-property", Notre Dame Journal of Formal Logic 34(2), 1993. [doi:10.1305/ndjfl/1093634655](https://doi.org/10.1305/ndjfl/1093634655)
- nLab, ["combinatory logic"](https://ncatlab.org/nlab/show/combinatory+logic), accessed 2026-11-18, for the match between combinator bases and linear, affine, relevant and intuitionistic implication.
- Wikipedia, ["B, C, K, W system"](https://en.wikipedia.org/wiki/B,_C,_K,_W_system) and ["Implicational propositional calculus"](https://en.wikipedia.org/wiki/Implicational_propositional_calculus), accessed 2026-11-18.
- Isabelle Archive of Formal Proofs, ["Implicational Logic"](https://www.isa-afp.org/browser_info/current/AFP/Implicational_Logic/document.pdf), accessed 2026-11-18, for the derivation of W from K, B and Peirce's law. Publication year not checked.
- Hirokawa, ["Balanced formulas, BCK-minimal formulas and their proofs"](https://link.springer.com/doi/10.1007/BFb0023874), 1992.
- Schmitz, ["Implicational Relevance Logic is 2-EXPTIME-Complete"](https://arxiv.org/abs/1402.0705), Journal of Symbolic Logic, 2016.
- Johansson (1937) for minimal logic, Glivenko (1929), Prawitz (1965) and Dyckhoff (1992), confirmed by the proof-theory agent.

Seen only in reference lists, or from memory:

- Statman, "Intuitionistic propositional logic is polynomial-space complete", Theoretical Computer Science 9(1), 1979.
- Kripke, "The problem of entailment", 1959, for decidability of relevant implication.
- Girard (1987) on linear logic and Anderson and Belnap (1975) on relevant logic.
- The claim that ∧ and ∨ are needed only when they occur in the theorem is from memory and unsourced.

Checked by running code today: the three independence truth tables, the classification of 103,035 formulas, the Lean embedding of the fragment, and the derivation of the identity from C and K.

Experiment code and the plan for tonight's knockouts are on the `robbie-experiments` branch of the nd-rl repo, under `code/experiments/current/rule_knockout_20261118/`.
