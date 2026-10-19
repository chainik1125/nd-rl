---
written_on: 2026-10-18
written_by: agent:claude
status: approved by Dan 2026-10-18, $20 (launched as `proof-strategies`)
---

# Proposal 28: a taxonomy of proof strategies in our pretraining data

## Why

The knockout experiment (`create-or-elicit`) removed one move, called "σ": assume ¬G, derive a contradiction, then
drop the double negation. Logicians call this proof by contradiction (classical reductio). We named it ad hoc and
measured nothing else about what our pretraining data teaches. Dan (2026-10-18) wants a semi-complete classification
of proof strategies, with canonical examples and each strategy's share of the pretraining data. That makes the next
knockout or dose experiment a choice from a known menu, and shows how our generator's mix differs from textbooks'.

## Plan

| step | what |
|---|---|
| taxonomy | ≈ 10–20 named strategies, with sub-types: direct proof, conditional proof, proof by cases, proof of a negation, **proof by contradiction proper** (σ), ex falso, double-negation elimination, modus tollens / contraposition, disjunctive syllogism, excluded middle and other classical lemmas, lemma reuse, … Each gets a standard name, a formal predicate on the pruned ND proof, and a classical / intuitionistic label, mapped to σ / σ⁺ and the existing `patterns.py` predicates |
| labels | "uses" (anywhere, multi-label) and "primary" (the top-level route to the conclusion), plus theorem-level "requires" where cheap (G4ip for classical-only; bounded search on a sample) |
| data | K12 (main), with the cap-6 set, the knockout set, the RL targets and transfer pool, holdout250, Charles's textbook release, and textbook72 (analysis only, never printed) beside it |
| examples | per strategy, the shortest clean instance plus a typical one, in ND and Lean (Lean-checked), with an English gloss; nothing in a textbook72 class |
| validation | ≥ 20 hand-checked proofs per strategy, a second independent implementation of key predicates, and the unlabelled share |
| optional | link strategy frequency to what models learn, from stored reads (CPU first) |
| write-up | a simple blog-style post (1,000–2,000 words) like the earlier one, every number checked against the tables |

**Cost:** mostly CPU, Claude-heavy; pods (≤ $20) only if a model analysis needs them. **Run:** `proof-strategies` on
agent2, `--effort max`, 72 h hard stop. **Brief:** [claude-heavy/BRIEF_proof-strategies.md](claude-heavy/BRIEF_proof-strategies.md).
