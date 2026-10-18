# Run brief: proof-strategies — a taxonomy of the proof strategies in our pretraining data

Run id `proof-strategies`. Repository `~/work/proof-strategies` (branch `dan_proof-strategies` from the fork's
`origin/dan_create-or-elicit`, made by `newrun`). Role: executor. `AGENT_POLICY.md` governs. Authorised by Dan
2026-10-18 (proposal 28).

- **Budget:** **$20 / 40 pod-hours.** This is mostly CPU work; use pods only where an analysis needs a model.
- **Thinking:** every role runs at `--effort max`.
- **Time:** hard stop 72 h; no hurry.

## The question

Dan: "classifying the proofs in the pretraining by proof strategy. For example, 'sigma' was previously used to
describe what I would generally call proof-by-contradiction, and there was a whole knockout experiment on it. I'd like a
semi-complete classification of proof strategies. We should have canonical examples for each strategy as well as a rough
percentage of the pre-training data that uses each strategy."

The deliverable is a **named, semi-complete taxonomy of proof strategies**, with:
- a precise, checkable definition of each;
- canonical examples;
- each strategy's share of the pretraining data.

It should replace ad-hoc labels like "σ" with names a logician would recognise, and say how they map.

## Data

- **Main:** K12 (`data/kh/train_k12.jsonl`, 155,000 ND proofs, the pretraining set of `state_env_october`).
- **Comparison sets** (classification only):
  - the cap-6 set `data/p2/train_depth3_f0_a1.jsonl`;
  - the knockout set K12⁻ from `create-or-elicit`;
  - the RL targets (`rl_targets`, 4,495) and transfer pool;
  - holdout250;
  - Charles's textbook release (nd-rl `origin/charles/main`, `data/textbook_problem/release_2026-10-02/`; base problems,
    reference proofs);
  - textbook72 (**scoring only**: classify its reference proofs, never train on them).

  Comparing the generator's mix with the textbooks' is one of the most useful outputs. Which strategies do textbooks use
  that our data rarely shows?

## Building blocks already on the fork

Reuse these where they fit and say where your definitions differ:
- `patterns.py` (P1 derived ∨-elimination, P2 reductio, …) and `patterns2.py` (nested ∨-elimination, →E chains, …);
- `coe/coe_sigma.py`: σ strict, σ chain, σ⁺, as in the `create-or-elicit` knockout;
- `intuit.py`: the G4ip prover, which labels classical-only theorems;
- `necessity.py`: does a theorem *require* a pattern?;
- `prune.py`: dependency pruning;
- `minlen.py`.

## Design (a suggestion)

1. **The taxonomy.** Start from standard natural-deduction and proof-theory vocabulary, then refine against the data
   until nearly every proof is covered. Candidates:
   - direct proof / forward chaining (→E, ∧E chains);
   - conditional proof (→I);
   - proof by cases (∨E);
   - ∨-introduction for disjunctive goals;
   - proof of a negation (¬I: assume A, derive ⊥);
   - **proof by contradiction proper** (classical reductio: assume ¬G, derive ⊥, then double negation; the
     knockout's σ);
   - ex falso / explosion;
   - double-negation elimination on a supplied ¬¬A;
   - modus tollens and contraposition;
   - disjunctive syllogism;
   - excluded middle and other classical lemmas;
   - lemma reuse (one derived line used by several later steps).

   Keep this list at a granularity a reader can hold in mind (≈ 10–20 strategies), and record finer variants as
   sub-types.

   For each strategy give:
   - its standard name(s), with a literature or textbook reference;
   - a formal predicate over the pruned ND proof (rules, box structure, citations);
   - whether it is **classical** (needs DN) or intuitionistic;
   - how it relates to σ / σ⁺ and to `patterns.py`'s predicates.
2. **Two levels of label, both reported:**
   - **"uses":** the strategy appears anywhere in the proof (multi-label);
   - **"primary":** how the conclusion is reached at the top level (one label per proof, with a stated precedence rule).
3. **Theorem level as well as proof level, where cheap.** Does the theorem *require* the strategy? Use `intuit.py` for
   classical-only theorems, and `necessity.py`-style bounded search for a few strategies on a sample. The data's one
   proof per theorem shows only how the generator proved it.
4. **Canonical examples.** For each strategy, the shortest clean instance plus one typical instance from the data,
   shown in ND and in Lean (rendered and Lean-checked), with a one-sentence plain-English gloss.
   - Never print a theorem in a textbook72 renaming class; use the `--sealed` guard method from nd-rl PR #26's
     `make_golden.py`.
   - Prefer data examples; hand-written ones are labelled as such.
5. **Validation.**
   - **Spot-check:** an independent subagent hand-checks ≥ 20 proofs per strategy (random, plus borderline cases).
   - **Second implementation:** a second subagent implements ≥ 3 key predicates independently, including proof by
     contradiction and proof by cases, and reports agreement.
   - **Coverage:** report the share of proofs the taxonomy leaves unlabelled, and characterise them.
6. **Percentages.** Report each strategy's share of K12 ("uses" and "primary"), with the comparison sets beside it, by
   proof length and box depth.
7. **Optional, if it sharpens the story (CPU first; pods only if needed):** link strategy frequency in pretraining to
   what our models learn. Examples:
   - per-strategy solve rates of `state_env_october` / `_r16` on held-out theorems, from existing stored reads;
   - the strategy mix of RL's own accepted proofs against the pretraining mix.

   Pre-register any pod job in `log.md`.

## Deliverables

1. `proof_strategies/TAXONOMY.md`: the definitions, the mapping to σ and `patterns.py`, the precedence rule, and the
   validation results.
2. `proof_strategies/TABLES.md`, and figures: the percentages in K12 and the comparison sets.
3. `proof_strategies/blog.md`: **a simple write-up in the style of the earlier blog post**
   (`docs/papers/create-or-elicit/blog.md` on nd-rl `dan`).
   - 1,000–2,000 words, for readers outside the subfield, with every term defined at first use.
   - It covers: the taxonomy as a short guide with one canonical example each (Lean plus English); the
     pretraining-mix figure; what textbooks use that our data lacks; and why it matters for the knockout result.
   - Before finishing, an independent subagent checks every number in it against `TABLES.md`.
4. Code under `proof_strategies/` (a classifier with tests); `numbers.md` § proof-strategies; `log.md`;
   `run_proof_strategies.md` (≤ 200 words); `STATUS.md` line `PROOF-STRATEGIES DONE <UTC>`; and
   `touch ~/runs/proof-strategies/executor.done`.
5. **The librarian** copies the taxonomy, tables, figures and blog to nd-rl `docs/papers/proof-strategies/`, and writes
   the summary as usual.

**For the reviewer:** re-implement two or three predicates blind and recount their percentages on K12; check the
canonical examples (Lean-accepted, correctly labelled, outside textbook72 classes); check the blog's numbers.

**Rules:**
- Do not contact anyone.
- textbook72 is scoring and analysis only, and never printed.
- Other people's branches are read-only.
- At most 4 concurrent processes on the VPS.
