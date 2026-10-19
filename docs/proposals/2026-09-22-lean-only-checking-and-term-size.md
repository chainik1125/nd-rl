---
written_on: 2026-09-22
written_by: agent:claude
---

# Proposal 9: Lean as the only checker, and proof size as the length unit

Decided by Dan on 2026-09-22 after the `lean_seq` adoption: retire `nd_verify` as reward and
checker, enforce "un-automated Lean" on the elaborated term, and test free-form Lean against
the ND-shaped fragment with term size as the unit. **Lines are still reported everywhere, for
consistency with the rest of the team; term size is reported beside them.**

## Phase 1 — Lean is the only checker (engineering; no new training)

1. `lean_check`: a Lean metaprogram (or `lean --run` script) that, for each candidate proof,
   elaborates it against the theorem statement and accepts iff (a) it elaborates, (b) `#print
   axioms` contains nothing beyond `propext`, `Classical.choice`, `Quot.sound` — in particular
   no `sorryAx` — and (c) every constant in the elaborated term is on the **allowlist**: the
   constructors and eliminators of `And`, `Or`, `Not`, `False`, `Iff` (`And.intro`, `And.left`,
   `And.right`, `Or.inl`, `Or.inr`, `Or.elim`, `False.elim`, `absurd`, `Not` as a function
   type, `Iff.intro`, `Iff.mp`, `Iff.mpr`) plus `Classical.byContradiction` (double negation) —
   nothing else: no `simp`, `tauto`, `decide`, `omega`, `aesop`, `Classical.em`, `Decidable.*`,
   `Or.resolve_*`, or any library lemma. Constants introduced by the proof's own `have`s and
   lambdas are of course allowed. The check inspects the term, not the source text, so tactic
   sugar (`intro`, `exact`, `constructor`, `cases`, `left`/`right`, `refine`) is fine when it
   elaborates to allowed constants only.
2. `term_size`: the number of inference nodes in the elaborated proof — applications of
   allowed constants plus lambda binders — reported beside the ND line count for every proof.
   For fragment (`lean_seq`) proofs also report the line count from the rendering.
3. Validation: on the 181,464 pool proofs and the 460 known disagreements from `lean-format`,
   report `lean_check` vs `nd_verify` (expected: agreement on all accepted proofs after the
   `BOTE` fix; the 137 `¬A ≡ A → False` and 117 unrestated-premise texts accepted by
   `lean_check` and labelled as ND-formality cases). Relabel the ladder transfer pool and the
   depth-3 pools with `L_true` **and** the minimum term size the bounded prover finds
   (`minlen.py` proofs translated and sized).
4. Reward for all subsequent runs = `lean_check` only. `nd_verify` stays in the repository as
   an audit tool the reviewer may still run on fragment proofs; it is no longer the reward or
   the checker of record. Throughput: report samples per second; batch elaboration.

## Phase 2 — free-form un-automated Lean vs the fragment (experiment)

Same model size and schedule as proposal 8. Two formats from the same cap-6 data: the
`lean_seq` fragment (one `have` per rule application, control) and **free-form** — the same
proofs rendered as compact terms (`⟨h.1, h.2⟩`, inline `Or.elim h (fun a => …) (fun b => …)`,
nested `fun`), i.e. the shortest term `nd2lean` can produce without `have` scaffolding —
under `lean_check`. Two Stage-1 seeds each. Measures: held-out greedy (same held-out set, both
units); pre-RL base rates of depth-3 (lambda nesting ≥ 3) and reductio (`byContradiction` on a
lambda) on the fixed target pools at pass@2,000; the depth-3 f = 0 dial; ladder rung T1 with
the frozen control, reporting `L*` in **lines** (via the ND proof the term denotes, when it
denotes one; otherwise via the bounded prover's ND length for that theorem) **and in term
size**.

**Pre-registered predictions.** Free-form base models solve more held-out theorems at equal
sample count than the fragment (≥ 3 pp) because terms are shorter (report tokens per proof for
both); depth-3 base rate higher still; expert iteration's increment over frozen no larger than
in the fragment; the `L*` in lines unchanged at 11 and, in term size, the free-form frontier ≥
the fragment's. The falsifier that would matter: free-form RL reaching a line-`L*` of 12, or
acquiring a rule-sequence pattern from f = 0 that neither the token nor the fragment model did.

## Budget, stop rule, deliverables

Phase 1 ≈ $5 (CPU on a small pod or the host); phase 2 ≈ $30. Stop phase 2 if `lean_check`
throughput makes a round exceed 3× the `lean_seq` round. Deliverables: `lean_check`, `term_size`
with tests; relabelled pools committed; `run_lean_only.md` (≤ 400 words + figures, both units
in every table); `numbers.md`/`log.md`; bucket upload; review; librarian merge into `dan`.
