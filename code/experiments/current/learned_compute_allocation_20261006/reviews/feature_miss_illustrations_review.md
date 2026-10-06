---
written_on: 2026-10-06
written_by: agent:codex
---

# Review of two post-hoc feature-miss illustrations

Accepted both simplified descriptions in the reader selection summary against the archived Lean strings in `artifacts/learned_compute_allocation_20261006/analysis/feature_misses_main_descriptive_v1.json`. No inference, new verifier check, prevalence estimate or tuning was performed; final joint-live outcomes remain unopened.

For `b3d8c290be045f21d09a`, define A=((R→S)∨P), B=(((S∨S)∨S)∨¬¬Q), G=((P→S)∧(S∧(P→R))). Candidate6 binds n6:A→G and n7:B→G. Both Or.elim branches invoke n6 n9, but the second branch binds n9:B. This is exactly the described wrong-branch implication application. Valid candidate1 instead invokes n13:A→G on the first branch and n14:B→G on the second. The archive supports eight complete candidates, one valid candidate, selected ID6, valid ID1, and rounded logits2.8235 and2.7272.

For `30025165b81905655c49`, define A=(¬((Q∧P)→(Q→S))∧¬¬P) and G=(¬P→((R∧¬(Q∧S))∧Q)). Candidate0 has n28:A, n29:¬A and correctly constructs n30:¬¬G using the contradiction n29 n28. Its final term is `Classical.byContradiction (fun hh => n29 hh)`. Here hh:¬G, whereas n29 requires an argument of type A. It should use the available double-negation proof n30 on hh, or directly eliminate the contradiction from n29 n28. Thus the description of the wrong hypothesis supplied to a negation, despite a useful intermediate statement and contradictory premises, is accurate. The archive supports seven valid complete alternatives and selected logit2.8118 above them.

The prose explicitly labels the inspection post-hoc and disclaims error-type prevalence. Its observation that some forward tasks have a short contradiction proof is warranted by this concrete example; it makes no unsupported claim about how common that property is. No requested corrections.
