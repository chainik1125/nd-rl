---
written_on: 2026-10-06
written_by: agent:codex
---

# Transfer scope and task compatibility

The strongest unchanged-method transfer protocol evaluates Charles's 187 ordinary sequents as primary, with the 225 derived sequents as a separately labeled projection diagnostic. Reuse the already completed historical 72 evaluation without rerunning or retuning it. This follow-up is separate from the closed sprint.

Source inspected: `origin/charles/main` commit `5d0084add453212e6bea8055d4d28cec790f268c`, release `data/textbook_problem/release_2026-10-02/README.md`, its derived records, and `code/textbook_problem/derive.py`. The ordinary files contain 87 + 72 + 14 + 14 problems. Metadata-only inspection of the 225 derived records finds 95 nonempty `given` fields, 29 `verifier+check` scoring contracts, and 144 designated curried twins. These categories overlap and are not disjoint populations. The release README's linked `candidates_derived/README.md` is absent from this pinned tree.

Native derived evaluation is not equivalent to solving its `prompt`. Prefix tasks require preserving supplied proof lines and completing that proof. Structural tasks require a line-budget, required-rule or forbidden-rule check on the native ND proof. Repair/cloze marker words are an additional interface; the release explicitly says those tasks were not feedable to its scored models, and its prior proof-state RL evaluation also excluded prefix tasks. A Lean-valid generated proof alone does not meet these contracts. Implementing native support would require validated prefix/state reconstruction, native output semantics and structural checks, not just an input string adapter.

Sequent-only projections can therefore be useful, but must archive the discarded public task requirements and be called ordinary theorem-solving diagnostics. Do not present all 412 as independent native-task successes or claim adaptation to repair, cloze, constraints, or supplied prefixes. Report derived results by nuance/base where useful; shared bases and designated twins limit independence. The release's own historical overlap audit is not proof that these exact r16 and sprint checkpoints never encountered equivalent material. Four long reference proofs lack a complete Lean rendering; do not use their reference renderer limitation to exclude otherwise parseable theorem statements.

## Three arms and pass-at-k terminology

The fixed-four-trained checkpoint, cost-trained checkpoints and no-cost checkpoints are distinct training arms. Returning first or selecting after a fixed number of attempts from each checkpoint is a deployment rule. Report both explicitly; neither the original r16 fixed-eight baseline nor the joint fixed-four control alone captures both meanings of “fixed budget.” Keep selected checkpoint ages, all caps, temperatures and original policies unchanged. Cost/no-cost means cover three training seeds; the separate fixed-four training control has one seed.

For original-model independent attempts, report sampled mean validity as the pass1 estimate and the fraction with any valid candidate among eight as oracle pass8. Selected fixed-eight correctness is a separate observable. For joint models, report first-proof correctness and oracle availability on the saved contextual forced-eight path. Contextual attempts are not iid; do not apply an iid pass-at-k estimator to them or equate average validity over path positions with first-proof success. Two paths per theorem give finite Monte Carlo estimates, and theorem intervals remain conditional on the selected trained checkpoints and recorded draws.

Primary usefulness comparisons remain the learned policy versus its own same-checkpoint fixed rules and count controls, alongside the cross-training-arm table. A gain against a weaker original generator alone does not establish learning when to stop. Neither transferred stopping behavior nor coherence of repeated proofs establishes task-level meta-learning without an adaptation experiment.

No generation, training, inference, outcome inspection or new statistical endpoint was performed in this compatibility review. Only repository source and release metadata were inspected.
