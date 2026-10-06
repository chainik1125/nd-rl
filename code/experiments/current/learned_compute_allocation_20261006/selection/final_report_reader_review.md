---
written_on: 2026-10-06
written_by: agent:codex
status: completed
---

# Reader review of the rewritten sprint report

Scope: read the main reader report and its selection, staged-stopping and continuation figures, without models, new experiments or source edits. I have substantial prior sprint context, so this is a cold-reader-style clarity check, not a genuinely independent fresh-context audit. No provenance-bound source was formatted or changed.

The report now answers the central question clearly: a separately learned confidence pipeline makes independent samples useful, but joint retry training does not beat returning the same model's first proof. The strongest fresh-pool baseline and strongest historical baseline are present. The distinction between generator improvement, proof recognition, allocation and policy-relative continuation harm is unusually clear. The final section appropriately says broader adaptation across tasks was not demonstrated and specifies the extra experiment needed. No material result or best-baseline omission was found.

Three small changes would make the executive conclusion harder to misread:

1. Add one direct sentence after the three executive findings: “This is not yet one proof-generating model that learns to recognize when it is done, and it is not a demonstration of meta-learning across tasks.” The report says this later; the user's main aim warrants the sentence up front. The successful threshold is a separately fitted deployment rule over a learned score, not an internally learned stopping decision of the original generator.
2. In the selection figure caption, remove “and the frozen head is one deterministic fit,” or explicitly say it is in the table only. The figure shows only three neural methods; the otherwise correct sentence invites the reader to search for a missing fourth row.
3. Distinguish the proposed frozen-backbone 385-parameter head reranking from earlier related full-BCE reranking. Artifact inspection confirms the exact proposed head diagnostic is unrun; the prior 32-theorem pilots instead used fully fine-tuned BCE seed0 epoch4 on no-cost seed71 and seed101 saved paths at counts 1/2/4/8. Thus “an unrun frozen-head reranking prerequisite, informed by earlier full-BCE pilots” is accurate; my initial suggestion to call it an extension without that distinction was too strong.

Figure checks: selection has its 400-theorem denominator, seed points, probability/percentage-point units and paired-interval meaning. The staged figure needs the adjacent caption to define BCE, the 400-theorem denominator and the cost; that caption supplies them. Its dashed lines could be mistaken for a continuous achievable frontier, but the explicit discrete-policy and within-scorer limitations prevent this. The continuation figure is self-contained, separates 64-theorem primary and common-16 secondary panels, defines probability mass, and explicitly disclaims horizon significance comparisons. No numerical replot is needed.

Optional plain-language refinement: define “formula skeleton” once as logical connective structure after removing atom names. “Incumbent” later could become “best proof already retained.” The resource and reproducibility sections are precise; their implementation details are appropriately separated from the main claims.

The frozen-head scope is careful: original weights are preserved, scoring still needs the full backbone, and proximity to one full critic is explicitly not a noninferiority result. Staged results disclose development-budget matching and test residuals. Historical results acknowledge the stronger syntax-plus-likelihood baseline and work mismatch. These qualifications should remain in the final report.
