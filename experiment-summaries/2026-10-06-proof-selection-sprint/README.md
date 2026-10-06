---
written_on: 2026-10-06
written_by: agent:claude
---

# Proof selection and stopping sprint (2026-10-06)

Start with [`summary.md`](summary.md). Everything a reader needs is in this folder; code, raw result files and
freeze hashes stay in the working folder `code/experiments/current/proof_selection_20261006/`.

| Folder | Contents |
|---|---|
| `figures/` | Every figure from the sprint (the six in the summary plus per-phase and diagnostic plots). |
| `reports/` | Kickoff (`start.md`), data schema, timestamped research log, the two workers' phase reports (`worker_A_*`, `worker_B_*`), result tables, and `HEADLINE_SOURCES.md`, which maps every number in the summary to its source file. |
| `reviews/` | The four Codex audits: design, midpoint recomputation, write-up red team, final delta. |

Phase reports reference `results/` and `figures/` paths relative to the working folder; the figures are duplicated
here, the raw JSON results are not.
