---
written_on: 2026-09-20
written_by: agent:codex
---

# Post-hoc emitted-proof-length audit

This audit answers a narrower question than the original generating/reference-length plots: how
long were the accepted proofs the models actually wrote?

The saved final discovery buffers were replayed through the current dependency/reiteration
compressor in `code/experiments/current/r1_cot/vendor/proof_diagnostics.py`. All records below
passed its verifier path. The pruner keeps every prompted premise, traces dependencies from the
final line, removes unused lines, and re-verifies the result.

| Condition | Direct proof-training cap | Final dependency-pruned tail beyond cap | Maximum | Count at maximum |
| --- | ---: | --- | ---: | ---: |
| Cap-6 offset, three EI rounds | 6 | 103×7, 7×8 | 8 | 7 |
| Cap-8 offset, seed 0, five EI rounds | 8 | 213×9, 94×10, 23×11, 2×12 | 12 | 2 |
| Cap-8 offset, seed 1, five EI rounds | 8 | 206×9, 79×10, 18×11, 2×12 | 12 | 2 |
| Cap-8, 8 layers, three hindsight-EI rounds | 8 | 218×9, 112×10, 40×11, 3×12, 1×13 | 13 | 1 |
| Cap-8, 16 layers, three hindsight-EI rounds | 8 | 213×9, 103×10, 32×11, 3×12, 2×13 | 13 | 2 |

For the cap-6 endpoint, the original raw histogram was 105×7 and 7×8; dependency pruning shortens
two of the seven-line certificates but leaves all seven eight-line certificates intact.
The [exact seven eight-line discovery records](results/cap6_eight_line_witnesses.json) retain the
source path, source-commit identity, and SHA-256 of the original round-three discovery buffer.

## Interpretation limits

These are verified written-length frontiers, not minimum-proof-length results. The compressor does
not search for alternate proofs, and the original study notes that one emitted 12-line certificate
can be manually replaced by an eight-line proof. The counts establish that the models generated
certificates longer than their direct supervision cap; they do not establish that the corresponding
theorems require that many lines.
