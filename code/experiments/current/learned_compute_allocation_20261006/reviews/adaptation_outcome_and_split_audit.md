---
written_on: 2026-10-06
written_by: agent:codex
status: preliminary_review
---

# Interface adaptation restores continuation with a clean-state cost

## Independent quantitative audit

All six conditions use the same 48 development theorem IDs and four candidate IDs per theorem. Input hash, random seed, batch, temperature, action/step bounds, and context cap agree. The adapted checkpoint's saved SHA256 matches its file. AST comparison of the original/adapted collector snapshots finds only added `zip(strict=True)` guards in sampling and likelihood functions, with no changed sampling algorithm.

| Condition | Original checkpoint | Adapted checkpoint | Adapted minus original, theorem-bootstrap 95% interval |
|---|---:|---:|---:|
| No previous proof | 180/192 | 156/192 | −12.50 pp [−20.83, −4.69] |
| One previous proof | 5/192 | 157/192 | +79.17 pp [70.31, 87.50] |
| Four previous proofs | 0/192 | 133/192 | +69.27 pp [59.90, 78.65] |

The intervals come from 50,000 independent bootstrap resamples of the 48 theorem-level means. Within the adapted checkpoint, four histories versus none gives −11.98 pp [−21.88, −1.56]; one history versus none gives +0.52 pp [−8.33, 9.90]. Twelve completed candidates, spanning both valid and invalid cases and all history counts, were independently rechecked in Lean with **zero disagreements**.

The clean-state regression is substantial in this panel and should accompany the restoration claim, rather than being described as negligible retention loss. The intervention improves the broken prefixed interface while changing original-state proof competence. One adaptation seed, repeated use of development data, and dynamically changing sampler batches limit generalization; these are exploratory paired uncertainty estimates, not confirmatory post-selection guarantees. Equal initial seeds do not imply identical tokenwise random draws after trajectories diverge.

The adapted probe's immutable configuration still says `unadapted frozen generator` in its scope field. That label is stale: the generator is adapted and then frozen. The checkpoint hash/path correctly identifies it. Interpret results from actual checkpoint provenance rather than that copied wording.

## Main split audit

An independent metadata-only calculation verifies 1,600 training, 320 development, and 400 final-test theorem records, each split balanced equally between forward-generated and ORE2 families. Canonical atom-bijection/premise-permutation keys, atom-erased skeletons, theorem IDs, split assignments, file hashes, code hashes, and the exposure-cache hash all match the manifest. There are zero canonical or skeleton overlaps between the three splits and zero matches against the declared training cache, public benchmark exclusions, and prior pilot cases.

Only theorem metadata and file hashes were used for this audit. No final-test candidate correctness labels, proof fields, or protected hidden benchmark files were inspected. The exposure check establishes consistency with the declared cache; it does not independently recover all original checkpoint training data.

ORE2 examples share the same two-branch disjunction-elimination proof schema despite distinct formula skeletons. Therefore the split supports formula-shape holdout, not unseen deduction-schema or proof-program-family transfer. Limit structural generalization claims accordingly.

## Evidence

- [Adaptation audit JSON](adaptation_independent_audit.json) and [reproduction script](audit_adaptation.py).
- [Main metadata audit JSON](main_metadata_independent_audit.json) and [metadata-only script](audit_main_metadata.py).
- Input artifact roots: `context/base_dev48_v1`, `context/format200_dev48_v1`, and `data/main_v1` under `artifacts/learned_compute_allocation_20261006/`.
