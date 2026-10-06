---
written_on: 2026-10-06
written_by: agent:codex
---

# Final selection v2 and serial queue review

Accepted for protocol freeze; no load-bearing issue found in this bounded review. This is source/inventory and CPU guard validation, not evidence that the forthcoming actual final registry or results are correct. No real TEST outcomes were opened and no GPU work was launched.

Reviewed source hashes:

- `selection/compare_frozen_head.py`: `4237e75c404cfedb25a999704354b861c0c5f8702dfa5920824dc6115858c443`.
- `selection/run_frozen_test_queue.py`: `806ee26742c7c56202452a6d899b8f966fb7c5d69515eb2e6dc2f3d2a9b33e6f`.
- `allocation/run_final_eval_queue.py`: `9972924e77dbb25de2aa03beac4f85676e0b2b5510c3b12c235bbc2f81729297`.

The frozen-head comparison binds the training extractor snapshot and current source, compares the ASTs of `public_features`, `prepare`, and `feature_scores`, checks feature order and environment source, and independently recomputes frozen-feature scores before using reported accuracy. Its TEST registration checks precede private outcome loading. The current comparison hash is registered in the actual `freeze_inputs/selection_v2/main_entries.json`. This supplements the previously reviewed v3 exact vector/logit equivalence and v4 unchanged normal extraction branch; it does not claim a new v4 GPU extraction.

The selection queue preflights all 37 source/model signatures and registered public inputs before any inference, requires the registered launcher hash, rejects existing outputs, and records immutable registry/entries hashes, commands, source snapshots, process IDs, and timing. Actual v2 entries contain 37 distinct commands: nine trained models across three permutations, three base readouts across three permutations, and the frozen head. Its GPU occupancy check is initial only; ongoing exclusive GPU scheduling remains a root coordination responsibility.

The allocation wrapper waits for all nine MAIN live-model completion ledgers bound to the MAIN registry before scheduling its historical jobs. It only reads launch status and checks summary-file existence at this barrier. The underlying registered launcher remains responsible for each job's detailed model/data/method validation. This is a barrier for the nine live evaluations, not a claim that all selector and causal analyses have completed.

Independent checks:

- `code/.venv/bin/python -m pytest -q code/tests/test_lca_frozen_head.py code/tests/test_lca_selection_protocol.py code/tests/test_lca_selection_aggregation.py code/tests/test_evaluation_freeze.py code/tests/test_lca_public_seal.py`: 26 passed.
- [audit_final_queues.py](audit_final_queues.py): actual v2 command uniqueness; synthetic missing completion for each of the nine models blocks historical scheduling; wrong registry and failed launch reject. All passed.

The repository's prescribed `docs/NOTE_TEMPLATE.md` was absent when checked; this note follows the existing dated review-note format.
