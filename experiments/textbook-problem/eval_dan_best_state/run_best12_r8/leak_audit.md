# Leak audit for eval_dan_best_state (run while seed 0 was in progress, 2026-10-02 ~16:20 UTC)

Trigger: seed 0 solved 76 of 104 candidate_v0 problems at least once (pre-registered band 50–70 %),
so the proposal's "above the bands" rule applies: check for a leak before any claim.

Pools the best-state cap-12 T1 r8 models were trained on, per `preregistration/best-state.md` and
`numbers.md` § best-state on Dan's `dan` branch (0c492902):
- Stage-1: `data/kh/train_k12.jsonl` (155,000 generator proofs, lengths 2–12), fetched from the bucket
  as `cap-horizon/data/kh/train_k12.jsonl.gz` (13,515,773 bytes); uncompressed md5 800b5486 matches
  Dan's record. 154,382 distinct renaming classes.
- T1 ladder: `rl_targets.jsonl` (3,000 records on Dan's branch, sha256 241951f5…; not the same class
  set as the pinned 4,495-record `rl_targets`), with replay from K12. Also audited: `rl_targets_v1`,
  `transfer`, `transfer_v1`, `heldout` from the same branch.

Method: renaming class by `canonical_theorem` or `canon_key` (never string match), each evaluated
set against each pool. The textbook 72 are counted in aggregate only.

| set | n | in a K12 class | in rl_targets | rl_targets_v1 | transfer | transfer_v1 | heldout |
|---|---|---|---|---|---|---|---|
| v0 | 104 | 2 | 1 | 0 | 0 | 0 | 0 |
| v1 | 80 | 0 | 1 | 0 | 0 | 1 | 0 |
| hand14 | 14 | 0 | 0 | 0 | 0 | 0 | 0 |
| batch3 | 21 | 0 | 0 | 0 | 0 | 0 | 0 |
| lean_v2 | 50 | 3 | 0 | 0 | 0 | 0 | 0 |
| derived_plain | 132 | 1 (no_dn) | 0 | 0 | 0 | 0 | 0 |
| textbook72 | 72 | 1 | 0 | 0 | 0 | 0 | 0 |

Verdict: overlap is at most 3 problems in any set and cannot account for 76 solves on v0. The result
stands as measured. The two v0 and one derived hits are not removed from the tables; they are named
here so a reader can discount them (names are in the input files; the 72 hit is not named).
