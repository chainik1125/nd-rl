---
written_on: 2026-11-03
written_by: agent:claude
---

# Amendment 2: the RL-pool control (Dan's rl_targets as the GRPO pool)

Written 2026-11-03 at about 08:40 UTC, after PREREG.md, Amendment 1 and the first results (every
arm, the knockout included, reached the bar within R ≈ 0.06 with our pool). Approved by the
coordinator. Its sha256 is recorded in `log.md` before the first GPU job it governs.

**Why.** 39.5 % of the L2 pool and 44.5 % of the L3 pool are strict σ-covered (Dan's rule:
classical-only and no ¬¬ in the sequent); 2.7 % of Dan's EI targets (`data/ladder/rl_targets.jsonl`,
123 of 4,495) are. The comparison with Dan's EI therefore confounds the algorithm with σ pressure.
This control changes only the pool.

- **Pool.** Dan's 4,495 rl_targets (`KO/pools/dan_rl_targets.jsonl`, md5 365450be, his registry's
  `data_md5` for the ladders' targets), as `--targets` in L2 and in L3 and as L3's
  `--targets_extra` (so the 0.33 extra share draws from the same file). Checked against the 106:
  0 renaming-class hits under canon_key, canon2 and the ndbench keys.
- **Recipe.** Identical to the main lines: L2 at fa28f6eb (39 iterations), L3 at f35d3daf with
  T_L3 364, bar 26.31, the snapshot grid, the read path (k 64, plain, B64, T 0.8, 512 / 96 / 10),
  the hold window (Amendment 1). Only the pool files change (`line.py --variant dp --pool_l2 …
  --pool_l3 … --pool_extra …`). `--lean_dump` keeps every distinct Lean-checked training sample
  (lean_gate's LEAN_GATE_DUMP, the GRPO process only).
- **Lines.**
  - `full_s0_dp` now (the reference: can Dan's pool elicit σ at all?).
  - Three ko-type lines `*_s0–2_dp`, on bases chosen by this rule:
    - ko_strict (Amendment 3) × our pool recovers (all three seeds reach the bar): run Dan's pool
      on the ko_strict bases;
    - ko_strict × our pool does not recover by the cap on any seed: run it on the current ko bases;
    - ko_strict still undecided when GPUs free up: start it on the ko_strict bases and tell the
      coordinator.
- **Reported.** E32 against R and the first / held crossings as for every line; and, from the dump,
  per iteration, the share of distinct Lean-accepted training proofs that are σ (Dan's classifier:
  DN of a NEGI line) and that use DN at all, so it shows whether the pool rewards σ. Dan's 2,471
  targets with ¬¬ in the sequent are expected to be solved by DN of a premise (not σ).
- **Machines.** As before: no third process on a line's GPU; serrano ≤ 2 of ours and ≤ 3 in total.
- **Budget.** With Amendment 3, at most about 95 GPU-h more (seven lines to the cap); stop and ask
  if the projection passes 180 GPU-h in all.
