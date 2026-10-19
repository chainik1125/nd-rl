# Run brief: mcts-a — build PUCT search with a learned value, and test it at test time against sampling

Run id `mcts-a`. Repository `~/work/mcts-a` (branch `dan_mcts-a` from the fork's `origin/dan`, made by `newrun`). Role:
executor. `AGENT_POLICY.md` governs, including per-arm compute rows. Budget **$15**, ceiling **30 pod-hours**. Authorised by
Dan 2026-10-02. Proposal: nd-rl `docs/proposals/2026-10-02-mcts.md` (proposal 22), **Phases 0 and A**. Phase B is the
separate run `mcts-b`, which only proceeds if this run's gate passes.

Read first:
- The proposal.
- The `search-expert` summary: its best-first expert used about ¼ of its budget, mostly because batching was poor.
- The `trajectory`, `trajectory-cap6` and `best-state` summaries.
- The AlphaProof Methods extract in proposal 20 (`docs/proposals/2026-10-02-exploration.md`).
- HTPS notes in `docs/literature/2026-09-29-lit-review/notes/htps-lample2022.md`.

**Code not on `dan`:**
- `state_search.py` (search-expert's best-first search) is only on `origin/dan_search-expert`.
- `trajectory`'s `tj_*.py` and `pod/tj/` are only on `origin/dan_trajectory`.

Take what you reuse with `git checkout <branch> -- <files>`.

## Model

`trajectory`'s cap-12 seeds s0–s2: Robbie's recipe, ALiBiGPT 9.56M, `lean_staten`, K12. Use both the end-of-pretraining
checkpoint (`stage1_best12_s{0,1,2}_b1200.pt`) and the r8 checkpoint (`la_T1_best12_s{0,1,2}_r8.pt`), from
`hf://buckets/dan-pandori/nd-rl/trajectory/ckpts/tj/`. `trajectory`'s per-seed groups A, B and C are in its artefacts.

## Phase 0: build

1. **PUCT search over proof states** in the state environment.
   - Prior π^{1/τ}.
   - AlphaProof's exploration factor, c(s) = c_init + log((N(s) + c_base + 1) / c_base).
   - K actions sampled per expansion, deduplicated by the resulting state.
   - **Progressive sampling:** K more actions at nodes where n(s) ≤ C·N(s)^α.
   - Invalid actions are discarded by the environment. Lean decides complete proofs.
   - **Batch leaf evaluations across many trees** so the GPU stays busy. Report GPU utilisation; `search-expert`
     lost most of its budget here.
2. **A value head** on the policy trunk.
   - Predict γ-discounted steps-to-go (AlphaProof: Q = γ^{−V−1}) plus a "solvable" logit.
   - **Train it with the trunk frozen,** so the policy is unchanged.
   - **Data:** states from rollouts on `rl_targets.jsonl` and the generator pool, never textbook72, holdout250 or
     any transfer pool. States on accepted proofs get their true steps-to-go; states on failed attempts are
     labelled "unsolved".
   - Report calibration on held-out targets.
3. **Tests:** CPU unit tests (search on toy theorems, deduplication, the value-head round trip), plus one GPU
   smoke test of ≤ 20 min.

## Phase A: test-time search, no policy training

At **equal GPU-seconds per theorem** (from the compute rows), compare three arms:
- **plain sampling:** `trajectory`'s k 256 protocol, or re-sample at matched GPU time;
- **PUCT without the value:** the prior plus progressive sampling (proposal 20's E1);
- **PUCT with the value.**

Run them on textbook72, holdout250, **group C** (per seed) and the long pools (`rr600` 13–16, `transfer_long2`), from
the end-of-pretraining and r8 checkpoints of all three seeds.

**Gate, pre-registered before the first pod:** PUCT with the value solves more of group C than plain sampling at
matched GPU time, beyond the seed spread, on ≥ 2 of 3 seeds, at r8.
- Write the gate's verdict into `STATUS.md` as `MCTS-A GATE: PASS` or `MCTS-A GATE: FAIL`, with the numbers.
- `mcts-b` reads it.

## Read-outs

- Solved per arm, pool, checkpoint and seed (group C is the headline).
- Value calibration (predicted against actual steps-to-go).
- Search statistics: nodes per proof, where in the tree the hard steps sit, and GPU utilisation.
- Compute per arm.

## Deliverables

- The search and value code, merged into the fork's `dan` after review.
- `preregistration/mcts-a.md`, `run_mcts_a.md` (≤ 350 words + table), `numbers.md` § mcts-a, `log.md`, `STATUS.md`
  lines `MCTS-A GATE: …` and `MCTS-A DONE <UTC>`.
- Bucket: `hf://buckets/dan-pandori/nd-rl/mcts-a/{ckpts,artifacts}`. Pods deleted; `touch ~/runs/mcts-a/executor.done`.

Never train or tune on textbook72 or holdout250. Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
