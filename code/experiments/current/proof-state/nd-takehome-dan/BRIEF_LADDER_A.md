# Run brief: ladder-A — RL technique ladder, Phase A (rungs T1–T6)

Run id `ladder-A`. Proposal: `~/nd-rl/docs/proposals/2026-09-17-rl-technique-ladder.md` (branch
`dan_proposals`; read it in full — the metric section is binding). `AGENT_POLICY.md` governs.
Budget $50 of pods; hard stop 36 h. Work in your own worktree branch `dan_ladder_a` of the fork.

Order of work:
1. **Pools first, committed once.** Build the transfer pool (≥ 1,500 theorems, `L_true`
   resolved in 7–14 by `minlen.py`, mixing the strict long generator with the textbook-shaped
   schemata pool; never sampled in training) and the RL-target pool, disjoint by renaming class
   from each other, from every Stage-1 set and from validation-36. Record the prover bound and
   the unresolved fraction. Commit `data/ladder/` with a `POOLS.md` describing them.
2. **Pre-register** `preregistration/ladder-A.md`: for each rung T1–T6, the predicted `L*` on
   transfer and on RL targets, the mechanism in one line, and the falsifier; the common sample
   budget (256 samples per target); two seeds; frozen control. Commit before the first pod.
3. Run T1 (baseline EI) and its frozen control first — they define `L*_frozen` and the
   reference. Then T2–T6 in the order you expect to be most informative, two seeds each, all
   from the take-home's cap-6 `abs` Stage-1 model. Same total sample budget per rung.
4. After every rung update `ladder.md` (rung, `L*` transfer / targets, `L*_frozen`, solve rate
   by `L_true` bin with Wilson intervals, base-reachability split, cost) and
   `figures/ladder_*.png`. Counts normalised; every counted proof's base reachability recorded.
5. End: `run_ladder_A.md` (≤ 400 words), bucket upload to
   `hf://buckets/dan-pandori/nd-rl/ladder-A/`, `STATUS.md` line `LADDER-A DONE <UTC>`,
   `touch ~/runs/ladder-A/executor.done`. Pods deleted first. If Phase A shows a rung with
   `L* − L*_frozen ≥ 2` on transfer on both seeds, say so in `QUESTIONS.md` and ask Dan for
   Phase B.
