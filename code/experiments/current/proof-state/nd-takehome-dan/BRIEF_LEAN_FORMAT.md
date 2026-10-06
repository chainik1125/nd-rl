# Run brief: lean-format — Lean as the training format for the from-scratch model (proposal 8)

Run id `lean-format`. Repository `~/work/lean-format` (branch `dan_lean_format`, a worktree of the
fork). `AGENT_POLICY.md` governs. The proposal is the design:
`~/nd-rl/docs/proposals/2026-09-20-lean-training-format.md` (branch `dan`; read it and the
decision note `~/nd-rl/docs/project_strategy/2026-09-20-lean-default.md`). Budget $30 of pods
(ask in `QUESTIONS.md` before exceeding); hard stop 36 h. **This run is the only exception to
the current pause** — launch nothing beyond it.

Before the first pod: `preregistration/lean-format.md` with the proposal's three predictions
as numbers (held-out within 2 pp of the token model; depth-3 f = 0 acquisition in 0.27–0.36;
transfer `L*` = 10), the mechanism you expect (named hypotheses vs line indices), the budget
and the stop rule; commit and push. Gate 0 checks its commit time against `~/pods.log`.

Pointers: `nd2lean.py` and `lean_prompts.py` (translator; agreement 181,464 / 181,464 with
`nd_verify`); the campaign's cap-6 sets and depth-3 f = 0 sets under `data/p2/`; the ladder's
transfer pool and pools under `data/ladder/` (copy from the `dan_ladder_a` branch or the
bucket `hf://buckets/dan-pandori/nd-rl/ladder-A/`); token-format results to compare against
are on file (`artifacts/p2/ei_depth3_f0_*`, the ladder's `la_T1_*`). Lean via `elan` (core, no
Mathlib) on the pod for reward checking; measure and report the throughput ratio against
`nd_verify`, and stop if a round exceeds 3× the token-format round time. Tokenise Lean one
symbol per token; report sequence-length statistics for both formats; **the model must be
identical in size** (4 layers, d = 256) and trained with the same schedule.

Deliverables: `run_lean_format.md` (≤ 400 words + figures; the decision-rule verdict stated
plainly: not worse on all three / worse, with the measured gaps), `numbers.md` and `log.md`
sections, bucket upload to `hf://buckets/dan-pandori/nd-rl/lean-format/`, `STATUS.md` ending
`LEAN-FORMAT DONE <UTC>`, then `touch ~/runs/lean-format/executor.done`. Pods deleted first.
