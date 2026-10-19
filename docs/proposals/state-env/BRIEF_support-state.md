# Run brief: support-state — is the support expansion about a move, or about state?

Run id `support-state`. Repository `~/work/support-state` (branch from the fork's `origin/dan`, which has
the state environment; bring `support-curves`' harness in with
`git checkout origin/dan_support-curves -- support.py sc_*.py data/sc/` — it was never merged). Role:
executor. `AGENT_POLICY.md` governs. Budget **$10**, ceiling **20 pod-hours**. Authorised by Dan
2026-09-28. Read first: the `support-curves`, `support-followups` and `state-env` summaries in
`~/nd-rl/experiment-summaries/`, `STATE_ENV.md`, proposal 12.

## Question

`support-curves` found 29 transfer theorems the whole-proof base (`stage1_a1_seq_s0`) never solves in
400,000 attempts, which its EI descendant solves at p̂ 0.022–0.9995; `support-followups` found the EI
proofs are improbable under the base at two or three steps (a "new move"), and that neither 10⁷ attempts
nor 8× parameters removes it. `state-env` then showed a state-conditioned base reaches 5–6× more transfer
theorems with no RL. So: **does a base that sees the proof state already reach the 29?** If it does, the
"new move" was a move the whole-proof model could not *track* its way to, and the support expansion was
about state, not capability. If not, the expansion survives a much stronger base.

## Design (a suggestion)

The `support-curves` protocol, with the state models: base = `state-env`'s SN-v2 Stage-1 s0 (and s1 if
the budget allows), EI = its T1 round-8 checkpoint, sampled in the environment (`Env(assign=True)`).
1. On the **29 survivors** first: the SN base at up to 200,000 attempts per temperature (T 0.8, T 1.0),
   stopping a theorem at 5 successes. This is the headline.
2. Then the **383-theorem** stage-1 comparison (base vs its EI, k = 10,000, T 0.8), the forward and reverse
   crux, and the per-theorem log p_base vs log p_EI scatter — the elicitation curve for the state model,
   next to the whole-proof one.
3. If a survivor is reached: compare the SN base's proof with the whole-proof EI proof at the step that was
   improbable (`support-followups` part D's worst step).

Lean alone decides; set `LEAN_GATE_DUMP`; label every number with model and md5.

## Pre-registration (before the first pod)

Expected number of the 29 the state base reaches; the falsifier for "the expansion is about state": the
SN base reaches ≥ 15 of the 29 within 200,000 attempts per temperature.

## Deliverables

`preregistration/support-state.md`; `run_support_state.md` (≤ 400 words + the scatter next to
`support-curves`'); `numbers.md` § support-state; `log.md`; `artifacts/ss/summary.json`; `STATUS.md` line
`SUPPORT-STATE DONE <UTC>`; bucket `hf://buckets/dan-pandori/nd-rl/support-state/artifacts`; pods
deleted; `touch ~/runs/support-state/executor.done`.
