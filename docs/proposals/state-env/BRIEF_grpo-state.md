# Run brief: grpo-state — GRPO in the proof-state environment, with support-expanding advantages (code phase)

Run id `grpo-state`. Repository `~/work/grpo-state` (branch `dan_grpo-state` from the fork's `origin/dan`, made by
`newrun`; starts after `compute-record` is recorded). Role: executor. `AGENT_POLICY.md` governs. Budget **$3**,
ceiling **6 pod-hours**. This is the **code phase only**: build it, test it, and write the experiment's
pre-registration and costing. The experiment itself (≈ $10–20) waits for Dan's RunPod top-up and is queued
separately. Authorised by Dan 2026-09-30. Claude quota is not the constraint: "if the only limiter is Claude quota,
I think we can start earlier".

Read first:
- nd-rl `docs/proposals/2026-09-28-after-state-env.md` item 3, and `docs/proposals/2026-09-29-literature-shortlist.md`
  item 3.
- `docs/literature/2026-09-29-lit-review/REVIEW.md` shortlist #3, and the notes on He et al. 2506.02355, Chen et al.
  2508.10751, and PKPO 2505.15201.
- The `run4-grpo-review` summary: GRPO acquired depth-3 in 40 / 40 surviving arms from zero-rate draws, against EI's
  6 / 8, on the token format.
- `grpo.py` (whole-proof; the advantage is at line 117), `state_ladder_ei.py`, `state_sample.py`, `state_train.py`.

## Why

The project's strongest "beyond the base" evidence is GRPO acquiring the depth-3 pattern from zero-rate draws. That
was on the token format, under `nd_verify`, and no checkpoint survives. The strongest models are now
state-environment policies, and step-level credit is the natural home for policy gradient. Two questions follow:
- Does GRPO in the state environment move the frontier or support differently from EI?
- Do advantages that reward *rare* correct proofs expand support rather than sharpen?

## What to build

- **`grpo_state.py`:** groups of G rollouts per theorem from the same start state. The reward is 1 if Lean accepts
  the literal proof, else 0. Every action in a rollout gets the rollout's advantage. The loss is the
  policy-gradient loss on the action tokens given the state, plus an optional KL to the reference.
- **Advantage variants** behind one flag:
  - default, R − mean;
  - unlikeliness rank penalty (He et al. §4.1, β_rank 0.25);
  - pass@k advantage (Chen et al.; PKPO);
  - distinct-proof bonus.

  Unit-test each variant on hand-made groups.
- **The same output conventions as the EI drivers:** rounds, found files, registry rows, and compute rows (use
  `compute-record`'s helper).
- **Tests:** CPU gradient-sanity and advantage tests in CI, a tiny CPU smoke run, and one GPU smoke of ≤ 30 min
  on the smallest pod.
- **Overlap with `search-expert`:** it also changes `state_sample.py` (per-action log-probs). Keep this run's
  changes in new files where you can, and merge `dan` often.

## The experiment (write it, don't run it)

`preregistration/grpo-state.md`, with the design, arms, seeds against `NOISE_FLOOR.md`, and the compute match to EI.
- **Arms:** EI, GRPO default, unlikeliness, and pass@k, on SN-cap12, 6 seeds.
- **Read-outs:** the `support-curves` protocol, and `transfer_long2` bins.
- Include a costed pod plan.

## Deliverables

- The code, merged into the fork's `dan` after review.
- `preregistration/grpo-state.md` (the experiment, for later), `run_grpo_state.md` (≤ 250 words: what was built and
  tested), `log.md`, and `STATUS.md` line `GRPO-STATE DONE <UTC>`.
- Pods deleted; `touch ~/runs/grpo-state/executor.done`.

Never run `test_run_once.sh`. At most 4 concurrent processes on the VPS.
