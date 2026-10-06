# STATUS — hand-off run (orchestration proposal, then ignition study)

Started 2026-09-16 18:29 UTC. Brief: HANDOFF_BRIEF.md. Previous: STATUS_campaign1.md, STATUS_followup.md.

## Plan
1. Task 1 (no pods): orchestration proposal at ~/nd-rl/docs/proposals/2026-09-16-automated-research-orchestration.md on branch dan_orchestration, linked from docs/STATE.md; draft PR, no reviewers. Target: done by ~20:30 UTC.
2. Task 2 (pods, ≤ 3 A40, ≤ ~$80 more): ignition study on dan_novelty → ignition.md, figures/ignition_*.png, numbers.md + log.md sections. Expected results written to log.md before any arm runs.

## Done
- 18:33 Task 1: proposal committed to nd-rl (82775b2, branch dan_orchestration), draft PR https://github.com/chainik1125/nd-rl/pull/3 (no reviewers).
- 18:37 Task 2 plan + pre-registered expectations committed (fbdde13) before the first pod was created (18:41).
- 18:41–21:58 Task 2 run: 16 Stage-1 models, 27 pre-RL samples, 16 EI arms, 33 intervention arms; everything pulled; pods deleted.
- 22:20 Task 2 written: ignition.md, figures/ignition_*.png (4), numbers.md §Ignition, log.md; self-check 0 verifier failures on 2,383 counted proofs.

## Running on pods
(none — p2 deleted 21:41, p3 21:55, p1 21:58 UTC; `runpodctl pod list` = []). Pod time: p1 3090 3.3 h ≈ $1.7; p2 A100 3.0 h ≈ $4.8; p3 A100 3.05 h ≈ $4.9; orphan ≈ $0.1 → ≈ $11.5 this run (≈ $30 cumulative of the $100 policy).

## Results (ignition.md)
- Reductio: ignition is the pre-RL base rate — 4 of 11 draws emit the pattern before RL and exactly those 4 ignite, in rate order; 7 zero-rate draws stay at 0/606. 1/(r·k·N) predicts the first-proof round within ~2.5×.
- Depth-3: base rate ≥ 2.5·10⁻⁴ → ignition by round 3; zero-rate draws split 3 ignite / 2 never (decided during training, not by sampling). Round-8 acquisition depends only on ignition (0.30–0.36 vs ≤ 0.004).
- Interventions: sibling transfer 11/11 ignite at once and reach the plateau; k = 128 and T = 1.0 only accelerate arms that would have ignited anyway (2/11 each), never the zero-rate ones.
- Base generalisation: depth-3 10/16 draws (CI 0.35–0.85), reductio 4/11 (0.11–0.69).
- Deviations: 3090 + 2×A100 instead of A40s; reductio candidates uncapped (7), depth-3 4; reductio s0 pre-RL sample = follow-up pass@10⁴ file; intervention (c) as one training step.

## Next step
None. A separate reviewer session should re-derive the counts from `artifacts/p2/ei_*` (found_1..8.jsonl kept), `artifacts/ign/cov_*.s0.jsonl` and `artifacts/ign/summary.json`.

DONE 2026-09-16T22:17:56Z
