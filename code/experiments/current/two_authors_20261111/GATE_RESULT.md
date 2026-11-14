---
written_on: 2026-11-14
written_by: agent:claude (orchestrator), for Leon
---

# Two authors: gate result (seed-0 bases, 2026-11-14)

Rules: `PREREG.md` (frozen 2026-11-11, amendment 1 of 2026-11-13). Full output: `authors_gate.py --authors
authors_<v>_s0 --control control_<v>_s0`; JSON in `TA/gate_{half,p10}_s0.json`. Points with 95 % bootstrap intervals
over theorems. Ran on two Runpod A40 pods (`TA/pods.tsv`), both deleted after md5-checked fetches.

## Observed

| gate | rule | `half` | `p10` |
|---|---|---|---|
| G1 step order on `marker_plain`, tag A − tag B | ≥ 50 | +51.2 [47.2, 55.3] (A 56.7 %, B 5.5 %): pass | +26.6 [22.9, 30.7] (A 31.7 %, B 5.1 %): fail |
| G2 untagged bundle excess (pooled); minus control | ≥ 10, difference above 0 | +9.1 [7.4, 10.8]; +8.4 [6.4, 10.5]: fail on the 10-point bar | +3.7 [2.9, 4.5]; +3.5 [2.7, 4.3]: fail |
| G3 untagged step order, route − non-route proofs, authors − control (pooled) | ≥ 10, interval above 0 | +15.7 [−16.9, 61.1], 46 theorems: fail | +6.4 [−21.2, 40.0], 28 theorems: fail |

- Controls: tag effect within ±1 point on every habit, bundle excess +0.7 (`half`) and +0.2 (`p10`).
- `half`, untagged: φ(step order, reiteration) 0.31 [0.25, 0.36] against −0.02 in the control; conjunct order does
  not join (φ about 0). On `marker_plain` alone the excess is +8.4 [6.5, 10.3].
- Route against reiteration (not the pre-registered marker): A-form reiteration in route proofs minus elsewhere,
  authors − control, +22.7 [7.0, 38.4] (`half`) and +18.8 [10.2, 28.9] (`p10`).
- `p10`, untagged: the default is author B (88 % of two-habit proofs on `marker_plain` are all-B), so there is little
  variation to bundle; on `route_unseen` the excess is +10.0 [7.8, 12.1].

## Decision under the pre-registration

Neither version passes G2, so the rule is: stop, no RL stage. Predictions: P1 (G1 passes on both) wrong, P2 (G2 at
`half`) wrong by about a point, P3 (G3 at `half` fails) and P4 (G3 at `p10` passes) wrong.

## Interpretation (orchestrator's)

- The models do hold a weak author without a tag at the even split: step order and reiteration go together, well
  above the control, but under the bar set in advance.
- The route is tied to reiteration, not to step order. The primary marker was the wrong one for G3.
- The 10 % version makes author A too rare to show up untagged on ordinary theorems.
