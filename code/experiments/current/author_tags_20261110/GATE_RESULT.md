---
written_on: 2026-11-10
written_by: agent:claude (orchestrator), for Leon
---

# Gate result: the tagged bases almost ignore the tag

Base read of the three tagged bases (`stage1_tagAB_s{0,1,2}`) under tag A, tag B and no tag, run
8:37–9:11 pm PT on 2026-11-10 (serrano GPU 0, 0.56 GPU-h). Rule and measures are in `PREREG.md`,
frozen before any base existed. Numbers below are observed; the last section is interpretation.

## Manipulation check (primary, pre-registered): fails on 3 of 3 seeds

`tag_check.py`, teacher-forced on reference proofs of theorems the bases never saw: 1,550 that
need the route (`sigma_unseen`, 1,613 route-opening actions) and 1,550 length-matched controls
(`ctrl_unseen`). Mean natural log P per action.

| seed | open, tag A | open, tag B | open, no tag | Δ_open (A − B) | Δ_ctrl (all actions, A − B) | Δ_open − Δ_ctrl | passes (≥ log 2 = 0.693) |
|---|---|---|---|---|---|---|---|
| 0 | −3.820 | −4.128 | −4.723 | +0.308 | −0.001 | +0.309 | no |
| 1 | −6.129 | −6.219 | −6.127 | +0.090 | +0.000 | +0.090 | no |
| 2 | −3.867 | −3.945 | −3.802 | +0.079 | +0.000 | +0.078 | no |

The route opening is 1.36, 1.09 and 1.08 times as likely under A as under B. The rule asked for 2.
By the frozen rule the bases "ignore the tag", and the RL arms as designed would be
uninterpretable for the question (tag B would be a 1 % base with an inert token).

## Propensity and capability under each tag (`tag_gate_summary.py`)

k 32 unforced and 16 forced attempts per theorem, % of attempts, mean over theorems; A − B with a
95 % paired bootstrap over theorems.

The 106 theorems that require the route:

| measure | seed | A | B | no tag | A − B |
|---|---|---|---|---|---|
| uses reductio anywhere | 0 | 1.44 | 0.88 | 0.18 | +0.56 [+0.06, +1.06] |
| | 1 | 1.77 | 1.71 | 1.09 | +0.06 [−0.44, +0.56] |
| | 2 | 0.80 | 0.80 | 0.97 | +0.00 [−0.35, +0.32] |
| solved when the opening is forced | 0 | 4.17 | 3.30 | 1.81 | +0.86 [−0.16, +2.04] |
| | 1 | 0.94 | 0.63 | 0.86 | +0.31 [−0.39, +1.18] |
| | 2 | 1.81 | 1.57 | 1.81 | +0.24 [−0.55, +1.10] |
| log P of opening with reductio on the goal (exact) | 0 | −15.94 | −17.07 | −17.28 | +1.13 [+1.04, +1.22] |
| | 1 | −18.71 | −19.07 | −16.57 | +0.36 [+0.31, +0.41] |
| | 2 | −16.65 | −16.80 | −16.62 | +0.15 [+0.11, +0.18] |

The 240 other theorems:

| measure | seed | A | B | no tag | A − B |
|---|---|---|---|---|---|
| solved, unforced | 0 | 13.98 | 14.14 | 9.05 | −0.16 [−0.89, +0.59] |
| | 1 | 17.64 | 17.85 | 16.15 | −0.21 [−1.02, +0.57] |
| | 2 | 15.16 | 14.96 | 14.75 | +0.20 [−0.52, +0.95] |
| uses reductio anywhere | 0 | 3.82 | 2.98 | 1.85 | +0.83 [+0.34, +1.34] |
| | 1 | 3.07 | 3.10 | 3.12 | −0.03 [−0.36, +0.30] |
| | 2 | 2.47 | 2.28 | 2.28 | +0.20 [−0.05, +0.46] |
| solved when the opening is forced | 0 | 1.04 | 1.01 | 0.56 | +0.03 [−0.49, +0.52] |
| | 1 | 3.06 | 2.88 | 2.64 | +0.17 [−0.38, +0.80] |
| | 2 | 1.77 | 1.81 | 1.22 | −0.03 [−0.52, +0.42] |

- Unforced solving of the 106 is 0.00–0.12 % of attempts under every tag (not tabulated).
- Forced-reductio success under B is within 1 point of A on the 106 on every seed, so prediction
  P5 (within 5 points) holds; P1 (the check passes on 2 of 3 bases, 75 %) was wrong.
- Quality checks on the rows: 0 forced-first mismatches, 0 replay mismatches, 0 Lean-accepted
  proofs rejected by the verifier, for all 9 base-and-tag reads.
- The exact first-move arm logged "tag states checked 0". That counter belongs to the sampler,
  which the arm does not use; its inputs do carry the tag, since its values differ between tags.

## Interpretation (not observed)

The route taught under tag A is nearly as available under tag B: what the base learned from the
1,550 route records was not tied to the tag. A likely reason is that the tag was redundant. The
route records are 1.98 % of A's records, nothing else differs between the authors, and tag B never
appears on a theorem that needs the route, so the corpus holds no example of "B, in this
situation, does something else". Seed 0 shows a small real effect in the predicted direction; the
other two seeds show almost none.

Decision under the frozen rule: the RL arms (about 16 GPU-h) are not run on these bases.
