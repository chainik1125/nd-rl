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

# Tie-strength pilot (PREREG Amendment 1): the tag binds from a 10 % tie

Added 2026-11-10, 11:22 pm PT. One base per strength, seed 0, same recipe and step count as the
first seed-0 base; pretrained 10:10–11:12 pm on serrano GPUs 0 and 3 (1.5 GPU-h), checked with the
same `tag_check.py` (`tag_gate.py` prints the table). Observed:

| tie p (route share of tag A's records) | A records | open, A | open, B | open, no tag | Δ_open − Δ_ctrl | A vs B | check | sanity sample, Lean-accepted of 256 (A / B / no tag) |
|---|---|---|---|---|---|---|---|---|
| 0.0198 (first base, seed 0) | 78,188 | −3.820 | −4.128 | −4.723 | +0.309 | 1.36× | fails | 218–229 under each |
| 0.10 | 15,500 | −3.127 | −6.416 | −4.964 | +3.294 | 27× | passes | 223 / 226 / 223 |
| 0.50 | 3,100 | −1.241 | −10.217 | −9.887 | +9.149 | about 9,400× | passes | 201 / 225 / 220 |
| 1.00 | 1,550 | −1.138 | −11.182 | −10.395 | +10.461 | about 35,000× | passes | 144 / 220 / 226 |

- Under tag B the route opening becomes less likely than on the first base (−4.13 → −6.42 →
  −10.22 → −11.18): the base learns that B does not take the route, on theorems it never saw.
- Ordinary proving under tag A: unchanged at p 0.10 (Δ_ctrl −0.005; sample 223 of 256), lower at
  0.50 (Δ_ctrl −0.173; 201) and much lower at 1.00 (Δ_ctrl −0.417; 144), where A is never seen on
  an ordinary proof.
- With no tag the base behaves between A and B at p 0.10 and like B at 0.50 and 1.00.
- Predictions (Amendment 1): pass at p10 35 %, p50 70 %, p100 85 %; all three passed.

By the amendment's rule the weakest passing tie, p 0.10, is the base for the next step: seeds 1
and 2, the full base read under A, B and no tag, and then, with Leon's go, the RL arms. One seed
per strength: the size of the effect at each p is not established, only that it is far above the
bar. The check measures propensity along reference proofs; whether capability (finishing when the
opening is forced) is gated too is what the full read measures.

# The 10 % tie on three seeds: the gate passes; propensity is gated, forced finishing is not different

Added 2026-11-11, 1:12 am PT. Bases `stage1_tagAB_p10_s{0,1,2}`; full read under A, B and no tag
(serrano GPU 3, 12:45–1:11 am, 0.44 GPU-h; a first attempt at 12:01 am was ended after 2 minutes by
the never-share gate). Rows in `AT/reads/pc_p10/`; `tag_gate.py` and
`tag_gate_summary.py --variant p10` print the numbers. Observed:

**Manipulation check (the frozen rule): passes on 3 of 3 seeds.**

| seed | open, A | open, B | open, no tag | Δ_open − Δ_ctrl | A vs B |
|---|---|---|---|---|---|
| 0 | −3.127 | −6.416 | −4.964 | +3.294 | 27× |
| 1 | −4.662 | −8.325 | −7.708 | +3.665 | 39× |
| 2 | −3.024 | −6.172 | −5.825 | +3.155 | 23× |

**Propensity and capability, % of attempts** (k 32 unforced, 16 forced; A − B with a 95 % paired
bootstrap over theorems):

| set | measure | seed | A | B | no tag | A − B |
|---|---|---|---|---|---|---|
| the 106 | uses reductio anywhere | 0 | 5.19 | 0.03 | 0.24 | +5.16 [+3.24, +7.34] |
| | | 1 | 4.95 | 0.68 | 1.12 | +4.27 [+2.24, +6.60] |
| | | 2 | 5.07 | 0.06 | 0.00 | +5.01 [+3.01, +7.31] |
| the 106 | solved when the opening is forced | 0 | 0.55 | 0.79 | 0.63 | −0.24 [−0.94, +0.39] |
| | | 1 | 0.63 | 0.71 | 0.71 | −0.08 [−0.71, +0.55] |
| | | 2 | 1.42 | 1.18 | 0.94 | +0.24 [−0.47, +0.94] |
| the 240 | uses reductio anywhere | 0 | 7.88 | 2.08 | 1.86 | +5.79 [+4.28, +7.46] |
| | | 1 | 4.52 | 2.63 | 2.34 | +1.89 [+1.08, +2.84] |
| | | 2 | 4.61 | 1.78 | 1.09 | +2.83 [+1.90, +3.83] |
| the 240 | solved, unforced | 0 | 13.52 | 12.28 | 10.53 | +1.24 [+0.16, +2.32] |
| | | 1 | 14.88 | 15.56 | 13.79 | −0.68 [−1.43, +0.08] |
| | | 2 | 13.46 | 13.15 | 12.70 | +0.31 [−0.42, +1.02] |
| the 240 | solved when the opening is forced | 0 | 1.98 | 1.28 | 1.25 | +0.69 [−0.03, +1.49] |
| | | 1 | 1.04 | 1.01 | 0.94 | +0.03 [−0.28, +0.38] |
| | | 2 | 1.28 | 1.08 | 1.49 | +0.21 [−0.24, +0.73] |

- Unforced solving of the 106 is 0 % under every tag except seed 2 under A (0.80 %).
- The exact first-move log-probability of opening with reductio on the goal is higher under A on
  every seed (the 106: +5.87, +1.53, +7.91 nats).
- QC: 0 forced-first and 0 replay mismatches in all 9 reads; 1 Lean-accepted proof rejected by the
  verifier (seed 0, no tag).

**Interpretation.** Propensity is gated by the tag: under B the model uses a reductio on the 106
in 0.03–0.68 % of attempts against about 5 % under A, and no tag behaves like B. Finishing when
the opening is forced does not differ between tags, but it is about 1 % under both, so this arm is
near its floor and says little about whether the capability itself is gated. (The forced opening
is a reductio on the goal; most required reductios are on a subgoal.) The RL arms are the test:
RL under B from these bases, against RL under A.
