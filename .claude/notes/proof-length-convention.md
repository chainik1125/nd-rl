---
written_on: 2026-10-14
---

# Proof length: body lines are the system of record (Robbie, 2026-10-14)

**Decision.** From 2026-10-14, proof length means **body lines**: lines after the premise block,
which is the ND proof's line count minus its premise count. Use it for the pretraining cap
("≤ 6 body lines"), RL target bands, eval bins and certificates. Main tables also show the group's
measure, so old results stay comparable.

**The group's measure is different.** Everyone else counts **nd_verify lines**, which include one
line per premise (`N1 F : PR`). The group is consistent on this; checked 2026-10-14:

| who | where | evidence |
|---|---|---|
| Dan | the cap-6 file `data/train.jsonl` (155k), ladder transfer and RL-target pools | `n_lines` includes the premises: `train_0` is 1 premise + 1 ORI line, `n_lines: 2` |
| Charles | release 2026-10-09, e.g. `k12_long16_10` "13–16 lines" | `n_lines` equals the line count of the proof, premises included, for `n_prem` 0–5 |
| Robbie | take-home, pretraining-ablations and combined-model posts; harness `ge[7]`, L*, the 792 | take-home `gen.py` takes `n_lines` from `verify_text`; `by_length.py` and `harness.py` bin on `n_lines` |

Leon's numbers and Dmitry's textbook "1–33 lines" were not checked.

**Converting.** body = `n_lines` − `n_prem`, because the premise block is mandatory. Minimality is
unaffected: the minimal total is n_prem + the minimal body. The old pools have 0–3 premises, so old
lengths are inflated by 0–3 lines: dev "7–14 lines" is body 4–13.

**Why body lines.**
- Premise count is now its own axis: the RL pool has 3–10 premises.
- Distractor-premise augmentation adds premises without adding difficulty.
- Under nd_verify lines, "cap 6" would change meaning as soon as distractors are added.

**Lean lines are a third measure.** In `lean_seq` every premise is written as
`have nK : F := hK ;`, but a box hypothesis is a `fun` binder, not a `have`. So Lean `have` lines =
nd_verify lines − boxes. The old charts are close to Lean lines, not to body lines.

Related: proofs are checked with Lean only, never nd_verify (CLAUDE.local.md).
