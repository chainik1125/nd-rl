---
written_on: 2026-09-24
written_by: agent:claude
status: proposed
---

# Proposal 11: the length horizon, and the error bar that tells us whether anything else is real

Proposal 10's three runs (`ds-composition`, `ds-generator`, `ds-rendering`) changed the training
set's composition, the generator's shape distribution and the Lean surface form. Eleven variants,
none of which replaced the control. Two things came out of them that determine what is worth
running next.

**1. The only manipulation that moved the horizon was raising the training cap.** On
`targets_reductio_req`, where every target needs a derived `DN`, every cap-6 arm in
`ds-composition` wrote **zero** accepted proofs of ≥ 8 lines across both seeds and 2,000 attempts
per target; the cap-8 arm (A3) wrote 29 and 33, solved 54 / 72 targets against the control's
28 / 26, and reached a frozen ladder `L*` of 11 against the control's 9. Composition, generator
shape and rendering all failed to move the same quantities. `ds-generator`'s conclusion was the
same from the other side: G2 changed the training distribution about as far as the generator
allows and changed transfer readiness not at all.

**2. Most of the differences those runs reported are smaller than their own noise floor.**
`ds-generator`'s G1 was a deliberate null — the control's distribution by construction — and it
differed from the control by **+49 % on the frozen ladder and 1.9× on required@8 coverage**.
`ds-rendering`'s 26-model seed sweep found the depth-3 held-out cell is **bimodal** (20 of 26
models above 0.44 or below 0.11, pooled within-arm sd 0.302), so its observed R2 − C0 gap of
0.129 would need ≈ 87 seeds per arm to detect. No run in this project has ever measured its own
error bar; every one of them has reported two-seed differences as findings.

These two facts point at one run each, and the second gates the first.

## Run 1 — `cap-horizon`: is the horizon a property of the training cap?

If the cap sets the horizon, `L*` should track it upward and the maximum reductio length the model
will write should sit at about the cap. If instead `L*` plateaus while the cap rises, the wall is
the model or the expert-iteration procedure, and the cap was a red herring that happened to move
two points. Either answer is worth more than another dataset-style arm, and the second would
redirect the project.

`ds-composition`'s A3 confounded two changes at once: it added 7–8-line proofs *and* removed
8,857 proofs from every ≤ 6-line bin to hold the total at 155,000. This run separates them and
extends the axis to cap 10 and cap 12. It reuses A3's and the control's existing checkpoints from
the bucket rather than retraining them.

Note that cap 6 is the **take-home's** constraint, not the SPAR project's. Every arm above cap 6
is labelled as outside it, as A3 was, and none is proposed for adoption as a take-home submission.

## Run 2 — `noise-floor`: what difference is this project able to resolve?

Four independently drawn raw pools from the control's *exact* generator settings, assembled to the
control's exact composition, × two Stage-1 seeds. By construction every difference between them is
noise. The output is a table of resolvable differences — per quantity, the size below which an
n = 2 comparison means nothing — to be quoted by every future run and applied retrospectively to
the standing findings. It also closes two n = 1 gaps left by proposal 10 (`ds-generator`'s G1
frozen ladder seed 0, `ds-composition`'s single-seed ladder for C0 and A1).

This is the cheaper run and the more important one. If it says the frozen ladder cannot resolve
anything below ±50 %, then several results this project currently treats as findings are not
findings, and `cap-horizon`'s predicted effects (monotone, large) are among the few that would
survive.

## What is deliberately not being run

- **The rendering seed sweep at n ≈ 87** (`ds-rendering`'s own first recommendation, ≈ $8.50).
  Its run-level question — does rendering change RL readiness — was already answered *no* by a
  paired ranking over five arms. The remaining question is mechanistic and the reviewer's own
  framing was "or stop asking". Stopping asking.
- **The depth-3 dial.** Three independent runs now agree that EI's round-4 endpoint lands in a
  narrow band (0.38–0.44 in `ds-composition`, 0.396–0.436 in `ds-generator`, 0.387–0.420 in
  `ds-rendering`) whatever base rate it starts from, across frozen rates spanning 0.032 to 0.322.
  Another dial cell buys nothing; the finding is strong enough to state.
- **The `nd2lean.py` looseness fix.** Needed (see below), but it changes the checker, so it must
  not land while runs that depend on checker comparability are in flight. Queued for after.

## Standing checker caveat

All three proposal-10 runs independently found Lean-accepts-and-`nd_verify`-rejects at a non-zero,
one-directional, rendering-dependent rate (2,208 in 16.3 M pairs; 718 per million in
`ds-generator`, a 22× rise; R3 8.8× the control). **Lean never once caught what `nd_verify`
missed.** Two named classes: `nd2lean.py`'s `BOTE` → `.elim` resolving as `Not.elim` rather than
`False.elim`, and the Lean rendering not enforcing premise re-statement. No counted proof anywhere
is affected, because acceptance is the conjunction. But proposal 9's plan to retire `nd_verify`
for a Lean-only term-allowlist check **must close these two classes first**, and until then "Lean
is the checker of record" should be read as "Lean **and** `nd_verify`".

## Budget

Authorised 2026-09-24 by Dan: up to $60, with a hard floor of **$130 RunPod balance**. The balance
at authorisation was $177.80, so **the floor binds at ≈ $48**, which is what these two runs are
sized to: `cap-horizon` $22 / 44 pod-hours, `noise-floor` $18 / 36 pod-hours, ≈ $8 unallocated.
Both register a `podbudget` ceiling before their first pod, so `pod_budget_watch` (80 % warn,
100 % delete) covers them from the start — the guard that was missing during the 2026-09-22
incident that cost proposal 10 ≈ $76 in idle pods.
